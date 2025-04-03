# ag.py
import argparse
import fileinput
import sys
import os
import re
import subprocess
import requests
import time
import itertools
import sys
import threading # <-- Added
import queue     # <-- Added
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any # <-- Added Any


# --- Constants ---
DEFAULT_API_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_MODEL_TYPE = "openai" # <-- Added default model type
DEFAULT_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_FREQUENCY_PENALTY = 0.5
DEFAULT_PRESENCE_PENALTY = 0.5
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'
ANSI_SAVE_CURSOR = '\033[s'
ANSI_RESTORE_CURSOR = '\033[u'
ANSI_CLEAR_SCREEN = '\033[2J\033[H' # Clear screen + cursor home

# --- Helper Functions (fetch_webpage, execute_command, process_input_directives) ---
# ... (保持不变) ...
def fetch_webpage(url: str) -> Optional[str]:
    """Fetches and extracts text content from a URL."""
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "aside"]): # Remove common noise
            element.decompose()
        # Get text, strip leading/trailing whitespace, reduce multiple newlines
        text = soup.get_text(separator='\n', strip=True)
        return re.sub(r'\n\s*\n', '\n\n', text).strip()
    except requests.RequestException as e:
        sys.stderr.write(f"Error fetching {url}: {str(e)}\n")
        return None
    except Exception as e:
        sys.stderr.write(f"Error parsing {url}: {str(e)}\n")
        return None

def execute_command(command: str) -> str:
    """Executes a shell command and returns its output or error."""
    try:
        result = subprocess.run(command, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True) # Use text=True for automatic decoding
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_output = f"Error executing command: `{command}`\nExit Code: {e.returncode}\n"
        if e.stdout:
            error_output += f"STDOUT:\n{e.stdout}\n"
        if e.stderr:
            error_output += f"STDERR:\n{e.stderr}\n"
        return error_output
    except Exception as e:
        return f"Unexpected error executing command `{command}`: {str(e)}\n"


def process_input_directives(input_str: str) -> str:
    """Processes @file and @url directives in the input string."""
    processed_lines = []
    for line in input_str.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('@file'):
            match = re.match(r'@file\(([^)]+)\)', stripped_line)
            if match:
                filename = match.group(1).strip() # Strip spaces inside parens
                try:
                    with open(filename, 'r', encoding='utf-8') as f: # Specify encoding
                        processed_lines.append(f"--- Content of {filename} ---\n{f.read()}\n--- End of {filename} ---")
                except FileNotFoundError:
                    sys.stderr.write(f"Error: File not found: {filename}\n")
                except IOError as e:
                    sys.stderr.write(f"Error reading {filename}: {str(e)}\n")
                except Exception as e: # Catch other potential errors
                     sys.stderr.write(f"Unexpected error processing file {filename}: {str(e)}\n")
            else:
                 processed_lines.append(line) # Keep line if regex doesn't match format
        elif stripped_line.startswith('@url'):
            match = re.match(r'@url\(([^)]+)\)', stripped_line)
            if match:
                url = match.group(1).strip() # Strip spaces inside parens
                webpage_content = fetch_webpage(url)
                if webpage_content:
                    processed_lines.append(f"--- Content from {url} ---\n{webpage_content}\n--- End of {url} ---")
                # Error message handled within fetch_webpage
            else:
                 processed_lines.append(line) # Keep line if regex doesn't match format
        else:
            processed_lines.append(line)
    return '\n'.join(processed_lines)


# --- API Worker Thread ---

def _api_worker(
    q: queue.Queue,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    frequency_penalty: Optional[float],
    presence_penalty: Optional[float]
):
    """Worker thread function to call the API and put chunks/signals onto the queue."""
    try:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True
        }
        if frequency_penalty is not None: params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None: params["presence_penalty"] = presence_penalty

        stream = client.chat.completions.create(**params)

        for chunk in stream:
            q.put(('CHUNK', chunk)) # Put the actual chunk object

        q.put(('DONE', None)) # Signal normal completion

    except APIError as e:
        q.put(('ERROR', f"API Error: {str(e)}"))
    except Exception as e:
        q.put(('ERROR', f"An unexpected error occurred during streaming: {str(e)}"))


# --- Modified stream_response Function ---

def stream_response(client: OpenAI, model: str, model_type: str, messages: List[Dict[str, str]],
                    max_tokens: int, frequency_penalty: Optional[float],
                    presence_penalty: Optional[float], hide_reasoning: bool) -> Optional[str]:
    """Streams response, optionally hiding reasoning with a spinner, supports Gemini-style delay."""

    full_response_chunks = []
    processed_response_chunks = []
    buffer = '' # Buffer for standard <think> tag processing

    is_reasoning_deepseek = False   # Track if we are actively processing DeepSeek reasoning_content
    in_think_block_tag = False      # Track if we are inside a <think> tag block
    spinner_active = False          # Track if the spinner animation is currently shown
    spinner_chars = itertools.cycle(['|', '/', '-', '\\'])
    first_content_received = False  # Track if any content has been received (for Gemini mode)

    # --- Markers for non-hidden reasoning ---
    REASONING_START_MARKER = "--- Reasoning Content ---"
    REASONING_END_MARKER = "--- End Reasoning ---"

    def start_spinner():
        nonlocal spinner_active
        if not spinner_active:
            sys.stdout.write("Thinking... ")
            spinner_active = True
            sys.stdout.flush()

    def update_spinner():
        if spinner_active:
            spinner_char = next(spinner_chars)
            sys.stdout.write(f"\rThinking... {spinner_char}")
            sys.stdout.flush()

    def stop_spinner(success=True):
        nonlocal spinner_active
        if spinner_active:
            symbol = "✅" if success else "❌"
            # Clear the line before writing the final status
            sys.stdout.write("\r" + " " * 20 + "\r") # Adjust width if needed
            sys.stdout.write(f"Thinking... {symbol}\n") # Add newline after stopping
            spinner_active = False
            sys.stdout.flush()

    q = queue.Queue()
    worker_thread = threading.Thread(
        target=_api_worker,
        args=(q, client, model, messages, max_tokens, frequency_penalty, presence_penalty),
        daemon=True # Allows main thread to exit even if worker is stuck
    )

    worker_thread.start()

    # --- Main loop to process queue items ---
    final_status_success = True # Assume success unless error occurs
    try:
        # Start spinner immediately if hiding reasoning (covers Gemini case)
        if hide_reasoning:
            start_spinner()

        while True:
            try:
                # Wait for a short time for an item from the queue
                item_type, data = q.get(timeout=0.1)

                if item_type == 'CHUNK':
                    chunk = data
                    delta = chunk.choices[0].delta
                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    content = delta.content

                    # --- Stop Gemini spinner on first content ---
                    if model_type == 'gemini' and content and not first_content_received and spinner_active:
                         stop_spinner()
                         first_content_received = True # Mark that content has started

                    # --- Handle DeepSeek Reasoning ---
                    if reasoning_content:
                        if hide_reasoning:
                            if not spinner_active: start_spinner() # Start if not already (e.g., if content came first)
                            update_spinner()
                        else:
                            if spinner_active: stop_spinner() # Stop spinner if reasoning starts and we're not hiding
                            if not is_reasoning_deepseek:
                                is_reasoning_deepseek = True
                                sys.stdout.write(REASONING_START_MARKER + "\n")
                                sys.stdout.write(ANSI_GREEN)
                                sys.stdout.flush()
                            sys.stdout.write(reasoning_content)
                            sys.stdout.flush()
                        continue # Skip content processing for this chunk

                    # --- Handle Standard Content (including <think> tags) ---
                    if content:
                        # If DeepSeek reasoning was just happening, end it
                        if is_reasoning_deepseek:
                            is_reasoning_deepseek = False
                            if not hide_reasoning: # Only print markers if not hidden
                                sys.stdout.write(ANSI_RESET)
                                sys.stdout.write(REASONING_END_MARKER + "\n")
                                sys.stdout.flush()
                            # If hiding, the spinner might be stopped below or already stopped

                        # Append to full response history regardless of hiding
                        full_response_chunks.append(content)
                        buffer += content

                        # Process buffer for <think> tags and normal output
                        processed_chunk_for_buffer = ''
                        temp_buffer = buffer
                        buffer = ''
                        output_buffer = '' # Collect output for this chunk before printing

                        while temp_buffer:
                            if not in_think_block_tag:
                                start_pos = temp_buffer.find('<think>')
                                if start_pos == -1: # No more <think> tags in buffer
                                    # Stop spinner *before* printing normal content if it's active
                                    if spinner_active:
                                        stop_spinner()
                                    output_buffer += temp_buffer
                                    processed_chunk_for_buffer += temp_buffer
                                    temp_buffer = ''
                                else: # Found <think> tag start
                                    before_think = temp_buffer[:start_pos]
                                    # Stop spinner *before* printing normal content if it's active
                                    if spinner_active and before_think:
                                        stop_spinner()
                                    output_buffer += before_think
                                    processed_chunk_for_buffer += before_think

                                    in_think_block_tag = True
                                    temp_buffer = temp_buffer[start_pos + len('<think>'):]

                                    if hide_reasoning:
                                        if not spinner_active: start_spinner() # Start if not already active
                                        update_spinner()
                                    else:
                                        output_buffer += ANSI_GREEN # Start green for think block
                            else: # Inside a <think> block
                                end_pos = temp_buffer.find('</think>')
                                if end_pos == -1: # Still inside <think> block
                                    if hide_reasoning:
                                        update_spinner()
                                        # Consume the buffer content without adding to output/processed
                                        temp_buffer = ''
                                    else:
                                        output_buffer += temp_buffer # Print thinking content
                                        # Do not add thinking content to processed_response_chunks
                                        temp_buffer = ''
                                else: # Found </think> tag end
                                    think_content = temp_buffer[:end_pos]
                                    in_think_block_tag = False
                                    temp_buffer = temp_buffer[end_pos + len('</think>'):]

                                    if hide_reasoning:
                                        update_spinner() # Update one last time before potential stop
                                        # Don't stop spinner yet, wait for actual content or end of stream
                                    else:
                                        output_buffer += think_content # Print thinking content
                                        # Do not add thinking content to processed_response_chunks
                                        output_buffer += ANSI_RESET # End green for think block

                        # Write the collected output for this chunk
                        sys.stdout.write(output_buffer)
                        processed_response_chunks.append(processed_chunk_for_buffer) # Add non-thinking parts to history
                        sys.stdout.flush() # Flush after processing content buffer

                elif item_type == 'DONE':
                    break # Worker finished successfully

                elif item_type == 'ERROR':
                    final_status_success = False
                    stop_spinner(success=False)
                    if is_reasoning_deepseek or in_think_block_tag:
                        if not hide_reasoning: sys.stdout.write(ANSI_RESET)
                    sys.stdout.flush()
                    sys.stderr.write(f"\n{data}\n") # Print error message from worker
                    return None # Indicate error

            except queue.Empty:
                # Queue is empty, update spinner if active
                if spinner_active:
                    update_spinner()
                # Check if worker thread is still alive, if not, something went wrong
                if not worker_thread.is_alive():
                     # It might have finished between the q.get and this check,
                     # or it might have died unexpectedly.
                     # Check queue again briefly in case 'DONE' or 'ERROR' just arrived.
                     try:
                         item_type, data = q.get_nowait()
                         if item_type == 'DONE': break
                         if item_type == 'ERROR':
                             final_status_success = False
                             stop_spinner(success=False)
                             if is_reasoning_deepseek or in_think_block_tag:
                                 if not hide_reasoning: sys.stdout.write(ANSI_RESET)
                             sys.stdout.flush()
                             sys.stderr.write(f"\n{data}\n")
                             return None
                         # If it was a CHUNK, process it (shouldn't happen often here)
                         # ... (add chunk processing logic if needed, though unlikely)

                     except queue.Empty:
                         # Worker died without sending DONE or ERROR
                         if final_status_success: # Avoid double message if error already handled
                             final_status_success = False
                             stop_spinner(success=False)
                             sys.stderr.write("\nWorker thread finished unexpectedly.\n")
                         return None # Indicate error
                     break # Exit loop if DONE/ERROR was found after thread died check


        # --- Final Cleanup ---
        # Stop spinner if it was still active at the end (e.g., hidden reasoning ended stream)
        if spinner_active:
            stop_spinner(success=final_status_success)

        # Reset color if non-hidden reasoning was interrupted or ended stream
        if is_reasoning_deepseek or in_think_block_tag:
             if not hide_reasoning:
                 sys.stdout.write(ANSI_RESET)
                 sys.stdout.flush()

        processed_response = "".join(processed_response_chunks)
        return processed_response # Return only the processed final 'content' for history

    except KeyboardInterrupt:
        final_status_success = False
        stop_spinner(success=False) # Stop spinner with cross mark on interrupt
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET) # Reset color if needed
        sys.stdout.flush()
        sys.stderr.write("\nResponse generation interrupted by user.\n")
        # Note: Worker thread might still be running, but daemon=True helps cleanup
        return None
    except Exception as e:
        final_status_success = False
        stop_spinner(success=False)
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        sys.stderr.write(f"\nAn unexpected error occurred in stream_response: {str(e)}\n")
        return None
    finally:
        # Ensure spinner is definitely stopped if an error occurred before final cleanup
        if spinner_active:
             # Clear the line properly before potentially exiting
             sys.stdout.write("\r" + " " * 20 + "\r")
             sys.stdout.flush()
        # Wait briefly for the worker thread to potentially finish cleanup, though it's a daemon
        # worker_thread.join(timeout=0.5) # Optional: wait briefly


# --- setup_tty_stdin and restore_stdin ---
# ... (保持不变) ...
def setup_tty_stdin() -> Tuple[int, Optional[str]]:
    """Reads from pipe if available, then switches stdin to /dev/tty."""
    original_stdin_fd = -1
    pipe_input = ""
    try:
        original_stdin_fd = os.dup(0)
        if not sys.stdin.isatty():
            pipe_input = sys.stdin.read()
            sys.stdin.close()
            try:
                # Try opening /dev/tty for interactive input
                new_stdin_fd = os.open('/dev/tty', os.O_RDONLY)
                os.dup2(new_stdin_fd, 0) # Replace file descriptor 0 (stdin)
                os.close(new_stdin_fd) # Close the extra descriptor
                # Re-open sys.stdin using the new file descriptor 0
                sys.stdin = open(0, 'r', closefd=False)
            except OSError as e:
                sys.stderr.write(f"Warning: Could not switch to interactive TTY input: {e}\n")
                sys.stderr.write("Interactive mode might not work as expected.\n")
                # Attempt to restore original stdin if TTY switch failed
                try:
                    os.dup2(original_stdin_fd, 0)
                    sys.stdin = open(0, 'r', closefd=False)
                except OSError:
                     sys.stderr.write("Could not restore original stdin.\n")
                     sys.exit("Fatal: Cannot continue without standard input.")

        # Try enabling readline for better interactive input experience
        try:
            import readline
            # You might want specific bindings, e.g., history search
            readline.parse_and_bind('tab: complete')
            # Consider loading/saving history:
            # history_file = os.path.join(os.path.expanduser("~"), ".ag_history")
            # try:
            #     readline.read_history_file(history_file)
            # except FileNotFoundError:
            #     pass
            # import atexit
            # atexit.register(readline.write_history_file, history_file)
        except ImportError:
            pass # Readline not available on all systems (e.g., standard Windows cmd)

    except Exception as e:
        sys.stderr.write(f"Error setting up TTY/stdin: {e}\n")
        # Ensure the original fd is closed if we saved it but failed later
        if original_stdin_fd != -1:
            try:
                os.close(original_stdin_fd)
            except OSError:
                pass # Ignore errors closing if it's already closed or invalid

    return original_stdin_fd, pipe_input


def restore_stdin(original_stdin_fd: int):
    """Closes the saved original standard input file descriptor."""
    if original_stdin_fd != -1:
        try:
            # It's generally better practice to restore the original descriptor
            # if possible, rather than just closing the saved one.
            # However, simply closing the saved descriptor prevents resource leaks.
            # If dup2 was used successfully to switch to /dev/tty,
            # restoring the original might interfere if the original pipe is closed.
            # Just closing the saved descriptor is safer in this complex setup.
            os.close(original_stdin_fd)
        except OSError as e:
            # Ignore errors like "bad file descriptor" if it was already closed
            pass
        except Exception as e:
            # Log unexpected errors during cleanup
            sys.stderr.write(f"Unexpected error closing saved stdin descriptor: {e}\n")


# --- Modified main Function ---

def main():
    """Main function to parse arguments and run the interactive loop."""
    parser = argparse.ArgumentParser(
        description='AG - Ask GPT from CLI. Interact with OpenAI-compatible APIs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument('--api-url', help='API endpoint URL', default=DEFAULT_API_URL)
    parser.add_argument('--model', help='Model name to use', default=DEFAULT_MODEL)
    # <-- Added model-type argument -->
    parser.add_argument('--model-type', help='Type of model API behavior', choices=['openai', 'gemini'], default=DEFAULT_MODEL_TYPE)
    parser.add_argument('--api-key', help='API key (or set OPENAI_API_KEY env var)', default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens for response', default=DEFAULT_MAX_TOKENS)
    # Use None as default sentinel for penalties, check before adding to API call
    parser.add_argument('--frequency-penalty', type=float, help='Frequency penalty (e.g., 0.0 to 2.0). Use -1.0 to disable.', default=DEFAULT_FREQUENCY_PENALTY)
    parser.add_argument('--presence-penalty', type=float, help='Presence penalty (e.g., 0.0 to 2.0). Use -1.0 to disable.', default=DEFAULT_PRESENCE_PENALTY)
    parser.add_argument('-g', '--grep-enabled', action='store_true', help='Act as an AI-powered grep filter for piped input')
    parser.add_argument('-r', '--hide-reasoning', action='store_true', help='Hide reasoning content (<think> or model-specific), show spinner instead. Automatically enabled for --model-type gemini.')
    parser.add_argument('raw_text', nargs='?', help='Direct question (optional). If provided, runs once and exits.')

    args = parser.parse_args()

    # <-- Automatically enable hide_reasoning for gemini -->
    if args.model_type == 'gemini':
        args.hide_reasoning = True
        # Optional: Inform the user if they didn't explicitly set -r
        # if not any(arg in sys.argv for arg in ['-r', '--hide-reasoning']):
        #    sys.stderr.write("Info: --hide-reasoning automatically enabled for --model-type gemini.\n")


    # Handle penalty disabling
    freq_penalty = args.frequency_penalty if args.frequency_penalty != -1.0 else None
    pres_penalty = args.presence_penalty if args.presence_penalty != -1.0 else None


    try:
        client = OpenAI(
            base_url=args.api_url,
            api_key=args.api_key
        )
    except Exception as e:
        sys.exit(f"Error initializing OpenAI client: {e}")

    messages: List[Dict[str, str]] = []
    original_stdin_fd, pipe_input = setup_tty_stdin() # Handle pipe and switch to TTY

    grep_prompt_template = """
I want you to act strictly as a Linux `grep` command filter.
Input will consist of text, potentially multi-line output from another command.
The pattern can be in natural languages that you need to understand it.
Your task is to filter this input based on a user-provided pattern.
***You MUST ONLY return the lines from the input that match the pattern.***
Do NOT include any explanations, introductions, summaries, or conversational text.
Do NOT add line numbers unless they were present in the original input lines.
If no lines match, return absolutely ***nothing*** (empty output).

Pattern: "{pattern}"

Input Text:
{input_text}

Matching Lines Only:"""

    try:
        # --- Single Shot Mode ---
        if args.raw_text:
            user_content = args.raw_text
            if pipe_input:
                # Prepend pipe input for context, then the user's raw text query
                user_content = f"{pipe_input.strip()}\n\nUser Query: {user_content}"

            if args.grep_enabled:
                if not pipe_input:
                    sys.exit("Error: --grep-enabled requires piped input.")
                # Format the prompt specifically for grep mode
                full_prompt = grep_prompt_template.format(pattern=args.raw_text, input_text=pipe_input)
                # Grep mode usually doesn't need conversation history
                messages = [{"role": "user", "content": full_prompt}]
            else:
                processed_content = process_input_directives(user_content)
                messages.append({"role": "user", "content": processed_content})

            # <-- Pass model_type and hide_reasoning to stream_response -->
            response_content = stream_response(
                client, args.model, args.model_type, messages, args.max_tokens,
                freq_penalty, pres_penalty, args.hide_reasoning
            )
            # No need to append assistant response if exiting
            # Ensure newline *after* response, including spinner's potential newline
            # sys.stdout.write("\n") # Spinner stop adds a newline now
            restore_stdin(original_stdin_fd) # Restore stdin before exiting
            sys.exit(0) # Exit after single shot

        # --- Interactive Mode ---
        sys.stderr.write("Entering interactive mode. Use Ctrl+D to submit, Ctrl+C to exit.\n")
        initial_pipe_input = pipe_input # Store initial pipe input for first message

        while True:
            # Use stderr for prompt to avoid mixing with stdout response/spinner
            sys.stderr.write("\n💥 Ask (Ctrl+D Submit, !cmd, @file, @url):\n")
            lines = []
            prompt_prefix = "> "
            try:
                while True:
                    # Read line by line using input(), prompted to stderr
                    line = input(prompt_prefix)
                    lines.append(line)
                    # prompt_prefix = "  " # Indent subsequent lines for multiline input clarity
            except EOFError: # Ctrl+D detected
                pass

            user_input = '\n'.join(lines).strip()

            # Prepend initial pipe input if this is the first interaction
            if initial_pipe_input:
                user_input = f"{initial_pipe_input.strip()}\n\nUser Query: {user_input}"
                initial_pipe_input = "" # Clear it after first use

            if not user_input:
                continue # Skip if only whitespace or empty input after stripping

            # --- Handle Commands ---
            command_executed = False
            temp_lines = []
            for line in user_input.splitlines():
                stripped_line = line.strip()
                if stripped_line.startswith('!'):
                    command_to_execute = stripped_line[1:].strip()
                    if command_to_execute.lower() == 'clear':
                        messages.clear()
                        # os.system('clear' if os.name == 'posix' else 'cls') # Alternative clear screen
                        sys.stdout.write(ANSI_CLEAR_SCREEN) # Use ANSI clear screen
                        sys.stdout.flush()
                        sys.stderr.write("Conversation history cleared.\n") # Use stderr for meta messages
                        command_executed = True
                        break # Don't process further lines if 'clear' is issued
                    elif command_to_execute:
                        sys.stderr.write(f"\nExecuting: `{command_to_execute}`\n---\n") # Command echo to stderr
                        output_from_command = execute_command(command_to_execute)
                        sys.stdout.write(output_from_command) # Command output to stdout
                        sys.stdout.write("---\n")
                        sys.stdout.flush()
                        command_executed = True
                        # Decide if command output should be part of the *next* prompt
                        # For now, just execute and don't send to LLM or process further lines
                        break
                    else:
                         # Line starts with '!' but no command follows, treat as normal text
                         temp_lines.append(line)
                else:
                    temp_lines.append(line)

            if command_executed:
                 continue # Go to next loop iteration if a command was run

            # Reconstruct user_input if only some lines were commands
            user_input = '\n'.join(temp_lines).strip()
            if not user_input: # Check again if processing commands left input empty
                continue

            # --- Process and Send to LLM ---
            processed_input = process_input_directives(user_input)
            messages.append({"role": "user", "content": processed_input})

            sys.stdout.write("\n💡:\n") # Use stdout for the AI response prefix
            sys.stdout.flush()
            # <-- Pass model_type and hide_reasoning to stream_response -->
            response_content = stream_response(
                client, args.model, args.model_type, messages, args.max_tokens,
                freq_penalty, pres_penalty, args.hide_reasoning
            )

            if response_content is not None:
                messages.append({"role": "assistant", "content": response_content})
            # Ensure a newline separation before the next prompt, spinner stop handles its own newline
            # sys.stdout.write("\n") # Removed as spinner stop adds newline

    except KeyboardInterrupt:
        sys.stderr.write("\nExiting...\n")
    finally:
        # --- Cleanup ---
        restore_stdin(original_stdin_fd)

if __name__ == '__main__':
    main()

