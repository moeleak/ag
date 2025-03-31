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
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple


# --- Constants ---
DEFAULT_API_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "deepseek-reasoner"
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

# --- Helper Functions ---

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
        # Use list format for command if not using shell=True for better security,
        # but shell=True is convenient for user input like `ls -l | grep py`
        # Keep shell=True here given the context, but be aware of risks.
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




ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'

def stream_response(client: OpenAI, model: str, messages: List[Dict[str, str]],
                    max_tokens: int, frequency_penalty: Optional[float],
                    presence_penalty: Optional[float], hide_reasoning: bool) -> Optional[str]:
    """Streams response, optionally hiding reasoning with a spinner."""

    full_response_chunks = []
    processed_response_chunks = []
    buffer = '' # Buffer for standard <think> tag processing

    is_reasoning_deepseek = False   # Track if we are actively processing DeepSeek reasoning_content
    in_think_block_tag = False      # Track if we are inside a <think> tag block
    spinner_active = False          # Track if the spinner animation is currently shown
    spinner_chars = itertools.cycle(['|', '/', '-', '\\'])

    # --- Markers for non-hidden reasoning ---
    REASONING_START_MARKER = "--- Reasoning Content ---"
    REASONING_END_MARKER = "\n--- End Reasoning ---"

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
            sys.stdout.write(f"\rThinking... {symbol}")
            spinner_active = False
            sys.stdout.flush()

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
            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, 'reasoning_content', None)
            content = delta.content

            # --- Handle DeepSeek Reasoning ---
            if reasoning_content:
                if hide_reasoning:
                    start_spinner()
                    update_spinner()
                else:
                    if not is_reasoning_deepseek:
                        is_reasoning_deepseek = True
                        sys.stdout.write(REASONING_START_MARKER + "\n")
                        sys.stdout.write(ANSI_GREEN)
                        sys.stdout.flush()
                    sys.stdout.write(reasoning_content)
                    sys.stdout.flush()
                continue
            # --- Handle Standard Content (including <think> tags) ---
            if content:
                # If DeepSeek reasoning was just happening, end it
                if is_reasoning_deepseek:
                    is_reasoning_deepseek = False
                    if not hide_reasoning: # Only print markers if not hidden
                        sys.stdout.write(ANSI_RESET)
                        sys.stdout.write(REASONING_END_MARKER + "\n")
                        sys.stdout.flush()
                    # If hiding, the spinner will be stopped below when content starts printing

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
                            # If spinner was active, stop it before printing normal content
                            if spinner_active:
                                stop_spinner()
                            output_buffer += temp_buffer
                            processed_chunk_for_buffer += temp_buffer
                            temp_buffer = ''
                        else: # Found <think> tag start
                            before_think = temp_buffer[:start_pos]
                            # If spinner was active, stop it before printing normal content
                            if spinner_active and before_think:
                                stop_spinner()
                            output_buffer += before_think
                            processed_chunk_for_buffer += before_think

                            in_think_block_tag = True
                            temp_buffer = temp_buffer[start_pos + len('<think>'):]

                            if hide_reasoning:
                                start_spinner()
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

        # --- Final Cleanup ---
        # Stop spinner if it was still active at the end of the stream
        stop_spinner()

        # Reset color if non-hidden reasoning was interrupted or ended stream
        if is_reasoning_deepseek or in_think_block_tag:
             if not hide_reasoning:
                 sys.stdout.write(ANSI_RESET)
                 sys.stdout.flush()

        processed_response = "".join(processed_response_chunks)
        return processed_response # Return only the processed final 'content' for history

    except KeyboardInterrupt:
        stop_spinner(success=False) # Stop spinner with cross mark on interrupt
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET) # Reset color if needed
        sys.stdout.flush()
        sys.stderr.write("\nResponse generation interrupted by user.\n")
        return None
    except APIError as e:
        stop_spinner(success=False)
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        sys.stderr.write(f"\nAPI Error: {str(e)}\n")
        return None
    except Exception as e:
        stop_spinner(success=False)
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        sys.stderr.write(f"\nAn unexpected error occurred during streaming: {str(e)}\n")
        return None
    finally:
        # Ensure spinner is definitely stopped if an error occurred before final cleanup
        if spinner_active:
             sys.stdout.write("\r" + " " * 20 + "\r") # Clear spinner line on unexpected exit
             sys.stdout.flush()


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
                new_stdin_fd = os.open('/dev/tty', os.O_RDONLY)
                os.dup2(new_stdin_fd, 0)
                os.close(new_stdin_fd)
                sys.stdin = open(0, 'r', closefd=False)
            except OSError as e:
                sys.stderr.write(f"Warning: Could not switch to interactive TTY input: {e}\n")
                sys.stderr.write("Interactive mode might not work as expected.\n")
                try:
                    os.dup2(original_stdin_fd, 0)
                    sys.stdin = open(0, 'r', closefd=False)
                except OSError:
                     sys.stderr.write("Could not restore original stdin.\n")
                     sys.exit("Fatal: Cannot continue without standard input.")

        try:
            import readline
            readline.parse_and_bind('tab: complete')
        except ImportError:
            pass

    except Exception as e:
        sys.stderr.write(f"Error setting up TTY/stdin: {e}\n")
        if original_stdin_fd != -1:
            try:
                os.close(original_stdin_fd)
            except OSError:
                pass

    return original_stdin_fd, pipe_input


def restore_stdin(original_stdin_fd: int):
    """Closes the saved original standard input file descriptor."""
    if original_stdin_fd != -1:
        try:
            os.close(original_stdin_fd)
        except OSError as e:
            pass
        except Exception as e:
            sys.stderr.write(f"Unexpected error closing saved stdin descriptor: {e}\n")



def main():
    """Main function to parse arguments and run the interactive loop."""
    parser = argparse.ArgumentParser(
        description='AG - Ask GPT from CLI. Interact with OpenAI-compatible APIs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument('--api-url', help='API endpoint URL', default=DEFAULT_API_URL)
    parser.add_argument('--model', help='Model name to use', default=DEFAULT_MODEL)
    parser.add_argument('--api-key', help='API key (or set OPENAI_API_KEY env var)', default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens for response', default=DEFAULT_MAX_TOKENS)
    # Use None as default sentinel for penalties, check before adding to API call
    parser.add_argument('--frequency-penalty', type=float, help='Frequency penalty (e.g., 0.0 to 2.0). Use -1.0 to disable.', default=DEFAULT_FREQUENCY_PENALTY)
    parser.add_argument('--presence-penalty', type=float, help='Presence penalty (e.g., 0.0 to 2.0). Use -1.0 to disable.', default=DEFAULT_PRESENCE_PENALTY)
    parser.add_argument('-g', '--grep-enabled', action='store_true', help='Act as an AI-powered grep filter for piped input')
    parser.add_argument('-r', '--hide-reasoning', action='store_true', help='Hide reasoning content, show spinner and checkmark instead')
    parser.add_argument('raw_text', nargs='?', help='Direct question (optional). If provided, runs once and exits.')

    args = parser.parse_args()

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
                user_content = f"{pipe_input.strip()}\n{user_content}" # Combine pipe and arg

            if args.grep_enabled:
                if not pipe_input:
                    sys.exit("Error: --grep-enabled requires piped input.")
                # Format the prompt specifically for grep mode
                full_prompt = grep_prompt_template.format(pattern=args.raw_text, input_text=pipe_input)
                # Grep mode usually doesn't need conversation history
                messages = [{"role": "user", "content": full_prompt}]
                # Optionally clear messages if you want grep to be stateless even if run after interactive mode
                # messages.clear()
                # messages.append({"role": "user", "content": full_prompt})
            else:
                processed_content = process_input_directives(user_content)
                messages.append({"role": "user", "content": processed_content})

            response_content = stream_response(client, args.model, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            # No need to append assistant response if exiting
            sys.stdout.write("\n") # Ensure newline after response
            restore_stdin(original_stdin_fd) # Restore stdin before exiting
            sys.exit(0) # Exit after single shot

        # --- Interactive Mode ---
        sys.stderr.write("Entering interactive mode. Use Ctrl+D to submit, Ctrl+C to exit.\n")
        initial_pipe_input = pipe_input # Store initial pipe input for first message

        while True:
            sys.stderr.write("\n💥 Ask (Ctrl+D Submit, !cmd, @file, @url):\n")
            lines = []
            try:
                while True:
                    line = input("> ") # Use stderr for prompt if stdout is for response
                    lines.append(line)
            except EOFError: # Ctrl+D detected
                pass

            user_input = '\n'.join(lines).strip()

            if initial_pipe_input:
                user_input = f"{initial_pipe_input.strip()}\n{user_input}"
                initial_pipe_input = ""
            if not user_input:
                continue

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
                        sys.stdout.write("\nConversation history cleared.\n")
                        command_executed = True
                    elif command_to_execute:
                        sys.stdout.write(f"\nExecuting: `{command_to_execute}`\n---\n")
                        output_from_command = execute_command(command_to_execute)
                        sys.stdout.write(output_from_command)
                        sys.stdout.write("---\n")
                        command_executed = True # Skip
                    else:
                         temp_lines.append(line)
                else:
                    temp_lines.append(line)

            if command_executed:
                 continue

            user_input = '\n'.join(temp_lines)
            if not user_input.strip():
                continue

            # --- Process and Send to LLM ---
            processed_input = process_input_directives(user_input)
            messages.append({"role": "user", "content": processed_input})

            sys.stdout.write("\n💡:\n")
            response_content = stream_response(client, args.model, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)

            if response_content is not None:
                messages.append({"role": "assistant", "content": response_content})
            sys.stdout.write("\n")

    except KeyboardInterrupt:
        sys.stderr.write("\nExiting...\n")
    finally:
        # --- Cleanup ---
        restore_stdin(original_stdin_fd)

if __name__ == '__main__':
    main()

