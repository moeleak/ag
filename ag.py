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

# --- Modified stream_response Function ---

def stream_response(client: OpenAI, model: str, model_type: str, messages: List[Dict[str, str]],
                    max_tokens: int, frequency_penalty: Optional[float],
                    presence_penalty: Optional[float], hide_reasoning: bool) -> Optional[str]:
    """Streams response, optionally hiding reasoning with a spinner, supports Gemini-style delay."""

    # --- 新增语言检测函数 ---
    def detect_language(text: str) -> str:
        """简单检测输入文本的主要语言"""                                                                                                                                        # 检查中文字符
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'
    LANGUAGE_PROMPTS = {
        'zh': "请用中文回答，且禁止加入拼音。",
        'en': "Please respond in English."
    }

    full_response_chunks = []
    processed_response_chunks = []
    buffer = ''

    if model_type == 'gemini' and messages:
        last_user_message = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), '')
        current_language = detect_language(last_user_message)

        if len(messages) == 1:
            language_prompt = LANGUAGE_PROMPTS.get(current_language, 'en')
            messages[0]['content'] = f"{language_prompt}\n\n{messages[0]['content']}"

    full_response_chunks = []
    processed_response_chunks = [] # Stores content intended for history (no reasoning/think tags)
    buffer = '' # Buffer for standard <think> tag processing

    is_reasoning_deepseek = False
    in_think_block_tag = False
    spinner_active = False
    # spinner_chars = itertools.cycle(['|', '/', '-', '\\'])
    # spinner_chars = itertools.cycle(['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'])
    # spinner_chars = itertools.cycle(['░', '▒', '▓', '▒'])
    # spinner_chars = itertools.cycle([' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█', '▉', '▊', '▋', '▌', '▍', '▎', '▏'])
    spinner_chars = itertools.cycle(['🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛'])
    #spinner_chars = itertools.cycle(['🩷','❤️','🧡','💛','💚','🩵','💙','💜','🖤','🩶','🤍','🤎'])
    #spinner_chars = itertools.cycle(['✊🏻','✊🏼','✊🏽','✊🏾','✊🏿'])
    #spinner_chars = itertools.cycle(['💭','💬','💡','🧠'])
    #spinner_chars = itertools.cycle(['😋','😛','😝','😜','🤪'])
    first_content_received_gemini = False # Specifically for Gemini initial content detection

    # --- NEW STATE FLAG ---
    strip_leading_newline_next_write = False

    REASONING_START_MARKER = "--- Reasoning Content ---"
    REASONING_END_MARKER = "--- End Reasoning ---"

    def start_spinner():
        nonlocal spinner_active
        if not spinner_active:
            sys.stderr.write("Thinking... ")
            spinner_active = True
            sys.stderr.flush()

    def update_spinner():
        if spinner_active:
            spinner_char = next(spinner_chars)
            sys.stderr.write(f"\rThinking... {spinner_char}")
            sys.stderr.flush()

    def stop_spinner(success=True):
        nonlocal spinner_active, strip_leading_newline_next_write
        if spinner_active:
            symbol = "✅" if success else "❌"
            sys.stderr.write("\r" + " " * 20 + "\r")
            sys.stderr.write(f"Thinking... {symbol}\n")
            spinner_active = False
            sys.stderr.flush()
            # --- SET FLAG HERE ---
            # Set the flag whenever the spinner stops due to incoming content or success
            if success:
                 strip_leading_newline_next_write = True

    q = queue.Queue()
    worker_thread = threading.Thread(
        target=_api_worker,
        args=(q, client, model, messages, max_tokens, frequency_penalty, presence_penalty),
        daemon=True
    )

    worker_thread.start()

    final_status_success = True
    try:
        if hide_reasoning:
            start_spinner()

        while True:
            try:
                item_type, data = q.get(timeout=0.1)

                if item_type == 'CHUNK':
                    chunk = data
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    if delta is None: continue

                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    content = delta.content

                    # --- Stop Gemini spinner ---
                    if model_type == 'gemini' and content and not first_content_received_gemini and spinner_active:
                         stop_spinner() # This will set strip_leading_newline_next_write = True
                         first_content_received_gemini = True

                    # --- Handle DeepSeek Reasoning ---
                    if reasoning_content:
                        if hide_reasoning:
                            if not spinner_active: start_spinner()
                            update_spinner()
                        else:
                            if spinner_active: stop_spinner() # Stop if showing reasoning
                            if not is_reasoning_deepseek:
                                is_reasoning_deepseek = True
                                sys.stdout.write(REASONING_START_MARKER + "\n" + ANSI_GREEN)
                                sys.stdout.flush()
                            sys.stdout.write(reasoning_content)
                            sys.stdout.flush()
                        continue # Skip normal content processing

                    # --- Handle Standard Content ---
                    if content:
                        # End DeepSeek reasoning display if it was active
                        if is_reasoning_deepseek:
                            is_reasoning_deepseek = False
                            if not hide_reasoning:
                                sys.stdout.write(ANSI_RESET + REASONING_END_MARKER)
                                sys.stdout.flush()

                        # Stop spinner if it's still active when regular content arrives
                        # (This handles cases like DeepSeek without reasoning_content but with hidden <think>,
                        # or just general delay before first content chunk when hide_reasoning is on)
                        # Exclude Gemini because its spinner stop is handled above.
                        if model_type != 'gemini' and spinner_active:
                             stop_spinner() # This will set strip_leading_newline_next_write = True

                        full_response_chunks.append(content) # Add raw content chunk to full history
                        buffer += content

                        # Process buffer for <think> tags and prepare output
                        processed_chunk_for_history = '' # Content excluding <think> tags for history list
                        temp_buffer = buffer
                        buffer = ''
                        output_buffer = '' # Content to be printed to stdout this iteration

                        while temp_buffer:
                            if not in_think_block_tag:
                                start_pos = temp_buffer.find('<think>')
                                if start_pos == -1: # No <think> tag start
                                    chunk_to_process = temp_buffer
                                    temp_buffer = ''

                                    # Stop spinner if needed (e.g., content arrived while spinner was on)
                                    # This might be redundant if stopped above, but safe.
                                    if spinner_active:
                                        stop_spinner() # Sets the flag

                                    output_buffer += chunk_to_process
                                    processed_chunk_for_history += chunk_to_process

                                else: # Found <think> tag start
                                    before_think = temp_buffer[:start_pos]
                                    processed_chunk_for_history += before_think # Add to history

                                    # Stop spinner if needed before processing pre-think content
                                    if spinner_active and before_think:
                                        stop_spinner() # Sets the flag

                                    output_buffer += before_think # Add pre-think content to output

                                    # Handle the <think> tag itself
                                    in_think_block_tag = True
                                    temp_buffer = temp_buffer[start_pos + len('<think>'):]
                                    if hide_reasoning:
                                        if not spinner_active: start_spinner() # Start spinner for hidden think block
                                        update_spinner()
                                    else:
                                        output_buffer += ANSI_GREEN # Start green for visible think block
                            else: # Inside a <think> block
                                end_pos = temp_buffer.find('</think>')
                                if end_pos == -1: # Still inside <think> block
                                    if hide_reasoning:
                                        update_spinner()
                                        # Consume buffer without adding to output/history
                                        temp_buffer = ''
                                    else:
                                        output_buffer += temp_buffer # Print visible thinking content
                                        temp_buffer = ''
                                else: # Found </think> tag end
                                    think_content = temp_buffer[:end_pos]
                                    in_think_block_tag = False
                                    temp_buffer = temp_buffer[end_pos + len('</think>'):]

                                    if hide_reasoning:
                                        update_spinner()
                                        # Content inside hidden <think> is not added to output/history
                                    else:
                                        output_buffer += think_content # Print visible thinking content
                                        output_buffer += ANSI_RESET # End green

                        # --- Apply LSTRIP before writing to stdout ---
                        if strip_leading_newline_next_write and output_buffer:
                            output_buffer = output_buffer.lstrip('\n')
                            # Only strip the *very first* time after spinner stops
                            if output_buffer: # Don't reset if stripping made it empty
                                strip_leading_newline_next_write = False

                        # --- Write to stdout and store processed chunk ---
                        if output_buffer:
                            sys.stdout.write(output_buffer)
                            sys.stdout.flush()
                        if processed_chunk_for_history:
                             processed_response_chunks.append(processed_chunk_for_history)


                elif item_type == 'DONE':
                    break # Worker finished successfully

                elif item_type == 'ERROR':
                    final_status_success = False
                    stop_spinner(success=False) # Stop spinner with failure
                    if is_reasoning_deepseek or in_think_block_tag:
                        if not hide_reasoning: sys.stdout.write(ANSI_RESET)
                    sys.stdout.flush()
                    sys.stderr.write(f"\n{data}\n")
                    return None # Indicate error

            except queue.Empty:
                if spinner_active:
                    update_spinner()
                if not worker_thread.is_alive():
                     # Check queue one last time in case DONE/ERROR arrived just now
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
                     except queue.Empty:
                         if final_status_success: # Avoid double message
                             final_status_success = False
                             stop_spinner(success=False)
                             sys.stderr.write("\nWorker thread finished unexpectedly.\n")
                         return None
                     break # Exit outer loop if DONE/ERROR found

        # --- Final Cleanup ---
        if spinner_active: # If loop finished while spinner was still active (e.g., hidden <think> at end)
            stop_spinner(success=final_status_success)

        if is_reasoning_deepseek or in_think_block_tag: # Reset color if needed
             if not hide_reasoning:
                 sys.stdout.write(ANSI_RESET)
                 sys.stdout.flush()

        # Join the chunks meant for history
        processed_response = "".join(processed_response_chunks)
        # We keep the lstrip here for the *returned value* to ensure history is clean,
        # even if the printed output was handled differently.
        return processed_response.lstrip('\n')

    except KeyboardInterrupt:
        final_status_success = False
        stop_spinner(success=False)
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        sys.stderr.write("\nResponse generation interrupted by user.\n")
        return None
    except Exception as e:
        final_status_success = False
        stop_spinner(success=False)
        if is_reasoning_deepseek or in_think_block_tag:
            if not hide_reasoning: sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        sys.stderr.write(f"\nAn unexpected error occurred in stream_response: {str(e)}\n")
        traceback.print_exc(file=sys.stderr) # Print full traceback
        return None
    finally:
        # Ensure cursor is visible and line is clear if spinner was ever active
        if 'spinner_active' in locals() and spinner_active:
             sys.stderr.write("\r" + " " * 20 + "\r")
             sys.stderr.flush()




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
            history_file = os.path.join(os.path.expanduser("~"), ".ag_history")
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass
            import atexit
            atexit.register(readline.write_history_file, history_file)
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

