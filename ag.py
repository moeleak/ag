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
import threading
import queue
import traceback
import wcwidth
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any

try:
    import readline
except ImportError:
    pass 

# 新增: 条件导入 termios
try:
    import termios
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False

# --- Constants ---
DEFAULT_API_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_MODEL_TYPE = "openai"
DEFAULT_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_FREQUENCY_PENALTY = 0.5
DEFAULT_PRESENCE_PENALTY = 0.5
USER_AGENT = "Mozilla/5.0 (Windows NT 1.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'
ANSI_SAVE_CURSOR = '\033[s'
ANSI_RESTORE_CURSOR = '\033[u'
ANSI_CLEAR_SCREEN = '\033[2J\033[H'

# --- Helper Functions (fetch_webpage, execute_command, process_input_directives) ---
def fetch_webpage(url: str) -> Optional[str]:
    """Fetches and extracts text content from a URL."""
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()
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
                                text=True)
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
                filename = match.group(1).strip()
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        processed_lines.append(f"--- Content of {filename} ---\n{f.read()}\n--- End of {filename} ---")
                except FileNotFoundError:
                    sys.stderr.write(f"Error: File not found: {filename}\n")
                except IOError as e:
                    sys.stderr.write(f"Error reading {filename}: {str(e)}\n")
                except Exception as e:
                     sys.stderr.write(f"Unexpected error processing file {filename}: {str(e)}\n")
            else:
                 processed_lines.append(line)
        elif stripped_line.startswith('@url'):
            match = re.match(r'@url\(([^)]+)\)', stripped_line)
            if match:
                url = match.group(1).strip()
                webpage_content = fetch_webpage(url)
                if webpage_content:
                    processed_lines.append(f"--- Content from {url} ---\n{webpage_content}\n--- End of {url} ---")
            else:
                 processed_lines.append(line)
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
            q.put(('CHUNK', chunk))
        q.put(('DONE', None))
    except APIError as e:
        q.put(('ERROR', f"API Error: {str(e)}"))
    except Exception as e:
        q.put(('ERROR', f"An unexpected error occurred during streaming: {str(e)}"))

# --- stream_response Function ---
def stream_response(client: OpenAI, model: str, model_type: str, messages: List[Dict[str, str]],
                    max_tokens: int, frequency_penalty: Optional[float],
                    presence_penalty: Optional[float], hide_reasoning: bool) -> Optional[str]:
    def detect_language(text: str) -> str:
        if re.search(r'[\u4e00-\u9fff]', text): return 'zh'
        return 'en'
    LANGUAGE_PROMPTS = {
        'zh': "请用中文回答，且禁止加入拼音。",
        'en': "Please respond in English."
    }
    if model_type == 'gemini' and messages:
        last_user_message = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), '')
        current_language = detect_language(last_user_message)
        if len(messages) == 1:
            language_prompt = LANGUAGE_PROMPTS.get(current_language, LANGUAGE_PROMPTS['en'])
            messages[0]['content'] = f"{language_prompt}\n\n{messages[0]['content']}"

    full_response_chunks, processed_response_chunks, buffer = [], [], ''
    is_reasoning_deepseek, in_think_block_tag, spinner_active = False, False, False
    spinner_chars = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])
    first_content_received_gemini, strip_leading_newline_next_write = False, False
    REASONING_START_MARKER, REASONING_END_MARKER = "--- Reasoning Content ---", "--- End Reasoning ---"
    thinking_indicator_char = "🤔"
    len_thinking_indicator_display_cells = wcwidth.wcwidth(thinking_indicator_char)
    if len_thinking_indicator_display_cells == -1: len_thinking_indicator_display_cells = 1
    thinking_indicator_on_screen = False

    def clear_thinking_indicator_if_on_screen():
        nonlocal thinking_indicator_on_screen
        if thinking_indicator_on_screen:
            sys.stdout.write(("\b" * len_thinking_indicator_display_cells) + \
                             (" " * len_thinking_indicator_display_cells) + \
                             ("\b" * len_thinking_indicator_display_cells))
            sys.stdout.flush()
            thinking_indicator_on_screen = False
    def write_thinking_chunk_with_indicator(text_chunk: str):
        nonlocal thinking_indicator_on_screen
        clear_thinking_indicator_if_on_screen()
        sys.stdout.write(text_chunk + thinking_indicator_char)
        sys.stdout.flush()
        thinking_indicator_on_screen = True
    def start_spinner():
        nonlocal spinner_active
        clear_thinking_indicator_if_on_screen()
        if not spinner_active:
            sys.stderr.write("Thinking... "); spinner_active = True; sys.stderr.flush()
    def update_spinner():
        if spinner_active:
            sys.stderr.write(f"\rThinking... {next(spinner_chars)}"); sys.stderr.flush()
    def stop_spinner(success=True):
        nonlocal spinner_active, strip_leading_newline_next_write
        if spinner_active:
            sys.stderr.write("\r" + " " * 20 + "\r" + f"Thinking... {'✅' if success else '❌'}\n")
            spinner_active = False; sys.stderr.flush()
            if success: strip_leading_newline_next_write = True
    
    q = queue.Queue()
    worker_thread = threading.Thread(target=_api_worker, args=(q, client, model, messages, max_tokens, frequency_penalty, presence_penalty), daemon=True)
    worker_thread.start()
    final_status_success = True
    try:
        if hide_reasoning: start_spinner()
        while True:
            try:
                item_type, data = q.get(timeout=0.1)
                if item_type == 'CHUNK':
                    chunk = data
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    if delta is None: continue
                    reasoning_content, content = getattr(delta, 'reasoning_content', None), delta.content
                    if model_type == 'gemini' and content and not first_content_received_gemini and spinner_active:
                        stop_spinner(); first_content_received_gemini = True
                    if reasoning_content:
                        if hide_reasoning:
                            if not spinner_active: start_spinner()
                            update_spinner()
                        else:
                            if spinner_active: stop_spinner()
                            if not is_reasoning_deepseek:
                                clear_thinking_indicator_if_on_screen(); is_reasoning_deepseek = True
                                sys.stdout.write(REASONING_START_MARKER + "\n" + ANSI_GREEN); sys.stdout.flush()
                                strip_leading_newline_next_write = False
                            write_thinking_chunk_with_indicator(reasoning_content)
                        continue
                    if content:
                        if is_reasoning_deepseek:
                            clear_thinking_indicator_if_on_screen(); is_reasoning_deepseek = False
                            if not hide_reasoning:
                                sys.stdout.write(ANSI_RESET + REASONING_END_MARKER + "\n"); sys.stdout.flush()
                                strip_leading_newline_next_write = True
                        if model_type != 'gemini' and spinner_active: stop_spinner()
                        full_response_chunks.append(content); buffer += content
                        processed_chunk_for_history_this_delta, current_processing_buffer, buffer = '', buffer, ''
                        output_to_print_for_normal_content, temp_idx = "", 0
                        while temp_idx < len(current_processing_buffer):
                            if not in_think_block_tag:
                                think_start_pos = current_processing_buffer.find('<think>', temp_idx)
                                if think_start_pos == -1:
                                    clear_thinking_indicator_if_on_screen()
                                    normal_chunk = current_processing_buffer[temp_idx:]
                                    output_to_print_for_normal_content += normal_chunk
                                    processed_chunk_for_history_this_delta += normal_chunk
                                    temp_idx = len(current_processing_buffer)
                                else:
                                    clear_thinking_indicator_if_on_screen()
                                    normal_chunk_before_think = current_processing_buffer[temp_idx:think_start_pos]
                                    output_to_print_for_normal_content += normal_chunk_before_think
                                    processed_chunk_for_history_this_delta += normal_chunk_before_think
                                    if output_to_print_for_normal_content:
                                        if strip_leading_newline_next_write:
                                            output_to_print_for_normal_content = output_to_print_for_normal_content.lstrip('\n')
                                            if output_to_print_for_normal_content: strip_leading_newline_next_write = False
                                        sys.stdout.write(output_to_print_for_normal_content); sys.stdout.flush()
                                        output_to_print_for_normal_content = ""
                                    temp_idx = think_start_pos + len('<think>'); in_think_block_tag = True
                                    if hide_reasoning:
                                        if not spinner_active: start_spinner()
                                        update_spinner()
                                    else:
                                        if spinner_active: stop_spinner()
                                        sys.stdout.write(ANSI_GREEN); sys.stdout.flush()
                            else: # Inside <think>
                                think_end_pos = current_processing_buffer.find('</think>', temp_idx)
                                if think_end_pos == -1:
                                    think_content_chunk = current_processing_buffer[temp_idx:]
                                    temp_idx = len(current_processing_buffer)
                                    if hide_reasoning: update_spinner()
                                    else: write_thinking_chunk_with_indicator(think_content_chunk)
                                else:
                                    think_content_chunk = current_processing_buffer[temp_idx:think_end_pos]
                                    temp_idx = think_end_pos + len('</think>'); in_think_block_tag = False
                                    if hide_reasoning: update_spinner()
                                    else:
                                        clear_thinking_indicator_if_on_screen()
                                        sys.stdout.write(think_content_chunk + ANSI_RESET); sys.stdout.flush()
                                        strip_leading_newline_next_write = True
                        if output_to_print_for_normal_content:
                            clear_thinking_indicator_if_on_screen()
                            if strip_leading_newline_next_write:
                                output_to_print_for_normal_content = output_to_print_for_normal_content.lstrip('\n')
                                if output_to_print_for_normal_content: strip_leading_newline_next_write = False
                            sys.stdout.write(output_to_print_for_normal_content); sys.stdout.flush()
                        if processed_chunk_for_history_this_delta:
                            processed_response_chunks.append(processed_chunk_for_history_this_delta)
                elif item_type == 'DONE':
                    clear_thinking_indicator_if_on_screen(); break
                elif item_type == 'ERROR':
                    final_status_success = False; clear_thinking_indicator_if_on_screen(); stop_spinner(success=False)
                    if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
                    sys.stdout.flush(); sys.stderr.write(f"\n{data}\n"); return None
            except queue.Empty:
                if spinner_active: update_spinner()
                if not worker_thread.is_alive():
                    clear_thinking_indicator_if_on_screen()
                    try:
                        item_type, data = q.get_nowait()
                        if item_type == 'DONE': break
                        if item_type == 'ERROR':
                            final_status_success = False; stop_spinner(success=False)
                            if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
                            sys.stdout.flush(); sys.stderr.write(f"\n{data}\n"); return None
                    except queue.Empty:
                        if final_status_success: # Only show error if we thought it was successful before
                            final_status_success = False; stop_spinner(success=False)
                            sys.stderr.write("\nWorker thread finished unexpectedly.\n")
                        return None # Error already handled or no error data from queue
                    break # Break from while True if DONE or ERROR handled from get_nowait
        clear_thinking_indicator_if_on_screen()
        if spinner_active: stop_spinner(success=final_status_success)
        if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning:
            sys.stdout.write(ANSI_RESET); sys.stdout.flush()
        processed_response = "".join(processed_response_chunks)
        return processed_response.lstrip('\n') if processed_response else ""
    except KeyboardInterrupt:
        final_status_success = False
        # Ensure ANSI state is reset immediately on interrupt for both stdout and stderr
        sys.stdout.write(ANSI_RESET)
        sys.stderr.write(ANSI_RESET) # In case spinner/colors were on stderr
        sys.stdout.flush()
        sys.stderr.flush()

        clear_thinking_indicator_if_on_screen()
        stop_spinner(success=False)
        
        # Reset reasoning block colors if active and not hidden
        if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning:
            sys.stdout.write(ANSI_RESET) # Ensures color reset on stdout
            sys.stdout.flush()
            sys.stderr.write('\n' + REASONING_END_MARKER + '\n') # Indicate end of reasoning on stderr

        sys.stderr.write(ANSI_GREEN + "^C Cancelled!\n" + ANSI_RESET) # User feedback
        sys.stderr.flush()

        # Try to flush stdin buffer if termios is available
        if TERMIOS_AVAILABLE:
            try:
                # Check stdin is a TTY and its file descriptor is valid before flushing
                if sys.stdin.isatty() and sys.stdin.fileno() >= 0:
                    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except Exception as e_flush:
                sys.stderr.write(f"Warn: Flushing stdin (TCIFLUSH) failed: {e_flush}\n")
                sys.stderr.flush()
        return None
    except Exception as e:
        final_status_success = False; clear_thinking_indicator_if_on_screen(); stop_spinner(success=False)
        if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
        sys.stdout.flush(); sys.stderr.write(f"\nAn unexpected error occurred in stream_response: {str(e)}\n")
        traceback.print_exc(file=sys.stderr); return None
    finally:
        clear_thinking_indicator_if_on_screen()
        if 'spinner_active' in locals() and spinner_active :
             sys.stderr.write("\r" + " " * (len("Thinking... ") + 5) + "\r"); sys.stderr.flush()
        if 'worker_thread' in locals() and worker_thread.is_alive(): worker_thread.join(timeout=1.0)

# --- Terminal Control Helper Functions (set_terminal_no_echoctl, restore_terminal_settings) ---
def set_terminal_no_echoctl(fd: int) -> Optional[List]:
    """Disables ECHOCTL terminal setting for the given file descriptor."""
    if not TERMIOS_AVAILABLE or not os.isatty(fd):
        return None
    try:
        old_settings = termios.tcgetattr(fd)
        new_settings = list(old_settings) # Create a mutable copy
        # lflag is at index 3. ECHOCTL is a bit in lflag.
        new_settings[3] &= ~termios.ECHOCTL 
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        return old_settings
    except termios.error as e:
        sys.stderr.write(f"Warn: Failed to set terminal no ECHOCTL: {e}\n")
        return None

def restore_terminal_settings(fd: int, old_settings: Optional[List]):
    """Restores terminal settings for the given file descriptor."""
    if not TERMIOS_AVAILABLE or not os.isatty(fd) or old_settings is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except termios.error as e:
        sys.stderr.write(f"Warn: Failed to restore terminal settings: {e}\n")

# --- setup_tty_stdin and restore_stdin ---
def setup_tty_stdin() -> Tuple[int, Optional[str]]:
    original_stdin_fd = -1
    pipe_input = ""
    readline_available = 'readline' in sys.modules
    try:
        original_stdin_fd = os.dup(0) # Save current stdin
        if not sys.stdin.isatty(): # If current stdin is not a TTY (e.g. piped input)
            pipe_input = sys.stdin.read() # Read all piped input
            sys.stdin.close() # Close the piped stdin
            try:
                # Try to open /dev/tty as the new stdin
                new_stdin_fd = os.open('/dev/tty', os.O_RDONLY)
                os.dup2(new_stdin_fd, 0) # Duplicate /dev/tty to fd 0 (stdin)
                os.close(new_stdin_fd) # Close original /dev/tty fd
                sys.stdin = open(0, 'r', closefd=False) # Re-open sys.stdin from fd 0
            except OSError as e:
                sys.stderr.write(f"Warning: Could not switch to TTY for interactive input: {e}\n")
                sys.stderr.write("Fallback to original (non-TTY) stdin. Interactive mode might be limited.\n")
                # Fallback: restore original stdin if TTY opening failed
                try:
                    os.dup2(original_stdin_fd, 0)
                    sys.stdin = open(0, 'r', closefd=False)
                except OSError as restore_e:
                     sys.stderr.write(f"Critical: Could not restore original stdin after TTY switch failure: {restore_e}\n")
        
        # Setup readline if stdin is now a TTY and readline is available
        if sys.stdin.isatty() and readline_available:
            try:
                readline.parse_and_bind('tab: complete')
                history_file = os.path.join(os.path.expanduser("~"), ".ag_history")
                try: readline.read_history_file(history_file)
                except FileNotFoundError: pass
                except Exception as hist_e: sys.stderr.write(f"Warn: Read history failed: {hist_e}\n")
                import atexit
                def save_history():
                    try: readline.write_history_file(history_file)
                    except Exception as save_hist_e: sys.stderr.write(f"Warn: Save history failed: {save_hist_e}\n")
                atexit.register(save_history)
            except Exception as readline_e: sys.stderr.write(f"Warn: Readline setup failed: {readline_e}\n")
        elif sys.stdin.isatty() and not readline_available:
            sys.stderr.write("Warn: readline unavailable. Editing features limited for Ctrl+C handling during input.\n")
            
    except Exception as e:
        sys.stderr.write(f"Error during TTY/stdin setup: {e}\n")
        # Ensure original_stdin_fd is closed if it was dup'd and not 0
        if original_stdin_fd != -1 and original_stdin_fd != 0:
            try: os.close(original_stdin_fd)
            except OSError: pass 
            original_stdin_fd = -1 # Mark as closed or invalid for restore_stdin
            
    return original_stdin_fd, pipe_input

def restore_stdin(original_stdin_fd: int):
    try:
        if original_stdin_fd != -1: # If original_stdin_fd was successfully saved
            current_stdin_fd = -1
            try:
                if sys.stdin and not sys.stdin.closed:
                    current_stdin_fd = sys.stdin.fileno()
                    sys.stdin.close()
            except Exception: # Ignore errors if sys.stdin is already weird
                pass

            try:
                os.dup2(original_stdin_fd, 0) # Restore original stdin fd to 0
                sys.stdin = open(0, 'r', closefd=False) # Re-open sys.stdin from new fd 0
            except Exception as e_dup: 
                sys.stderr.write(f"Warn: Restore stdin using dup2 failed: {e_dup}\n")
            finally:
                # Close the saved original_stdin_fd only if it's not fd 0 itself
                if original_stdin_fd != 0:
                    try: os.close(original_stdin_fd)
                    except OSError: pass
    except Exception as e: 
        sys.stderr.write(f"Error during stdin restoration: {e}\n")

# --- Modified main Function ---
def main():
    parser = argparse.ArgumentParser(
        description='AG - Ask GPT from CLI. Interact with OpenAI-compatible APIs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--api-url', help='API endpoint URL', default=DEFAULT_API_URL)
    parser.add_argument('--model', help='Model name to use', default=DEFAULT_MODEL)
    parser.add_argument('--model-type', help='Type of model API behavior', choices=['openai', 'gemini'], default=DEFAULT_MODEL_TYPE)
    parser.add_argument('--api-key', help='API key (or set OPENAI_API_KEY env var)', default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens for response', default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--frequency-penalty', type=float, help='Frequency penalty (e.g., 0.0 to 2.0). Use -1.0 to disable.', default=DEFAULT_FREQUENCY_PENALTY)
    parser.add_argument('--presence-penalty', type=float, help='Presence penalty (e.g., 0.0 to 2.0). Use -1.0 to disable.', default=DEFAULT_PRESENCE_PENALTY)
    parser.add_argument('-g', '--grep-enabled', action='store_true', help='Act as an AI-powered grep filter for piped input')
    parser.add_argument('-r', '--hide-reasoning', action='store_true', help='Hide reasoning content (<think> or model-specific), show spinner instead. Automatically enabled for --model-type gemini.')
    parser.add_argument('raw_text', nargs='?', help='Direct question (optional). If provided, runs once and exits.')

    args = parser.parse_args()

    if args.model_type == 'gemini':
        args.hide_reasoning = True

    freq_penalty = args.frequency_penalty if args.frequency_penalty != -1.0 else None
    pres_penalty = args.presence_penalty if args.presence_penalty != -1.0 else None

    try:
        client = OpenAI( base_url=args.api_url, api_key=args.api_key )
    except Exception as e:
        sys.exit(f"Error initializing OpenAI client: {e}")

    messages: List[Dict[str, str]] = []
    original_stdin_fd, pipe_input = setup_tty_stdin()
    readline_available = 'readline' in sys.modules and hasattr(readline, 'get_line_buffer')
    
    original_termios_settings = None
    # After setup_tty_stdin, sys.stdin should be the TTY if successful
    if TERMIOS_AVAILABLE and sys.stdin.isatty():
        try:
            fd = sys.stdin.fileno()
            original_termios_settings = set_terminal_no_echoctl(fd)
        except Exception as e_set_termios: # Catch if fileno() or isatty() fails for some reason
            sys.stderr.write(f"Warn: Could not get/set termios settings for stdin: {e_set_termios}\n")


    grep_prompt_template = """I want you to act strictly as a Linux `grep` command filter.
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
        if args.raw_text: # Single-shot mode
            user_content = args.raw_text
            if pipe_input:
                user_content = f"Context from piped input:\n{pipe_input.strip()}\n\nUser Query: {user_content}"
            if args.grep_enabled:
                if not pipe_input: sys.exit("Error: --grep-enabled requires piped input for single-shot mode.")
                messages = [{"role": "user", "content": grep_prompt_template.format(pattern=args.raw_text, input_text=pipe_input)}]
            else:
                messages.append({"role": "user", "content": process_input_directives(user_content)})
            response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            if response_content is not None and not response_content.endswith('\n'): sys.stdout.write("\n")
            sys.stdout.flush(); sys.exit(0)

        # Interactive mode
        sys.stderr.write("Entering interactive mode. Use Ctrl+D to submit, Ctrl+C to interact/exit.\n")
        initial_pipe_input = pipe_input 

        while True: # Outer loop for each full user query/command
            sys.stderr.write("\n💥 Ask (Ctrl+D Submit, !cmd, @file, @url):\n")
            lines_for_current_prompt = []
            
            while True: 
                prompt_char = "> " if not lines_for_current_prompt else "  "
                try:
                    current_line_text = input(prompt_char)
                    lines_for_current_prompt.append(current_line_text)
                
                except EOFError: 
                    sys.stderr.write(ANSI_GREEN + "^D EOF!\n" + ANSI_RESET) 
                    break 

                except KeyboardInterrupt: 
                    sys.stderr.write(ANSI_GREEN + "^C Cancelled!\n" + ANSI_RESET); # Script's own cancel message
                    
                    prompt_lines_has_content = False
                    if readline_available:
                        try:
                            if readline.get_line_buffer() or bool(lines_for_current_prompt): 
                                 prompt_lines_has_content = True
                        except Exception:
                            prompt_lines_has_content = False

                    if prompt_lines_has_content:
                        lines_for_current_prompt.clear()
                        # readline itself handles clearing the visual line on SIGINT if it's managing input.
                        # If ECHOCTL is off, no terminal-generated ^C should appear here.
                        continue 
                    else:
                        raise # Caught by outer try/except KeyboardInterrupt in main
            
            user_input = '\n'.join(lines_for_current_prompt).strip()
            
            if initial_pipe_input:
                user_input = f"Context from piped input:\n{initial_pipe_input.strip()}\n\nUser Query: {user_input}"
                initial_pipe_input = "" 
            
            if not user_input: 
                continue

            command_executed_this_turn, clear_command_issued = False, False
            user_input_for_llm = user_input 
            first_line_stripped = user_input.splitlines()[0].strip() if user_input else ""

            if first_line_stripped.startswith('!'):
                command_to_execute = first_line_stripped[1:].strip()
                if command_to_execute.lower() == 'clear':
                    messages.clear(); sys.stdout.write(ANSI_CLEAR_SCREEN); sys.stdout.flush()
                    sys.stderr.write("Conversation history cleared.\n")
                    clear_command_issued, command_executed_this_turn = True, True
                elif command_to_execute:
                    sys.stderr.write(f"\nExecuting: `{command_to_execute}`\n---\n")
                    output_from_command = execute_command(command_to_execute)
                    sys.stdout.write(output_from_command)
                    if not output_from_command.endswith('\n'): sys.stdout.write("\n")
                    sys.stdout.write("---\n"); sys.stdout.flush()
                    command_executed_this_turn, user_input_for_llm = True, ""
            
            if clear_command_issued or (command_executed_this_turn and user_input_for_llm == ""):
                continue 
            if not user_input_for_llm.strip():
                continue

            processed_input_for_llm = process_input_directives(user_input_for_llm)
            messages.append({"role": "user", "content": processed_input_for_llm})
            sys.stdout.write("💡:\n"); sys.stdout.flush()
            response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            if response_content is not None:
                messages.append({"role": "assistant", "content": response_content})
                if not response_content.endswith('\n') and response_content != "": sys.stdout.write("\n")
                sys.stdout.flush()

    except KeyboardInterrupt: 
        # This catches Ctrl+C if it was raised from the inner input loop (empty prompt)
        # or if stream_response re-raised it (which it currently doesn't, it returns None)
        # The main purpose is to allow graceful exit.
        # The "Cancelled!" message for this specific case (empty prompt Ctrl+C)
        # is already printed in the inner loop's exception handler.
        # Ensure a newline if ^C was pressed so term prompt isn't on same line.
        sys.stderr.write("\n") 
        pass 
    except Exception as e:
        sys.stderr.write(f"\nAn unexpected error occurred in main loop: {e}\n")
        traceback.print_exc(file=sys.stderr)
    finally:
        # Restore terminal settings first, before stdin FD is potentially changed by restore_stdin
        if TERMIOS_AVAILABLE and original_termios_settings is not None:
            current_stdin_is_tty = False
            current_stdin_fd = -1
            try:
                if sys.stdin and not sys.stdin.closed:
                    current_stdin_fd = sys.stdin.fileno()
                    if os.isatty(current_stdin_fd):
                        current_stdin_is_tty = True
            except Exception: # stdin might be in a weird state
                 pass 
            
            if current_stdin_is_tty:
                restore_terminal_settings(current_stdin_fd, original_termios_settings)
            # else: # Log if settings couldn't be restored (optional)
                # sys.stderr.write("Warn: Did not restore termios settings (stdin not a TTY or inaccessible at exit).\n")
        
        restore_stdin(original_stdin_fd)
        # Final newline to ensure shell prompt is clean
        sys.stderr.write(ANSI_RESET) # Ensure colors are reset before exiting.
        sys.stderr.flush()


if __name__ == '__main__':
    main()

