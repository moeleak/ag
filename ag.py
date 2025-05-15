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
# sys is imported again here, one import at the top is sufficient
import threading # Keep this one
import queue
import traceback
import wcwidth
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse

try:
    from googlesearch import search as google_search_func # Alias to avoid name collision
except ImportError:
    google_search_func = None # Placeholder if not installed

try:
    import readline
except ImportError:
    pass # readline is not available

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
DEFAULT_SEARCH_RESULTS_LIMIT = 10 # Number of search results to fetch by default
DEFAULT_SEARCH_SLEEP_INTERVAL = 0 # Sleep interval for google search (seconds)
USER_AGENT = "Mozilla/5.0 (Windows NT 1.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

ANSI_GREEN = '\033[32m'
ANSI_RESET = '\033[0m'
ANSI_SAVE_CURSOR = '\033[s'
ANSI_RESTORE_CURSOR = '\033[u'
ANSI_CLEAR_SCREEN = '\033[2J\033[H'
ANSI_EL = '\033[K' # Erase to end of line

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
    except requests.RequestException:
        return None
    except Exception:
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
    """Processes @file, @url, and @search directives in the input string."""
    processed_lines = []
    spinner_chars_local = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])

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
                try:
                    parsed_display_url_obj = urlparse(url)
                    display_url_at_url = f"{parsed_display_url_obj.scheme}://{parsed_display_url_obj.netloc}"
                    if len(display_url_at_url) > 50 : display_url_at_url = display_url_at_url[:47] + "..."
                except Exception:
                    display_url_at_url = url[:50] + "..." if len(url) > 50 else url

                sys.stderr.write(f"Fetching content from URL: {display_url_at_url} ... ")
                sys.stderr.flush()
                
                content_fetched_url = False
                q_url = queue.Queue()
                fetch_thread_url = threading.Thread(target=lambda func, u, q_res: q_res.put(func(u)), args=(fetch_webpage, url, q_url))
                fetch_thread_url.start()

                webpage_content_result = None
                while fetch_thread_url.is_alive():
                    sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... {next(spinner_chars_local)}")
                    sys.stderr.flush()
                    time.sleep(0.1)
                    try:
                        webpage_content_result = q_url.get_nowait()
                        content_fetched_url = True 
                        break
                    except queue.Empty:
                        continue
                
                fetch_thread_url.join() 
                
                if not content_fetched_url: 
                    try: 
                        webpage_content_result = q_url.get_nowait() 
                    except queue.Empty: 
                        webpage_content_result = None 

                if webpage_content_result:
                    sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... ✅ Done.\n")
                    processed_lines.append(f"--- Content from {url} ---\n{webpage_content_result}\n--- End of {url} ---")
                else:
                    sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... ❌ Failed.\n")
                sys.stderr.flush()
            else:
                 processed_lines.append(line) 
        elif stripped_line.startswith('@search'):
            if google_search_func is None:
                sys.stderr.write(f"{ANSI_EL}Error: The 'googlesearch-python' library is not installed. @search functionality is unavailable.\n"
                                 f"{ANSI_EL}Please install it using: pip install googlesearch-python\n")
                processed_lines.append(line) 
                continue

            match = re.match(r'@search\(([^)]+)\)', stripped_line)
            if match:
                query = match.group(1).strip()
                sys.stderr.write(f"Searching for '{query}'...  ") 
                sys.stderr.flush()
                
                search_results_urls = None 
                search_error_msg = None

                def perform_search_in_thread(q_search_results, search_query_param):
                    nonlocal search_error_msg
                    try:
                        results = list(google_search_func(
                            search_query_param,
                            num_results=DEFAULT_SEARCH_RESULTS_LIMIT,
                            lang='en',
                            sleep_interval=DEFAULT_SEARCH_SLEEP_INTERVAL 
                        ))
                        q_search_results.put(results)
                    except Exception as e:
                        search_error_msg = str(e) 
                        q_search_results.put(None) 

                search_q = queue.Queue()
                search_thread = threading.Thread(target=perform_search_in_thread, args=(search_q, query))
                search_thread.start()

                while search_thread.is_alive():
                    sys.stderr.write(f"\r{ANSI_EL}Searching for '{query}'... {next(spinner_chars_local)}")
                    sys.stderr.flush()
                    time.sleep(0.1)
                
                search_thread.join() 
                
                try:
                    search_results_urls = search_q.get_nowait() 
                except queue.Empty: 
                    if not search_error_msg: 
                        search_error_msg = "Search thread did not return a result."

                if search_error_msg: 
                    sys.stderr.write(f"\r{ANSI_EL}Searching for '{query}'... ❌ Error: {search_error_msg}\n")
                elif search_results_urls is None: 
                    sys.stderr.write(f"\r{ANSI_EL}Searching for '{query}'... ❌ An unspecified error occurred during search.\n")
                elif not search_results_urls: 
                    sys.stderr.write(f"\r{ANSI_EL}Searching for '{query}'... No results found.\n")
                else: 
                    sys.stderr.write(f"\r{ANSI_EL}Searching for '{query}'... ✅ Found {len(search_results_urls)} results.\n")
                sys.stderr.flush()

                if search_results_urls: 
                    num_total_results = len(search_results_urls)
                    processed_lines.append(f"--- Search results for query: \"{query}\" (top {num_total_results}) ---")
                    
                    base_fetch_msg_template = f"Fetching content from {num_total_results} result(s)"
                    sys.stderr.write(f"{base_fetch_msg_template} [0/{num_total_results}]...")
                    sys.stderr.flush()
                    
                    successful_fetches = 0

                    for i, url_from_search in enumerate(search_results_urls):
                        try:
                            parsed_url = urlparse(url_from_search)
                            netloc_display = parsed_url.netloc
                            if len(netloc_display) > 30: 
                                netloc_display = netloc_display[:27] + "..."
                            display_url_segment = f"{parsed_url.scheme}://{netloc_display}"
                        except Exception: 
                            display_url_segment = url_from_search[:30] + "..." if len(url_from_search) > 30 else url_from_search
                        
                        progress_text = f"[{i+1}/{num_total_results}] ({display_url_segment})"
                        sys.stderr.write(f"\r{ANSI_EL}{base_fetch_msg_template} {progress_text}... {next(spinner_chars_local)}")
                        sys.stderr.flush()
                        
                        q_fetch_search = queue.Queue()
                        fetch_thread_search = threading.Thread(target=lambda func, u, q_res: q_res.put(func(u)), 
                                                               args=(fetch_webpage, url_from_search, q_fetch_search))
                        fetch_thread_search.start()

                        webpage_content_from_search = None
                        content_fetched_search = False
                        while fetch_thread_search.is_alive():
                            sys.stderr.write(f"\r{ANSI_EL}{base_fetch_msg_template} {progress_text}... {next(spinner_chars_local)}")
                            sys.stderr.flush()
                            time.sleep(0.1)
                            try:
                                webpage_content_from_search = q_fetch_search.get_nowait()
                                content_fetched_search = True
                                break 
                            except queue.Empty:
                                continue
                        
                        fetch_thread_search.join() 

                        if not content_fetched_search: 
                           try: 
                               webpage_content_from_search = q_fetch_search.get_nowait() 
                           except queue.Empty: 
                               webpage_content_from_search = None 

                        if webpage_content_from_search:
                            successful_fetches += 1
                            processed_lines.append(f"  --- Content from search result: {url_from_search} ---\n{webpage_content_from_search}\n  --- End of content for {url_from_search} ---")
                        else:
                            processed_lines.append(f"  --- Could not fetch content from search result: {url_from_search} ---")
                    
                    final_status_symbol = "✅" if successful_fetches == num_total_results else ("⚠️" if successful_fetches > 0 else "❌")
                    final_summary_msg = f"\r{ANSI_EL}{base_fetch_msg_template}... {final_status_symbol} Done. ({successful_fetches}/{num_total_results} succeeded)\n"
                    sys.stderr.write(final_summary_msg)
                    sys.stderr.flush()

                    processed_lines.append(f"--- End of search results for \"{query}\" ---")
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
            if messages[-1]['role'] == 'user':
                 messages[-1]['content'] = f"{language_prompt}\n\n{messages[-1]['content']}"

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
        nonlocal spinner_active, first_content_received_gemini
        if model_type == 'gemini' and first_content_received_gemini:
            return
        clear_thinking_indicator_if_on_screen() 
        if not spinner_active:
            sys.stderr.write("Thinking... "); spinner_active = True; sys.stderr.flush()

    def update_spinner():
        nonlocal first_content_received_gemini
        if model_type == 'gemini' and first_content_received_gemini:
            return
        if spinner_active:
            sys.stderr.write(f"\r{ANSI_EL}Thinking... {next(spinner_chars)}"); sys.stderr.flush()

    def stop_spinner(success=True):
        nonlocal spinner_active, strip_leading_newline_next_write
        if spinner_active:
            sys.stderr.write(f"\r{ANSI_EL}Thinking... {'✅' if success else '❌'}\n") 
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
                    
                    reasoning_content = getattr(delta, 'reasoning_content', None)
                    content = delta.content
                    
                    if model_type == 'gemini' and content and not first_content_received_gemini and spinner_active:
                        stop_spinner() 
                        first_content_received_gemini = True

                    if reasoning_content:
                        if hide_reasoning:
                            if not spinner_active: start_spinner() 
                            update_spinner()
                        else:
                            if spinner_active: stop_spinner() 
                            if not is_reasoning_deepseek: 
                                clear_thinking_indicator_if_on_screen() 
                                is_reasoning_deepseek = True
                                sys.stdout.write(REASONING_START_MARKER + "\n" + ANSI_GREEN); sys.stdout.flush()
                                strip_leading_newline_next_write = False 
                            write_thinking_chunk_with_indicator(reasoning_content)
                        continue 
                    
                    if content: 
                        if is_reasoning_deepseek: 
                            clear_thinking_indicator_if_on_screen() 
                            is_reasoning_deepseek = False 
                            if not hide_reasoning:
                                sys.stdout.write(ANSI_RESET + REASONING_END_MARKER + "\n"); sys.stdout.flush()
                                strip_leading_newline_next_write = True 
                        
                        if model_type != 'gemini' and spinner_active: 
                            stop_spinner()

                        full_response_chunks.append(content) 
                        buffer += content 
                        processed_chunk_for_history_this_delta = ''
                        output_to_print_for_normal_content = ""
                        
                        current_processing_buffer = buffer 
                        buffer = '' 

                        temp_idx = 0
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

                                    temp_idx = think_start_pos + len('<think>')
                                    in_think_block_tag = True 
                                    if hide_reasoning:
                                        if not spinner_active: start_spinner()
                                        update_spinner()
                                    else: 
                                        if spinner_active: stop_spinner() 
                                        sys.stdout.write(ANSI_GREEN); sys.stdout.flush() 
                                        strip_leading_newline_next_write = False
                            else: 
                                think_end_pos = current_processing_buffer.find('</think>', temp_idx)
                                if think_end_pos == -1: 
                                    think_content_chunk = current_processing_buffer[temp_idx:]
                                    temp_idx = len(current_processing_buffer) 
                                    if hide_reasoning: update_spinner()
                                    else: write_thinking_chunk_with_indicator(think_content_chunk)
                                else: 
                                    think_content_chunk = current_processing_buffer[temp_idx:think_end_pos]
                                    if not hide_reasoning: 
                                        write_thinking_chunk_with_indicator(think_content_chunk) 
                                        clear_thinking_indicator_if_on_screen() 
                                        sys.stdout.write(ANSI_RESET); sys.stdout.flush() 
                                        strip_leading_newline_next_write = True 
                                    elif hide_reasoning:
                                         update_spinner() 
                                    
                                    temp_idx = think_end_pos + len('</think>')
                                    in_think_block_tag = False 
                        
                        if output_to_print_for_normal_content: 
                            clear_thinking_indicator_if_on_screen()
                            if strip_leading_newline_next_write:
                                output_to_print_for_normal_content = output_to_print_for_normal_content.lstrip('\n')
                                if output_to_print_for_normal_content: strip_leading_newline_next_write = False
                            sys.stdout.write(output_to_print_for_normal_content); sys.stdout.flush()

                        if processed_chunk_for_history_this_delta:
                            processed_response_chunks.append(processed_chunk_for_history_this_delta)

                elif item_type == 'DONE':
                    clear_thinking_indicator_if_on_screen() 
                    break 
                elif item_type == 'ERROR':
                    final_status_success = False 
                    clear_thinking_indicator_if_on_screen()
                    stop_spinner(success=False) 
                    if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
                    sys.stdout.flush() 
                    sys.stderr.write(f"\n{data}\n") 
                    return None 
            
            except queue.Empty: 
                if spinner_active: update_spinner() 
                if not worker_thread.is_alive(): 
                    clear_thinking_indicator_if_on_screen()
                    try:
                        item_type, data = q.get_nowait()
                        if item_type == 'DONE': 
                            break 
                        if item_type == 'ERROR': 
                            final_status_success = False; stop_spinner(success=False)
                            if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
                            sys.stdout.flush(); sys.stderr.write(f"\n{data}\n"); return None
                    except queue.Empty: 
                        if final_status_success: 
                            final_status_success = False; stop_spinner(success=False)
                            sys.stderr.write(f"\n{ANSI_EL}Worker thread finished unexpectedly.\n")
                        return None 
                    break 
        
        clear_thinking_indicator_if_on_screen()
        if spinner_active: stop_spinner(success=final_status_success) 
        
        if not hide_reasoning:
            if is_reasoning_deepseek: 
                sys.stdout.write(ANSI_RESET) 
                sys.stdout.write("\n" + REASONING_END_MARKER + "\n") 
            if in_think_block_tag: 
                sys.stdout.write(ANSI_RESET) 
        sys.stdout.flush()

        processed_response = "".join(processed_response_chunks)
        return processed_response.lstrip('\n') if processed_response else "" 

    except KeyboardInterrupt:
        final_status_success = False 
        sys.stdout.write(ANSI_RESET); sys.stderr.write(ANSI_RESET) 
        sys.stdout.flush(); sys.stderr.flush()
        
        clear_thinking_indicator_if_on_screen(); stop_spinner(success=False) 

        if not hide_reasoning: 
            if is_reasoning_deepseek or in_think_block_tag:
                 sys.stdout.write(ANSI_RESET); sys.stdout.flush() 
                 sys.stderr.write(f"\n{REASONING_END_MARKER if is_reasoning_deepseek else '</think>'}\n(Interrupted)\n")

        sys.stderr.write(f"{ANSI_GREEN}\n^C Cancelled Stream!\n{ANSI_RESET}"); sys.stderr.flush()
        
        if TERMIOS_AVAILABLE:
            try:
                if sys.stdin.isatty() and sys.stdin.fileno() >= 0:
                    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except Exception as e_flush:
                sys.stderr.write(f"Warn: Flushing stdin (TCIFLUSH) failed: {e_flush}\n"); sys.stderr.flush()
        return None 
    except Exception as e:
        final_status_success = False 
        clear_thinking_indicator_if_on_screen(); stop_spinner(success=False)
        if not hide_reasoning and (is_reasoning_deepseek or in_think_block_tag): sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()
        sys.stderr.write(f"\n{ANSI_EL}An unexpected error occurred in stream_response: {str(e)}\n")
        traceback.print_exc(file=sys.stderr)
        return None 
    finally:
        clear_thinking_indicator_if_on_screen()
        if 'spinner_active' in locals() and spinner_active : 
             sys.stderr.write(f"\r{ANSI_EL}"); sys.stderr.flush() 
        
        if 'worker_thread' in locals() and worker_thread.is_alive(): 
            worker_thread.join(timeout=1.0) 

# --- Terminal Control Helper Functions (set_terminal_no_echoctl, restore_terminal_settings) ---
def set_terminal_no_echoctl(fd: int) -> Optional[List]:
    if not TERMIOS_AVAILABLE or not os.isatty(fd):
        return None
    try:
        old_settings = termios.tcgetattr(fd)
        new_settings = list(old_settings) 
        new_settings[3] &= ~termios.ECHOCTL  
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        return old_settings
    except termios.error:
        return None

def restore_terminal_settings(fd: int, old_settings: Optional[List]):
    if not TERMIOS_AVAILABLE or not os.isatty(fd) or old_settings is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except termios.error:
        pass 

# --- Readline completer for directives and commands ---
COMPLETION_TOKENS = ["@file(", "@url(", "@search(", "!clear"]

def custom_directive_completer(text: str, state: int) -> Optional[str]:
    """Completer for @directives and !commands."""
    
    # --- Begin Debug Prints for Completer ---
    # These will print to stderr every time tab is pressed in interactive mode
    # sys.stderr.write(f"DEBUG_COMPLETER: text_arg='{text}', state={state}\n") # Using sys.stderr.write for atomicity
    line_buffer_debug = "N/A"
    begin_idx_debug = -1
    end_idx_debug = -1
    current_typing_debug = "N/A" 

    if 'readline' in sys.modules and hasattr(readline, 'get_line_buffer'): # Check specific functions
        try:
            line_buffer_debug = readline.get_line_buffer()
            begin_idx_debug = readline.get_begidx()
            end_idx_debug = readline.get_endidx()
            # Only slice if indices are valid, otherwise, it can cause issues.
            if begin_idx_debug != -1 and end_idx_debug != -1 and begin_idx_debug <= end_idx_debug :
                current_typing_debug = line_buffer_debug[begin_idx_debug:end_idx_debug]
            else: # If indices are not valid, fall back to the text argument
                current_typing_debug = text 
        except Exception as e:
            sys.stderr.write(f"DEBUG_COMPLETER: Error getting readline info: {e}\n")
            current_typing_debug = text # Fallback to text arg if readline functions fail
            
    # sys.stderr.write(f"DEBUG_COMPLETER: text_arg='{text}', state={state}, line_buffer='{line_buffer_debug}', "
    #                  f"begin_idx={begin_idx_debug}, end_idx={end_idx_debug}, "
    #                  f"current_typing='{current_typing_debug}'\n")
    sys.stderr.flush() # Ensure debug output is seen immediately
    # --- End Debug Prints for Completer ---

    current_typing_for_logic = current_typing_debug

    options = [token for token in COMPLETION_TOKENS if token.startswith(current_typing_for_logic)]
    # sys.stderr.write(f"DEBUG_COMPLETER: options_found={options}\n")
    sys.stderr.flush() 
    
    if state < len(options):
        # sys.stderr.write(f"DEBUG_COMPLETER: Returning option: {options[state]}\n")
        sys.stderr.flush()
        return options[state]
    else:
        # sys.stderr.write(f"DEBUG_COMPLETER: No more options for this state.\n")
        sys.stderr.flush()
        return None

# --- setup_tty_stdin and restore_stdin ---
def setup_tty_stdin() -> Tuple[int, Optional[str]]:
    original_stdin_fd = -1
    pipe_input_content = "" 
    readline_available_flag = ('readline' in sys.modules and 
                               hasattr(sys.modules['readline'], 'read_history_file') and
                               hasattr(sys.modules['readline'], 'set_completer') and
                               hasattr(sys.modules['readline'], 'parse_and_bind') and
                               hasattr(sys.modules['readline'], 'get_line_buffer') and 
                               hasattr(sys.modules['readline'], 'get_begidx') and    
                               hasattr(sys.modules['readline'], 'get_endidx') and
                               hasattr(sys.modules['readline'], 'get_completer_delims')) # Added check for get_completer_delims

    try:
        original_stdin_fd = os.dup(0) 
        if not sys.stdin.isatty(): 
            # sys.stderr.write("DEBUG_SETUP: Initial stdin is not a TTY. Attempting to switch to /dev/tty.\n")
            pipe_input_content = sys.stdin.read() 
            sys.stdin.close() 
            try:
                new_stdin_fd = os.open('/dev/tty', os.O_RDONLY)
                os.dup2(new_stdin_fd, 0) 
                os.close(new_stdin_fd) 
                sys.stdin = open(0, 'r', closefd=False) 
                # sys.stderr.write("DEBUG_SETUP: Successfully switched stdin to /dev/tty.\n")
            except OSError as e:
                # sys.stderr.write(f"WARN_SETUP: Could not switch to TTY for interactive input: {e}\n")
                # sys.stderr.write("WARN_SETUP: Fallback to original (non-TTY) stdin. Interactive mode might be limited.\n")
                try:
                    os.dup2(original_stdin_fd, 0)
                    sys.stdin = open(0, 'r', closefd=False)
                    # sys.stderr.write("DEBUG_SETUP: Restored original (non-TTY) stdin.\n")
                except OSError as restore_e:
                     sys.stderr.write(f"CRITICAL_SETUP: Could not restore original stdin after TTY switch failure: {restore_e}\n")
        
        if sys.stdin.isatty() and readline_available_flag:
            # sys.stderr.write("DEBUG_SETUP: stdin is a TTY and readline is available. Setting up readline features.\n")
            try:
                default_delims = readline.get_completer_delims()
                # sys.stderr.write(f"DEBUG_SETUP: Default readline completer delimiters: '{default_delims}'\n")
                
                # <<<< KEY CHANGE: MODIFY DELIMITERS >>>>
                # If '@' or '!' are in default_delims, completion of "@s" might not work as expected.
                # Let's try removing common special characters used by our directives from delimiters.
                new_delims = default_delims
                for char_to_remove in ['@', '!', '(', ')']: # Remove chars that are part of our tokens
                    new_delims = new_delims.replace(char_to_remove, '')
                
                if new_delims != default_delims:
                    readline.set_completer_delims(new_delims)
                    # sys.stderr.write(f"DEBUG_SETUP: MODIFIED readline completer delimiters to: '{new_delims}'\n")
                # else:
                #    sys.stderr.write(f"DEBUG_SETUP: Default delimiters did not contain '@', '!', '(', ')'. No changes made.\n")


                readline.set_completer(custom_directive_completer)
                readline.parse_and_bind('tab: complete')

                history_file = os.path.join(os.path.expanduser("~"), ".ag_history")
                try: readline.read_history_file(history_file)
                except FileNotFoundError: pass 
                except Exception: pass 
                
                import atexit
                def save_history():
                    try: readline.write_history_file(history_file)
                    except Exception: pass 
                atexit.register(save_history)
                # sys.stderr.write("DEBUG_SETUP: Readline history and completer setup complete.\n")
            except Exception as e_readline:
                # sys.stderr.write(f"WARN_SETUP: Readline setup failed with exception: {e_readline}\n")
                traceback.print_exc(file=sys.stderr) 
        elif not sys.stdin.isatty():
            None
        elif not readline_available_flag:
            # sys.stderr.write("DEBUG_SETUP: readline module not available or not fully functional, completion skipped.\n")
            if 'readline' not in sys.modules:
                None
                # sys.stderr.write("DEBUG_SETUP: 'readline' module not in sys.modules. Please install it (e.g., 'pip install gnureadline' on macOS, 'pyreadline3' on Windows).\n")
            else:
                missing_attrs = []
                for attr in ['read_history_file', 'set_completer', 'parse_and_bind', 'get_line_buffer', 
                             'get_begidx', 'get_endidx', 'get_completer_delims']:
                    if not hasattr(sys.modules['readline'], attr): missing_attrs.append(attr)
                # if missing_attrs:
                #     sys.stderr.write(f"DEBUG_SETUP: 'readline' module is imported but missing attributes: {', '.join(missing_attrs)}\n")
                # else:
                #     sys.stderr.write(f"DEBUG_SETUP: 'readline' module is imported and all checked attributes are present.\n")


    except Exception as e:
        # sys.stderr.write(f"ERROR_SETUP: Error during TTY/stdin setup: {e}\n")
        traceback.print_exc(file=sys.stderr)
        if original_stdin_fd != -1 and original_stdin_fd != 0: 
            try: os.close(original_stdin_fd) 
            except OSError: pass 
            original_stdin_fd = -1 
            
    return original_stdin_fd, pipe_input_content

def restore_stdin(original_stdin_fd: int):
    try:
        if original_stdin_fd != -1: 
            current_stdin_fd = -1
            try:
                if sys.stdin and not sys.stdin.closed:
                    current_stdin_fd = sys.stdin.fileno() 
                    sys.stdin.close()
            except Exception: 
                pass

            try:
                os.dup2(original_stdin_fd, 0) 
                sys.stdin = open(0, 'r', closefd=False) 
            except Exception: 
                pass
            finally:
                if original_stdin_fd != 0: 
                    try: os.close(original_stdin_fd)
                    except OSError: pass
    except Exception: 
        pass

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
    original_stdin_fd, initial_pipe_input_content = setup_tty_stdin() 
    
    is_readline_available_in_main = ('readline' in sys.modules and 
                                   hasattr(sys.modules['readline'], 'get_line_buffer'))
    
    original_termios_settings = None
    if TERMIOS_AVAILABLE and sys.stdin.isatty(): 
        try:
            fd_val = sys.stdin.fileno() 
            if fd_val >=0: 
                 original_termios_settings = set_terminal_no_echoctl(fd_val)
        except Exception: 
            pass

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
    
    has_used_initial_pipe_input = False

    try: 
        if args.raw_text: 
            user_content = args.raw_text
            if initial_pipe_input_content: 
                user_content = f"Context from piped input:\n{initial_pipe_input_content.strip()}\n\nUser Query: {user_content}"
            
            processed_user_content = process_input_directives(user_content)

            if args.grep_enabled:
                if not initial_pipe_input_content: 
                     sys.exit("Error: --grep-enabled requires piped input when providing a direct question (raw_text).")
                messages = [{"role": "user", "content": grep_prompt_template.format(pattern=args.raw_text, input_text=initial_pipe_input_content)}]
            else:
                messages = [{"role": "user", "content": processed_user_content}]

            response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            if response_content is not None and not response_content.endswith('\n'): sys.stdout.write("\n") 
            sys.stdout.flush(); sys.exit(0) 

        sys.stderr.write("Entering interactive mode. Use Ctrl+D to submit, !cmd, @file, @url, @search, Tab for completion:\n")
        
        while True: 
            sys.stderr.write(f"\n{ANSI_RESET}💥 Ask (Ctrl+D Submit, !cmd, @file, @url, @search, Tab completion):\n")
            
            current_prompt_accumulator: List[str] = [] 

            if initial_pipe_input_content and not has_used_initial_pipe_input:
                current_prompt_accumulator.append(f"Context from initial piped input:\n{initial_pipe_input_content.strip()}")
            while True: 
                current_accumulated_str_for_prompt_char = '\n'.join(current_prompt_accumulator).strip()
                prompt_char = "  " if current_accumulated_str_for_prompt_char else "> " 
                
                try:
                    user_line_input = input(prompt_char) 
                    stripped_user_line = user_line_input.strip()

                    if stripped_user_line.startswith('!'):
                        command_str_from_input = stripped_user_line[1:].strip()
                        
                        if command_str_from_input.lower() == 'clear':
                            messages.clear() 
                            current_prompt_accumulator.clear() 
                            sys.stdout.write(ANSI_CLEAR_SCREEN); sys.stdout.flush() 
                            sys.stderr.write("Conversation history and current input cleared.\n")
                            break 
                        elif command_str_from_input: 
                            sys.stderr.write(f"\nExecuting: `{command_str_from_input}`\n---\n")
                            command_output_str = execute_command(command_str_from_input)
                            sys.stderr.write(command_output_str) 
                            if not command_output_str.endswith('\n'): sys.stderr.write("\n") 
                            sys.stderr.write("---\n"); sys.stderr.flush()
                            
                            formatted_cmd_output_for_prompt = (
                                f"--- User executed command `{command_str_from_input}` "
                                f"which produced the following output ---\n{command_output_str.strip()}\n"
                                f"--- End of command output ---"
                            )
                            current_prompt_accumulator.append(formatted_cmd_output_for_prompt)
                            sys.stderr.write("Command output added to current prompt. Continue typing or Ctrl+D to submit.\n")
                        else: 
                            current_prompt_accumulator.append(user_line_input) 
                    else: 
                        current_prompt_accumulator.append(user_line_input)

                except EOFError: 
                    sys.stderr.write(ANSI_GREEN + "^D EOF!\n" + ANSI_RESET) 
                    break 

                except KeyboardInterrupt: 
                    sys.stderr.write(ANSI_GREEN + "\n^C Cancelled!\n" + ANSI_RESET) 
                    
                    readline_buffer_non_empty = False
                    if is_readline_available_in_main: 
                        try:
                            if 'readline' in sys.modules and hasattr(sys.modules['readline'], 'get_line_buffer'):
                                if sys.modules['readline'].get_line_buffer(): readline_buffer_non_empty = True
                        except Exception: pass 
                    
                    current_input_has_content = bool(''.join(current_prompt_accumulator).strip()) or readline_buffer_non_empty

                    if current_input_has_content:
                        current_prompt_accumulator.clear() 
                        # sys.stderr.write("\nCurrent input cleared due to ^C. Enter new input or Ctrl+D/Ctrl+C again.\n")
                    else: 
                        sys.stderr.write(ANSI_RESET); sys.stderr.flush()
                        raise 
            
            final_prompt_str_for_llm = '\n'.join(current_prompt_accumulator).strip()
            
            if not final_prompt_str_for_llm: 
                continue 

            if initial_pipe_input_content and not has_used_initial_pipe_input and \
               any(initial_pipe_input_content.strip() in part for part in current_prompt_accumulator):
                has_used_initial_pipe_input = True
            
            processed_input_for_llm = process_input_directives(final_prompt_str_for_llm)
            messages.append({"role": "user", "content": processed_input_for_llm})
            
            sys.stdout.write(f"{ANSI_RESET}💡:\n"); sys.stdout.flush() 
            response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            
            if response_content is not None: 
                messages.append({"role": "assistant", "content": response_content}) 
                if response_content and not response_content.endswith('\n'): 
                    sys.stdout.write("\n") 
                sys.stdout.flush()
            sys.stdout.write(ANSI_RESET) 
            sys.stdout.flush()

    except KeyboardInterrupt: 
        pass 
    except Exception as e: 
        sys.stderr.write(f"\n{ANSI_RESET}An unexpected error occurred in main loop: {e}\n")
        traceback.print_exc(file=sys.stderr)
    finally:
        if TERMIOS_AVAILABLE and original_termios_settings is not None:
            current_stdin_is_tty = False; current_stdin_fd_val = -1
            try: 
                if sys.stdin and not sys.stdin.closed:
                    current_stdin_fd_val = sys.stdin.fileno()
                    if current_stdin_fd_val >=0 and os.isatty(current_stdin_fd_val):
                        current_stdin_is_tty = True
            except Exception: pass 
            
            if current_stdin_is_tty: 
                restore_terminal_settings(current_stdin_fd_val, original_termios_settings)
        
        restore_stdin(original_stdin_fd) 
        sys.stderr.write(ANSI_RESET); sys.stdout.write(ANSI_RESET) 
        sys.stderr.flush(); sys.stdout.flush()

if __name__ == '__main__':
    main()

