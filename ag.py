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
import threading
import queue
import traceback
import wcwidth
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse
import glob # for file path completion

# --- prompt_toolkit imports ---
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import ANSI
    # from prompt_toolkit.key_binding import KeyBindings # Not used for now
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    # Fallback for systems where prompt_toolkit might not be installed,
    # though the goal is to make it a primary dependency for interactive mode.
    # For this conversion, we assume it's available.
    if not PROMPT_TOOLKIT_AVAILABLE:
        print("Warning: prompt_toolkit is not installed. Interactive mode will have limited features or fail.", file=sys.stderr)


try:
    # readline is no longer the primary for interactive mode, but keep for fallback or non-interactive if ever needed.
    # For this specific request, we're removing its active usage in interactive mode.
    pass # import readline # No longer explicitly used for prompt
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

# Default parameters for keyword extraction LLM (can be overridden by CLI args)
DEFAULT_SEARCH_KW_MAX_TOKENS = 1000
DEFAULT_SEARCH_KW_TEMPERATURE = 0.2


ANSI_GREEN = '\033[32m'
ANSI_MAGENTA = '\033[35m'
ANSI_RESET = '\033[0m'
ANSI_SAVE_CURSOR = '\033[s'
ANSI_RESTORE_CURSOR = '\033[u'
ANSI_CLEAR_SCREEN = '\033[2J\033[H'
ANSI_EL = '\033[K' # Erase to end of line

# --- Helper Functions (fetch_webpage, execute_command, process_input_directives) ---
# These functions remain largely unchanged as they don't directly deal with readline
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
    except Exception: # Catching other potential errors during parsing
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


def process_input_directives(
    input_str: str,
    main_llm_client: Optional[OpenAI] = None,
    main_llm_model: Optional[str] = None,
    # Parameters for dedicated keyword LLM, passed from args
    search_kw_api_url_param: Optional[str] = None,
    search_kw_model_name_param: Optional[str] = None,
    search_kw_api_key_param: Optional[str] = None,
    search_kw_max_tokens_param: int = DEFAULT_SEARCH_KW_MAX_TOKENS,
    search_kw_temperature_param: float = DEFAULT_SEARCH_KW_TEMPERATURE
) -> str:
    """Processes @file, @url, and @search directives in the input string.
    If an LLM client and model are available (either main or dedicated for keywords),
    @search will attempt keyword extraction."""
    processed_lines = []
    spinner_chars_local = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])

    for line in input_str.splitlines():
        stripped_line = line.strip()
        if stripped_line.startswith('@file'):
            match = re.match(r'@file\(([^)]+)\)', stripped_line)
            if match:
                filename = match.group(1).strip()
                # Handle cases where filename might be quoted
                if (filename.startswith("'") and filename.endswith("'")) or \
                   (filename.startswith('"') and filename.endswith('"')):
                    filename = filename[1:-1]
                
                try:
                    expanded_filename = os.path.expanduser(os.path.expandvars(filename))
                    with open(expanded_filename, 'r', encoding='utf-8') as f:
                        processed_lines.append(f"--- Content of {filename} (resolved to {expanded_filename}) ---\n{f.read()}\n--- End of {filename} ---")
                except FileNotFoundError:
                    sys.stderr.write(f"Error: File not found: {filename} (resolved to {expanded_filename if 'expanded_filename' in locals() else filename})\n")
                except IOError as e:
                    sys.stderr.write(f"Error reading {filename}: {str(e)}\n")
                except Exception as e:
                     sys.stderr.write(f"Unexpected error processing file {filename}: {str(e)}\n")
            else:
                 processed_lines.append(line) # Malformed @file directive
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
                fetch_thread_url = threading.Thread(target=lambda func, u, q_res: q_res.put(func(u)), args=(fetch_webpage, url, q_url), daemon=True)
                fetch_thread_url.start()

                webpage_content_result = None
                try:
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
                except KeyboardInterrupt:
                    sys.stderr.write(f"\r{ANSI_EL}\n^C @url fetch for \"{display_url_at_url}\" cancelled.\n{ANSI_RESET}")
                    sys.stderr.flush()
                    processed_lines.append(f"--- @url fetch for {url} was cancelled by user ---")
                    continue # Continue to the next line in input_str

                if not content_fetched_url:
                    try:
                        webpage_content_result = q_url.get_nowait()
                    except queue.Empty:
                        webpage_content_result = None

                if webpage_content_result:
                    sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... ✅ Done.\n")
                    processed_lines.append(f"--- Content from {url} ---\n{webpage_content_result}\n--- End of {url} ---")
                else:
                    if not any(f"cancelled by user" in s for s in processed_lines if url in s): # Check if already cancelled
                        sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... ❌ Failed.\n")
                sys.stderr.flush()
            else:
                 processed_lines.append(line) # Malformed @url directive
        elif stripped_line.startswith('@search'):
            # Ensure google_search_func is imported or defined (placeholder in original code)
            try:
                from googlesearch import search as google_search_func
            except ImportError:
                google_search_func = None

            if google_search_func is None:
                sys.stderr.write(f"{ANSI_EL}Error: The 'googlesearch-python' library is not installed. @search functionality is unavailable.\n"
                                 f"{ANSI_EL}Please install it using: pip install googlesearch-python\n")
                processed_lines.append(line) # Keep the original @search directive text
                continue

            match = re.match(r'@search\(([^)]+)\)', stripped_line)
            if match:
                original_query = ""
                try:
                    original_query = match.group(1).strip()
                    query_to_search = original_query 

                    def clean_keywords_output(raw_keywords: Optional[str]) -> str:
                        if not raw_keywords: return ""
                        text = raw_keywords.strip()
                        if not text: return ""
                        think_start_tag = "<think>"
                        think_end_tag = "</think>"
                        last_end_tag_pos = text.rfind(think_end_tag)
                        if last_end_tag_pos != -1:
                            content_after = text[last_end_tag_pos + len(think_end_tag):].strip()
                            if content_after: return content_after
                            else: return "" 
                        else:
                            if think_start_tag in text: return ""
                            else: return text
                    
                    llm_for_keywords_enabled = False
                    final_kw_client = None
                    final_kw_model = None

                    if search_kw_model_name_param: 
                        chosen_kw_api_url = search_kw_api_url_param
                        if not chosen_kw_api_url: 
                            chosen_kw_api_url = str(main_llm_client.base_url) if main_llm_client and main_llm_client.base_url else DEFAULT_API_URL
                        
                        chosen_kw_api_key = search_kw_api_key_param
                        if chosen_kw_api_key is None and main_llm_client : 
                            chosen_kw_api_key = main_llm_client.api_key
                        
                        if chosen_kw_api_url and not isinstance(chosen_kw_api_url, str):
                            chosen_kw_api_url = str(chosen_kw_api_url)

                        try:
                            if not chosen_kw_api_url:
                                raise ValueError("Keyword LLM API URL is empty or None after fallbacks.")
                            final_kw_client = OpenAI(base_url=chosen_kw_api_url, api_key=chosen_kw_api_key)
                            final_kw_model = search_kw_model_name_param
                            llm_for_keywords_enabled = True
                            sys.stderr.write(f"Using dedicated LLM for keywords: {final_kw_model} at {chosen_kw_api_url}\n")
                        except Exception as e_kw_client:
                            sys.stderr.write(f"\r{ANSI_EL}Error initializing dedicated keyword LLM client ({search_kw_model_name_param}): {e_kw_client}. Keyword LLM disabled. ⚠️\n")
                    elif main_llm_client and main_llm_model: 
                        final_kw_client = main_llm_client
                        final_kw_model = main_llm_model
                        llm_for_keywords_enabled = True

                    if llm_for_keywords_enabled and final_kw_client and final_kw_model:
                        q_kw = queue.Queue()
                        
                        def extract_keywords_task_threaded(q_result, client_param, model_param, messages_param, max_tokens_param_kw, temperature_param_kw):
                            try:
                                completion_kw = client_param.chat.completions.create(
                                    model=model_param, messages=messages_param,
                                    max_tokens=max_tokens_param_kw, temperature=temperature_param_kw )
                                if completion_kw.choices and completion_kw.choices[0].message and completion_kw.choices[0].message.content:
                                    res = completion_kw.choices[0].message.content 
                                    q_result.put(('SUCCESS', res if res else None)) 
                                else:
                                    q_result.put(('FAILURE', "No content in LLM response for keywords."))
                            except APIError as e_api_kw: q_result.put(('ERROR', f"API Error during keyword extraction: {str(e_api_kw)}"))
                            except Exception as e_task_kw: q_result.put(('ERROR', f"Unexpected error during keyword extraction: {str(e_task_kw)}"))

                        keyword_extraction_prompt_system = (
                            "You are an AI assistant specialized in optimizing search queries. From the user's input, "
                            "extract the core keywords that would be most effective for a web search.\n"
                            "- Return ONLY the keywords.\n"
                            "- Keywords should be separated by a single space.\n"
                            "- Do not include any explanations, introductions, or conversational text.\n"
                            "- Focus on proper nouns, technical terms, and essential concepts from the query.\n"
                            "- If the query is already very short and keyword-like, you can return it as is." )
                        kw_messages = [{"role": "system", "content": keyword_extraction_prompt_system},
                                       {"role": "user", "content": f"Query: \"{original_query}\""}]
                        
                        kw_thread = threading.Thread(target=extract_keywords_task_threaded, 
                                                     args=(q_kw, final_kw_client, final_kw_model, kw_messages, 
                                                           search_kw_max_tokens_param, search_kw_temperature_param), daemon=True)
                        
                        sys.stderr.write(f"Extracting keywords using {final_kw_model}... ")
                        sys.stderr.flush(); kw_thread.start()
                        kw_extraction_done, extracted_keywords_from_llm, status_kw_extraction, kw_data_from_q_error_info = False, None, "PENDING", None 

                        while kw_thread.is_alive():
                            sys.stderr.write(f"\r{ANSI_EL}Extracting keywords using {final_kw_model}... {next(spinner_chars_local)}")
                            sys.stderr.flush(); time.sleep(0.1)
                            try:
                                status_type_kw, kw_data_from_q = q_kw.get_nowait()
                                if status_type_kw == 'SUCCESS': extracted_keywords_from_llm = kw_data_from_q; status_kw_extraction = "SUCCESS"
                                elif status_type_kw == 'FAILURE': status_kw_extraction = "FAILURE_API"; kw_data_from_q_error_info = kw_data_from_q
                                elif status_type_kw == 'ERROR': status_kw_extraction = "ERROR_THREAD"; kw_data_from_q_error_info = kw_data_from_q
                                kw_extraction_done = True; break
                            except queue.Empty: continue
                        kw_thread.join()

                        if not kw_extraction_done:
                            try:
                                status_type_kw, kw_data_from_q = q_kw.get_nowait()
                                if status_type_kw == 'SUCCESS': extracted_keywords_from_llm = kw_data_from_q; status_kw_extraction = "SUCCESS"
                                elif status_type_kw == 'FAILURE': status_kw_extraction = "FAILURE_API"; kw_data_from_q_error_info = kw_data_from_q
                                elif status_type_kw == 'ERROR': status_kw_extraction = "ERROR_THREAD"; kw_data_from_q_error_info = kw_data_from_q
                            except queue.Empty: status_kw_extraction = "ERROR_NO_RESULT"
                        
                        if status_kw_extraction == "SUCCESS":
                            processed_keywords = clean_keywords_output(extracted_keywords_from_llm)
                            if processed_keywords: query_to_search = processed_keywords; sys.stderr.write(f"\r{ANSI_EL}Keywords processed from LLM: '{query_to_search}' ✅\n")
                            else: sys.stderr.write(f"\r{ANSI_EL}LLM keyword output empty or unsuitable after processing. Using original query. ✅\n")
                        elif status_kw_extraction == "FAILURE_API": sys.stderr.write(f"\r{ANSI_EL}LLM keyword extraction issue ({kw_data_from_q_error_info if kw_data_from_q_error_info else 'reason unknown'}), using original. ⚠️\n")
                        elif status_kw_extraction == "ERROR_THREAD": sys.stderr.write(f"\r{ANSI_EL}Keyword extraction failed ({kw_data_from_q_error_info if kw_data_from_q_error_info else 'thread error details missing'}). Using original. ❌\n")
                        elif status_kw_extraction == "ERROR_NO_RESULT": sys.stderr.write(f"\r{ANSI_EL}Keyword extraction failed (thread did not return result). Using original. ❌\n")
                        else: sys.stderr.write(f"\r{ANSI_EL}Keyword extraction status unclear or took too long, using original query. ⚠️\n")
                        sys.stderr.flush()
                    
                    sys.stderr.write(f"Searching online for: '{query_to_search}'... "); sys.stderr.flush()
                    search_results_urls, search_error_msg = None, None

                    def perform_search_in_thread(q_search_results, search_query_param):
                        nonlocal search_error_msg
                        try:
                            results = list(google_search_func(
                                search_query_param, num_results=DEFAULT_SEARCH_RESULTS_LIMIT,
                                lang='en', sleep_interval=DEFAULT_SEARCH_SLEEP_INTERVAL ))
                            q_search_results.put(results)
                        except Exception as e: search_error_msg = str(e); q_search_results.put(None)

                    search_q = queue.Queue()
                    search_thread = threading.Thread(target=perform_search_in_thread, args=(search_q, query_to_search), daemon=True)
                    search_thread.start()

                    while search_thread.is_alive():
                        display_query_for_spinner = query_to_search if len(query_to_search) < 30 else query_to_search[:27] + "..."
                        sys.stderr.write(f"\r{ANSI_EL}Searching online for: '{display_query_for_spinner}'... {next(spinner_chars_local)}")
                        sys.stderr.flush(); time.sleep(0.1)
                    search_thread.join()

                    try: search_results_urls = search_q.get_nowait()
                    except queue.Empty:
                        if not search_error_msg: search_error_msg = "Search thread did not return a result."
                    
                    display_query_for_status = query_to_search if len(query_to_search) < 40 else query_to_search[:37] + "..."
                    if search_error_msg: sys.stderr.write(f"\r{ANSI_EL}Searching for '{display_query_for_status}'... ❌ Error: {search_error_msg}\n")
                    elif search_results_urls is None: sys.stderr.write(f"\r{ANSI_EL}Searching for '{display_query_for_status}'... ❌ Unspecified error.\n")
                    elif not search_results_urls: sys.stderr.write(f"\r{ANSI_EL}Searching for '{display_query_for_status}'... No results.\n")
                    else: sys.stderr.write(f"\r{ANSI_EL}Searching for '{display_query_for_status}'... ✅ Found {len(search_results_urls)} results.\n")
                    sys.stderr.flush()

                    if search_results_urls:
                        num_total_results = len(search_results_urls)
                        processed_lines.append(f"--- Search results for query: \"{query_to_search}\" (from original: \"{original_query}\", top {num_total_results}) ---")
                        base_fetch_msg_template = f"Fetching content from {num_total_results} result(s)"
                        sys.stderr.write(f"{base_fetch_msg_template} [0/{num_total_results}]..."); sys.stderr.flush()
                        successful_fetches = 0

                        for i, url_from_search in enumerate(search_results_urls):
                            try:
                                parsed_url = urlparse(url_from_search)
                                netloc_display = parsed_url.netloc
                                if len(netloc_display) > 30: netloc_display = netloc_display[:27] + "..."
                                display_url_segment = f"{parsed_url.scheme}://{netloc_display}"
                            except Exception: display_url_segment = url_from_search[:30]+"..." if len(url_from_search)>30 else url_from_search

                            progress_text = f"[{i+1}/{num_total_results}] ({display_url_segment})"
                            sys.stderr.write(f"\r{ANSI_EL}{base_fetch_msg_template} {progress_text}... {next(spinner_chars_local)}")
                            sys.stderr.flush()

                            q_fetch_search = queue.Queue()
                            fetch_thread_search = threading.Thread(target=lambda func, u, q_res: q_res.put(func(u)),
                                                                   args=(fetch_webpage, url_from_search, q_fetch_search), daemon=True)
                            fetch_thread_search.start()
                            webpage_content_from_search, content_fetched_search = None, False
                            while fetch_thread_search.is_alive():
                                sys.stderr.write(f"\r{ANSI_EL}{base_fetch_msg_template} {progress_text}... {next(spinner_chars_local)}")
                                sys.stderr.flush(); time.sleep(0.1)
                                try: webpage_content_from_search = q_fetch_search.get_nowait(); content_fetched_search = True; break
                                except queue.Empty: continue
                            fetch_thread_search.join()

                            if not content_fetched_search:
                               try: webpage_content_from_search = q_fetch_search.get_nowait()
                               except queue.Empty: webpage_content_from_search = None

                            if webpage_content_from_search:
                                successful_fetches += 1
                                processed_lines.append(f"  --- Content from search result: {url_from_search} ---\n{webpage_content_from_search}\n  --- End of content for {url_from_search} ---")
                            else:
                                processed_lines.append(f"  --- Could not fetch content from search result: {url_from_search} ---")
                        
                        final_status_symbol = "✅" if successful_fetches == num_total_results else ("⚠️" if successful_fetches > 0 else "❌")
                        final_summary_msg = f"\r{ANSI_EL}{base_fetch_msg_template}... {final_status_symbol} Done. ({successful_fetches}/{num_total_results} succeeded)\n"
                        sys.stderr.write(final_summary_msg); sys.stderr.flush()
                        processed_lines.append(f"--- End of search results for \"{query_to_search}\" (from original: \"{original_query}\") ---")
                
                except KeyboardInterrupt:
                    display_cancelled_query = original_query[:50] + "..." if len(original_query) > 50 else original_query
                    sys.stderr.write(f"\r{ANSI_EL}\n^C @search for \"{display_cancelled_query}\" cancelled.\n{ANSI_RESET}")
                    sys.stderr.flush()
                    processed_lines.append(f"--- @search for \"{original_query}\" was cancelled by user ---")
                    # continue processing other lines if this was part of a larger input
            else:
                 processed_lines.append(line) # Malformed @search directive
        else: # Not a directive
            processed_lines.append(line)
    return '\n'.join(processed_lines)


# --- API Worker Thread (remains unchanged) ---
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
            "model": model, "messages": messages,
            "max_tokens": max_tokens, "stream": True
        }
        if frequency_penalty is not None: params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None: params["presence_penalty"] = presence_penalty
        stream = client.chat.completions.create(**params)
        for chunk in stream: q.put(('CHUNK', chunk))
        q.put(('DONE', None))
    except APIError as e: q.put(('ERROR', f"API Error: {str(e)}"))
    except Exception as e: q.put(('ERROR', f"An unexpected error occurred during streaming: {str(e)}"))

# --- stream_response Function (remains largely unchanged, terminal output aspects are fine) ---
def stream_response(client: OpenAI, model: str, model_type: str, messages: List[Dict[str, str]],
                    max_tokens: int, frequency_penalty: Optional[float],
                    presence_penalty: Optional[float], hide_reasoning: bool) -> Optional[str]:

    def detect_language(text: str) -> str:
        if re.search(r'[\u4e00-\u9fff]', text): return 'zh'
        return 'en'

    LANGUAGE_PROMPTS = {'zh': "请用中文回答，且禁止加入拼音。", 'en': "Please respond in English."}

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
            sys.stdout.flush(); thinking_indicator_on_screen = False

    def write_thinking_chunk_with_indicator(text_chunk: str):
        nonlocal thinking_indicator_on_screen
        clear_thinking_indicator_if_on_screen()
        sys.stdout.write(text_chunk + thinking_indicator_char); sys.stdout.flush()
        thinking_indicator_on_screen = True

    def start_spinner():
        nonlocal spinner_active, first_content_received_gemini
        if model_type == 'gemini' and first_content_received_gemini: return
        clear_thinking_indicator_if_on_screen()
        if not spinner_active: sys.stderr.write("Thinking... "); spinner_active = True; sys.stderr.flush()

    def update_spinner():
        nonlocal first_content_received_gemini
        if model_type == 'gemini' and first_content_received_gemini: return
        if spinner_active: sys.stderr.write(f"\r{ANSI_EL}Thinking... {next(spinner_chars)}"); sys.stderr.flush()

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
        if hide_reasoning and model_type == 'gemini': start_spinner()
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
                                sys.stdout.write(ANSI_RESET + "\n" + REASONING_END_MARKER + "\n"); sys.stdout.flush()
                                strip_leading_newline_next_write = True
                        if model_type != 'gemini' and spinner_active and not in_think_block_tag: stop_spinner()
                        full_response_chunks.append(content); buffer += content
                        processed_chunk_for_history_this_delta, output_to_print_for_normal_content = '', ""
                        current_processing_buffer, buffer = buffer, ''
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
                                        sys.stdout.write(REASONING_START_MARKER + ANSI_GREEN); sys.stdout.flush()
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
                                        if think_content_chunk: write_thinking_chunk_with_indicator(think_content_chunk)
                                        clear_thinking_indicator_if_on_screen()
                                        sys.stdout.write(ANSI_RESET + REASONING_END_MARKER + '\n'); sys.stdout.flush()
                                        strip_leading_newline_next_write = True
                                    elif hide_reasoning and spinner_active: stop_spinner()   
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
                    if spinner_active: stop_spinner(success=final_status_success) 
                    break
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
                        if item_type == 'DONE':
                            if spinner_active: stop_spinner(success=final_status_success) 
                            break
                        if item_type == 'ERROR':
                            final_status_success = False; stop_spinner(success=False)
                            if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
                            sys.stdout.flush(); sys.stderr.write(f"\n{data}\n"); return None
                    except queue.Empty:
                        if final_status_success: final_status_success = False; stop_spinner(success=False)
                        sys.stderr.write(f"\n{ANSI_EL}Worker thread finished unexpectedly.\n"); return None 
                    break 
        clear_thinking_indicator_if_on_screen()
        if spinner_active: stop_spinner(success=final_status_success) 
        if not hide_reasoning: 
            if is_reasoning_deepseek: sys.stdout.write(ANSI_RESET + "\n" + REASONING_END_MARKER + "\n")
            elif in_think_block_tag: sys.stdout.write(ANSI_RESET + "\n" + REASONING_END_MARKER + "\n")
        sys.stdout.flush() 
        processed_response = "".join(processed_response_chunks)
        return processed_response.lstrip('\n') if processed_response else ""
    except KeyboardInterrupt:
        final_status_success = False 
        sys.stdout.write(ANSI_RESET); sys.stderr.write(ANSI_RESET) 
        sys.stdout.flush(); sys.stderr.flush()
        clear_thinking_indicator_if_on_screen(); stop_spinner(success=False)
        if not hide_reasoning and (is_reasoning_deepseek or in_think_block_tag):
             sys.stdout.write(ANSI_RESET + "\n" + REASONING_END_MARKER + "\n"); sys.stdout.flush()
        sys.stderr.write(f"{ANSI_GREEN}\n^C Cancelled Stream!\n{ANSI_RESET}"); sys.stderr.flush()
        if TERMIOS_AVAILABLE:
            try:
                if sys.stdin.isatty() and sys.stdin.fileno() >= 0:
                    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
            except Exception as e_flush:
                sys.stderr.write(f"Warn: Flushing stdin (TCIFLUSH) failed: {e_flush}\n"); sys.stderr.flush()
        return None
    except Exception as e:
        final_status_success = False; clear_thinking_indicator_if_on_screen(); stop_spinner(success=False)
        if not hide_reasoning and (is_reasoning_deepseek or in_think_block_tag): 
            sys.stdout.write(ANSI_RESET + "\n" + REASONING_END_MARKER + "\n"); sys.stdout.flush() 
        sys.stderr.write(f"\n{ANSI_EL}An unexpected error occurred in stream_response: {str(e)}\n")
        traceback.print_exc(file=sys.stderr); return None
    finally:
        clear_thinking_indicator_if_on_screen()
        if 'spinner_active' in locals() and spinner_active: sys.stderr.write(f"\r{ANSI_EL}"); sys.stderr.flush() 
        if 'worker_thread' in locals() and worker_thread.is_alive(): worker_thread.join(timeout=1.0) 


# --- Terminal Control Helper Functions (set_terminal_no_echoctl, restore_terminal_settings) ---
# These might be less necessary or could conflict with prompt_toolkit's own terminal handling.
# For now, we'll comment out their usage and rely on prompt_toolkit.
def set_terminal_no_echoctl(fd: int) -> Optional[List]:
    if not TERMIOS_AVAILABLE or not os.isatty(fd): return None
    try:
        old_settings = termios.tcgetattr(fd)
        new_settings = list(old_settings)
        new_settings[3] &= ~termios.ECHOCTL # type: ignore[attr-defined] # termios.ECHOCTL is platform-dependent
        termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
        return old_settings
    except termios.error: return None

def restore_terminal_settings(fd: int, old_settings: Optional[List]):
    if not TERMIOS_AVAILABLE or not os.isatty(fd) or old_settings is None: return
    try: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except termios.error: pass


# --- prompt_toolkit Completer ---
COMPLETION_TOKENS_PT = ["@file(", "@url(", "@search(", "!clear"]

if PROMPT_TOOLKIT_AVAILABLE:
    class AgCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            word_before_cursor = document.get_word_before_cursor(WORD=True)

            # Handle @file() completion
            if text.lstrip().startswith("@file("): # Use lstrip to allow leading spaces
                # Find the start of the @file( directive
                directive_match = re.match(r'(.*@file\()', text)
                if not directive_match: # Should not happen if startswith passed
                    return

                prefix_text = directive_match.group(1) # e.g. "  @file("
                path_typed_in_directive = text[len(prefix_text):]
                
                # Allow closing parenthesis to be typed without breaking completion
                if path_typed_in_directive.endswith(")"):
                    path_typed_in_directive = path_typed_in_directive[:-1]


                expanded_partial_path = os.path.expanduser(os.path.expandvars(path_typed_in_directive))
                
                search_dir = os.path.dirname(expanded_partial_path)
                item_prefix_for_glob = os.path.basename(expanded_partial_path)

                if not search_dir: # Path is relative to current dir or just a prefix
                    search_dir = "." 
                
                glob_pattern = os.path.join(search_dir, item_prefix_for_glob + "*")

                try:
                    for path_match_from_glob in glob.glob(glob_pattern):
                        # path_match_from_glob is what glob found (e.g., /home/user/doc.txt or relative/path)
                        # We need to offer this path to be inserted.
                        
                        # Make it a displayable path, append / if dir
                        display_path = path_match_from_glob
                        if os.path.isdir(path_match_from_glob) and not display_path.endswith(os.sep):
                            display_path += os.sep
                        
                        # The text to insert to complete the path typed so far.
                        # The start_position is negative length of the path_typed_in_directive part.
                        yield Completion(
                            text=display_path,  # This is the text that will replace/complete `path_typed_in_directive`
                            start_position=-len(path_typed_in_directive),
                            display=display_path, # How it shows in the completion menu
                            display_meta="path"
                        )
                except Exception:
                    pass # Ignore globbing errors
                return # Processed @file, no further completion

            # Handle other simple completions for the whole line (e.g. !clear, @url() )
            # This replaces the entire input line.
            # More advanced would be to complete based on word_before_cursor if not at start of line.
            # For simplicity, this version completes from the start of the line or word.
            # If `text` is what the user typed for the *current token being completed*
            token_to_complete = word_before_cursor if word_before_cursor else text

            for token_prefix in COMPLETION_TOKENS_PT:
                if token_prefix.startswith(token_to_complete):
                    yield Completion(
                        text=token_prefix,
                        start_position=-len(token_to_complete),
                        display=token_prefix,
                        display_meta="directive/command"
                    )
else: # Fallback if prompt_toolkit is not available (though script will likely fail in interactive)
    class AgCompleter(Completer): # type: ignore[no-redef]
        def get_completions(self, document, complete_event):
            yield from []


# --- setup_tty_stdin and restore_stdin ---
def setup_tty_stdin() -> Tuple[int, Optional[str]]:
    original_stdin_fd = -1
    pipe_input_content = ""
    
    # Check if readline was imported - no longer primary, so remove its specific setup
    # readline_available_flag = ('readline' in sys.modules and ...) 
    
    try:
        original_stdin_fd = os.dup(0) # Save current stdin
        if not sys.stdin.isatty(): # If current stdin is not a TTY (e.g., piped input)
            pipe_input_content = sys.stdin.read() # Read all piped content
            sys.stdin.close() # Close the piped stdin

            # Try to open /dev/tty as the new stdin
            try:
                new_stdin_fd = os.open('/dev/tty', os.O_RDONLY)
                os.dup2(new_stdin_fd, 0) # Set new_stdin_fd to be file descriptor 0 (stdin)
                os.close(new_stdin_fd) # Close original fd for /dev/tty, it's now stdin
                # Re-open sys.stdin as a Python file object based on the new fd 0
                sys.stdin = open(0, 'r', closefd=False) 
            except OSError:
                # If /dev/tty cannot be opened, restore original stdin (which was the pipe)
                # This means interactive mode might not work as expected if pipe was the only input.
                sys.stderr.write("Warning: Could not open /dev/tty. Interactive input might not work after pipe.\n")
                try:
                    os.dup2(original_stdin_fd, 0)
                    sys.stdin = open(0, 'r', closefd=False)
                except OSError as restore_e:
                    sys.stderr.write(f"CRITICAL_SETUP: Failed to restore original stdin: {restore_e}\n")
        
        # prompt_toolkit setup (history) is now handled in main() when PromptSession is created.
        # Removed readline specific delims, completer, history file loading from here.

    except Exception as e:
        sys.stderr.write(f"Error in setup_tty_stdin: {e}\n")
        traceback.print_exc(file=sys.stderr)
        if original_stdin_fd != -1 and original_stdin_fd != 0: # If we dup'd a non-stdin fd
            try: os.close(original_stdin_fd) # Clean it up
            except OSError: pass
            original_stdin_fd = -1 # Mark as closed
            
    return original_stdin_fd, pipe_input_content

def restore_stdin(original_stdin_fd: int):
    try:
        if original_stdin_fd != -1: # If we have a saved original stdin fd
            try: # Close current sys.stdin Python object if it exists and is open
                if sys.stdin and not sys.stdin.closed: sys.stdin.close()
            except Exception: pass
            
            try: # Restore original_stdin_fd to be fd 0
                os.dup2(original_stdin_fd, 0)
                # Re-open sys.stdin Python file object for the restored fd 0
                sys.stdin = open(0, 'r', closefd=False) 
            except Exception as e: 
                sys.stderr.write(f"Warning: Failed to restore stdin to original fd {original_stdin_fd}: {e}\n")
            finally: # Clean up the original_stdin_fd if it wasn't fd 0 itself
                if original_stdin_fd != 0: # Don't close if it *was* the original stdin fd (0)
                    try: os.close(original_stdin_fd)
                    except OSError: pass
    except Exception as e:
        sys.stderr.write(f"Error in restore_stdin: {e}\n")


# --- Modified main Function ---
def main():
    parser = argparse.ArgumentParser(description='AG - Ask GPT from CLI.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ... (parser arguments remain the same) ...
    parser.add_argument('--api-url', default=DEFAULT_API_URL, help="Base API URL for the main LLM.")
    parser.add_argument('--model', default=DEFAULT_MODEL, help="Model name for the main LLM.")
    parser.add_argument('--model-type', choices=['openai', 'gemini'], default=DEFAULT_MODEL_TYPE, help="Type of the main LLM.")
    parser.add_argument('--api-key', default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY), help="API key for the main LLM.")
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for the main LLM response.")
    parser.add_argument('--frequency-penalty', type=float, default=DEFAULT_FREQUENCY_PENALTY, help="Frequency penalty for the main LLM. Set to -1.0 to disable.")
    parser.add_argument('--presence-penalty', type=float, default=DEFAULT_PRESENCE_PENALTY, help="Presence penalty for the main LLM. Set to -1.0 to disable.")
    
    parser.add_argument('--search-kw-api-url', default=None, help="API URL for search keyword extraction LLM. Defaults to main API URL if not set.")
    parser.add_argument('--search-kw-model', default=None, help="Model name for search keyword extraction LLM. Defaults to main model if not set.")
    parser.add_argument('--search-kw-api-key', default=None, help="API key for search keyword extraction LLM. Defaults to main API key if not set.")
    parser.add_argument('--search-kw-max-tokens', type=int, default=DEFAULT_SEARCH_KW_MAX_TOKENS, help="Max tokens for search keyword extraction LLM.")
    parser.add_argument('--search-kw-temperature', type=float, default=DEFAULT_SEARCH_KW_TEMPERATURE, help="Temperature for search keyword extraction LLM (0.0-2.0).")

    parser.add_argument('-g', '--grep-enabled', action='store_true', help="Enable strict grep-like filtering mode (requires piped input and query text).")
    parser.add_argument('-r', '--hide-reasoning', action='store_true', help="Hide <think> and reasoning content, showing only a spinner.")
    parser.add_argument('raw_text', nargs='?', help="Raw text input for a single query. If provided, script exits after processing.")
    args = parser.parse_args()

    if not PROMPT_TOOLKIT_AVAILABLE and not args.raw_text :
        sys.exit("Error: prompt_toolkit is required for interactive mode but not installed. Please run: pip install prompt_toolkit")


    if args.model_type == 'gemini': args.hide_reasoning = True
    freq_penalty = args.frequency_penalty if args.frequency_penalty != -1.0 else None
    pres_penalty = args.presence_penalty if args.presence_penalty != -1.0 else None

    client = None
    try: client = OpenAI( base_url=args.api_url, api_key=args.api_key )
    except Exception as e: sys.stderr.write(f"Warning: Error initializing main OpenAI client: {e}\n")

    messages: List[Dict[str, str]] = []
    original_stdin_fd, initial_pipe_input_content = setup_tty_stdin()
    
    # original_termios_settings = None # Commented out, relying on prompt_toolkit
    # if TERMIOS_AVAILABLE and sys.stdin.isatty():
    #     try:
    #         fd_val = sys.stdin.fileno()
    #         if fd_val >=0: original_termios_settings = set_terminal_no_echoctl(fd_val)
    #     except Exception: pass

    grep_prompt_template = ("I want you to act strictly as a Linux `grep` command filter.\n"
                            "Input will consist of text, potentially multi-line output from another command.\n"
                            "The pattern can be in natural languages that you need to understand it.\n"
                            "Your task is to filter this input based on a user-provided pattern.\n"
                            "***You MUST ONLY return the lines from the input that match the pattern.***\n"
                            "Do NOT include any explanations, introductions, summaries, or conversational text.\n"
                            "Do NOT add line numbers unless they were present in the original input lines.\n"
                            "If no lines match, return absolutely ***nothing*** (empty output).\n"
                            "Pattern: \"{pattern}\"\nInput Text:\n{input_text}\nMatching Lines Only:")
    has_used_initial_pipe_input = False

    try:
        if args.raw_text: # Single query mode
            user_content = args.raw_text
            if initial_pipe_input_content:
                user_content = f"Context from piped input:\n{initial_pipe_input_content.strip()}\n\nUser Query: {user_content}"
            
            processed_user_content = process_input_directives(
                user_content, client, args.model,
                args.search_kw_api_url, args.search_kw_model, args.search_kw_api_key,
                args.search_kw_max_tokens, args.search_kw_temperature )

            if args.grep_enabled:
                if not initial_pipe_input_content: sys.exit("Error: --grep-enabled requires piped input with raw_text.")
                if not client: sys.exit("Error: Main LLM client not initialized, required for --grep-enabled.")
                messages = [{"role": "user", "content": grep_prompt_template.format(pattern=args.raw_text, input_text=initial_pipe_input_content)}]
            else:
                messages = [{"role": "user", "content": processed_user_content}]
            
            if not client and not args.grep_enabled and not ("@search" in user_content or "@file" in user_content or "@url" in user_content):
                 sys.exit("Error: Main LLM client not initialized and no directives found in raw_text input.")
            elif not client and args.grep_enabled: 
                 sys.exit("Error: Main LLM client not initialized, required for --grep-enabled.")

            if client: 
                response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
                if response_content is not None and not response_content.endswith('\n'): sys.stdout.write("\n")
            elif not ("@search" in user_content or "@file" in user_content or "@url" in user_content): # Only directives processed
                sys.stderr.write("No main LLM client and no LLM-dependent directives to process. Directives like @file were processed.\n")
            
            sys.stdout.flush(); sys.exit(0)

        # Interactive mode
        if not client:
             sys.exit(f"Error: Main LLM client failed to initialize. Interactive mode requires a functioning main LLM. API URL: {args.api_url}")
        if not PROMPT_TOOLKIT_AVAILABLE: # Should have exited earlier, but as a safeguard
             sys.exit("Error: prompt_toolkit is required for interactive mode but not installed.")


        history = FileHistory(os.path.join(os.path.expanduser("~"), ".ag_pt_history"))
        completer = AgCompleter()
        session = PromptSession(history=history, completer=completer, complete_while_typing=True)
        
        # For readline's get_line_buffer check on KeyboardInterrupt
        # prompt_toolkit's session.default_buffer.text can be used if needed.
        # is_readline_available_in_main = False # Not using readline here

        sys.stderr.write("Entering interactive mode. Use Ctrl+D to submit, !cmd, @file, @url, @search, Tab for completion:\n")
        while True:
            sys.stderr.write(f"\n{ANSI_RESET}💥 Ask (Ctrl+D Submit, !cmd, @file, @url, @search, Tab completion):\n")
            current_prompt_accumulator: List[str] = []
            if initial_pipe_input_content and not has_used_initial_pipe_input:
                current_prompt_accumulator.append(f"Context from initial piped input:\n{initial_pipe_input_content.strip()}")
            
            # Multi-line input loop
            while True:
                is_continuation = bool(''.join(current_prompt_accumulator).strip())
                prompt_char_str = ANSI_MAGENTA + "... " + ANSI_RESET if is_continuation else ANSI_MAGENTA + ">>> " + ANSI_RESET
                
                try:
                    # Use ANSI() wrapper if prompt_char_str contains ANSI escapes
                    user_line_input = session.prompt(ANSI(prompt_char_str)) # prompt_toolkit understands ANSI
                    stripped_user_line = user_line_input.strip()

                    if stripped_user_line.startswith('!'):
                        command_str = stripped_user_line[1:].strip()
                        if command_str.lower() == 'clear':
                            messages.clear(); current_prompt_accumulator.clear()
                            sys.stdout.write(ANSI_CLEAR_SCREEN); sys.stdout.flush()
                            sys.stderr.write("Conversation history and current input cleared.\n"); break # Break inner loop to restart prompt
                        elif command_str:
                            sys.stderr.write(f"\nExecuting: `{command_str}`\n---\n")
                            cmd_output = execute_command(command_str)
                            sys.stderr.write(cmd_output)
                            if not cmd_output.endswith('\n'): sys.stderr.write("\n")
                            sys.stderr.write("---\n"); sys.stderr.flush()
                            current_prompt_accumulator.append(f"--- User executed command `{command_str}` output ---\n{cmd_output.strip()}\n--- End command output ---")
                            sys.stderr.write("Command output added to prompt. Continue or Ctrl+D.\n")
                        else: # '!' or '! '
                            current_prompt_accumulator.append(user_line_input)
                    else:
                        current_prompt_accumulator.append(user_line_input)
                except EOFError: # Ctrl+D
                    sys.stderr.write(ANSI_GREEN + "^D EOF!\n" + ANSI_RESET); break # Break inner loop to process accumulated input
                except KeyboardInterrupt: # Ctrl+C
                    sys.stderr.write(ANSI_GREEN + "\n^C Cancelled!\n" + ANSI_RESET)
                    # Check if there's anything in the current line buffer of prompt_toolkit or accumulator
                    current_line_in_pt_buffer = ""
                    if hasattr(session, 'default_buffer') and hasattr(session.default_buffer, 'text'):
                        current_line_in_pt_buffer = session.default_buffer.text

                    if bool(''.join(current_prompt_accumulator).strip()) or bool(current_line_in_pt_buffer.strip()):
                        current_prompt_accumulator.clear() 
                        # prompt_toolkit usually clears its own buffer on Ctrl+C,
                        # or one could do session.default_buffer.reset() if needed.
                        # Forcing a break here to get a fresh prompt line might be good.
                        break # Break inner loop, will show main "Ask" prompt again
                    else: # Truly empty, user wants to exit
                        sys.stderr.write(ANSI_RESET); sys.stderr.flush(); raise # Re-raise to exit main loop
            
            final_prompt_str = '\n'.join(current_prompt_accumulator).strip()
            if not final_prompt_str: continue # If Ctrl+D was pressed on an empty accumulator

            if initial_pipe_input_content and not has_used_initial_pipe_input and \
               any(initial_pipe_input_content.strip() in p for p in current_prompt_accumulator):
                has_used_initial_pipe_input = True # Mark as used
            
            processed_input = process_input_directives(
                final_prompt_str, client, args.model,
                args.search_kw_api_url, args.search_kw_model, args.search_kw_api_key,
                args.search_kw_max_tokens, args.search_kw_temperature )
            
            messages.append({"role": "user", "content": processed_input})
            sys.stdout.write(f"{ANSI_RESET}💡:\n"); sys.stdout.flush()
            
            response_content = stream_response(client, args.model, args.model_type, messages, 
                                               args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            
            if response_content is not None:
                messages.append({"role": "assistant", "content": response_content})
                if response_content and not response_content.endswith('\n'): sys.stdout.write("\n")
                sys.stdout.flush()
            sys.stdout.write(ANSI_RESET); sys.stdout.flush()

    except KeyboardInterrupt: # Ctrl+C at the main "Ask" prompt or during processing (not stream)
        sys.stderr.write(ANSI_RESET + "\nExiting due to KeyboardInterrupt.\n")
    except Exception as e:
        sys.stderr.write(f"\n{ANSI_RESET}Unexpected error in main loop: {e}\n"); traceback.print_exc(file=sys.stderr)
    finally:
        # Restore termios settings if they were changed (commented out for now)
        # if TERMIOS_AVAILABLE and original_termios_settings is not None:
        #     current_stdin_is_tty = False; current_stdin_fd_val = -1
        #     try:
        #         if sys.stdin and not sys.stdin.closed:
        #             current_stdin_fd_val = sys.stdin.fileno()
        #             if current_stdin_fd_val >=0 and os.isatty(current_stdin_fd_val): current_stdin_is_tty = True
        #     except Exception: pass
        #     if current_stdin_is_tty: restore_terminal_settings(current_stdin_fd_val, original_termios_settings)
        
        restore_stdin(original_stdin_fd) # Crucial to restore original stdin
        sys.stderr.write(ANSI_RESET); sys.stdout.write(ANSI_RESET)
        sys.stderr.flush(); sys.stdout.flush()

if __name__ == '__main__':
    main()

