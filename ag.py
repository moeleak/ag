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
import threading # Keep this one
import queue
import traceback
import wcwidth
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse
import glob # Added for file path completion

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

# Default parameters for keyword extraction LLM (can be overridden by CLI args)
DEFAULT_SEARCH_KW_MAX_TOKENS = 60
DEFAULT_SEARCH_KW_TEMPERATURE = 0.2


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
                    continue

                if not content_fetched_url:
                    try:
                        webpage_content_result = q_url.get_nowait()
                    except queue.Empty:
                        webpage_content_result = None

                if webpage_content_result:
                    sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... ✅ Done.\n")
                    processed_lines.append(f"--- Content from {url} ---\n{webpage_content_result}\n--- End of {url} ---")
                else:
                    if not any(f"cancelled by user" in s for s in processed_lines if url in s):
                        sys.stderr.write(f"\r{ANSI_EL}Fetching content from URL: {display_url_at_url} ... ❌ Failed.\n")
                sys.stderr.flush()
            else:
                 processed_lines.append(line) # Malformed @url directive
        elif stripped_line.startswith('@search'):
            if google_search_func is None:
                sys.stderr.write(f"{ANSI_EL}Error: The 'googlesearch-python' library is not installed. @search functionality is unavailable.\n"
                                 f"{ANSI_EL}Please install it using: pip install googlesearch-python\n")
                processed_lines.append(line)
                continue

            match = re.match(r'@search\(([^)]+)\)', stripped_line)
            if match:
                original_query = ""
                try:
                    original_query = match.group(1).strip()
                    query_to_search = original_query

                    # Determine LLM client and model for keyword extraction
                    llm_for_keywords_enabled = False
                    final_kw_client = None
                    final_kw_model = None

                    if search_kw_model_name_param:  # Dedicated keyword model specified by user
                        chosen_kw_api_url = search_kw_api_url_param
                        if not chosen_kw_api_url: # Fallback to main LLM's URL or default
                            chosen_kw_api_url = str(main_llm_client.base_url) if main_llm_client and main_llm_client.base_url else DEFAULT_API_URL
                        
                        chosen_kw_api_key = search_kw_api_key_param
                        if chosen_kw_api_key is None and main_llm_client : # Fallback to main LLM's key if specific KW key not given
                            chosen_kw_api_key = main_llm_client.api_key
                        # If chosen_kw_api_key is still None, it's passed as None to OpenAI client.

                        # Ensure URL is string
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
                            # Fallback is implicitly handled as llm_for_keywords_enabled remains False
                    elif main_llm_client and main_llm_model:  # Fallback to main LLM if available and no dedicated one
                        final_kw_client = main_llm_client
                        final_kw_model = main_llm_model
                        llm_for_keywords_enabled = True

                    if llm_for_keywords_enabled and final_kw_client and final_kw_model:
                        q_kw = queue.Queue()
                        
                        def extract_keywords_task_threaded(q_result, client_param, model_param, messages_param, max_tokens_param_kw, temperature_param_kw):
                            try:
                                completion_kw = client_param.chat.completions.create(
                                    model=model_param,
                                    messages=messages_param,
                                    max_tokens=max_tokens_param_kw,
                                    temperature=temperature_param_kw # Use new temperature parameter
                                )
                                if completion_kw.choices and completion_kw.choices[0].message and completion_kw.choices[0].message.content:
                                    res = completion_kw.choices[0].message.content.strip()
                                    q_result.put(('SUCCESS', res if res else None))
                                else:
                                    q_result.put(('FAILURE', "No content in LLM response for keywords."))
                            except APIError as e_api_kw:
                                q_result.put(('ERROR', f"API Error during keyword extraction: {str(e_api_kw)}"))
                            except Exception as e_task_kw:
                                q_result.put(('ERROR', f"Unexpected error during keyword extraction: {str(e_task_kw)}"))

                        keyword_extraction_prompt_system = (
                            "You are an AI assistant specialized in optimizing search queries. From the user's input, "
                            "extract the core keywords that would be most effective for a web search.\n"
                            "- Return ONLY the keywords.\n"
                            "- Keywords should be separated by a single space.\n"
                            "- Do not include any explanations, introductions, or conversational text.\n"
                            "- Focus on proper nouns, technical terms, and essential concepts from the query.\n"
                            "- If the query is already very short and keyword-like, you can return it as is."
                        )
                        kw_messages = [
                            {"role": "system", "content": keyword_extraction_prompt_system},
                            {"role": "user", "content": f"Query: \"{original_query}\""}
                        ]
                        
                        kw_thread = threading.Thread(target=extract_keywords_task_threaded, 
                                                     args=(q_kw, final_kw_client, final_kw_model, kw_messages, 
                                                           search_kw_max_tokens_param, search_kw_temperature_param), daemon=True)
                        
                        sys.stderr.write(f"Extracting keywords using {final_kw_model}... ")
                        sys.stderr.flush()
                        kw_thread.start()

                        kw_extraction_done = False
                        extracted_keywords_from_llm = None
                        status_kw_extraction = "PENDING"

                        while kw_thread.is_alive():
                            sys.stderr.write(f"\r{ANSI_EL}Extracting keywords using {final_kw_model}... {next(spinner_chars_local)}")
                            sys.stderr.flush()
                            time.sleep(0.1)
                            try:
                                status_type_kw, kw_data_from_q = q_kw.get_nowait()
                                if status_type_kw == 'SUCCESS': extracted_keywords_from_llm = kw_data_from_q; status_kw_extraction = "SUCCESS"
                                elif status_type_kw == 'FAILURE': status_kw_extraction = "FAILURE_API" 
                                elif status_type_kw == 'ERROR': status_kw_extraction = "ERROR_THREAD"
                                kw_extraction_done = True; break
                            except queue.Empty: continue
                        
                        kw_thread.join()

                        if not kw_extraction_done:
                            try:
                                status_type_kw, kw_data_from_q = q_kw.get_nowait()
                                if status_type_kw == 'SUCCESS': extracted_keywords_from_llm = kw_data_from_q; status_kw_extraction = "SUCCESS"
                                elif status_type_kw == 'FAILURE': status_kw_extraction = "FAILURE_API"
                                elif status_type_kw == 'ERROR': status_kw_extraction = "ERROR_THREAD"
                            except queue.Empty: status_kw_extraction = "ERROR_NO_RESULT"
                        
                        if status_kw_extraction == "SUCCESS" and extracted_keywords_from_llm:
                            query_to_search = extracted_keywords_from_llm
                            sys.stderr.write(f"\r{ANSI_EL}Keywords extracted: '{query_to_search}' ✅\n")
                        elif status_kw_extraction == "SUCCESS" and not extracted_keywords_from_llm:
                            sys.stderr.write(f"\r{ANSI_EL}Keyword extraction returned empty, using original query. ✅\n")
                        elif status_kw_extraction == "FAILURE_API":
                             sys.stderr.write(f"\r{ANSI_EL}LLM keyword extraction issue ({kw_data_from_q if 'kw_data_from_q' in locals() else 'reason unknown'}), using original. ⚠️\n")
                        elif status_kw_extraction in ["ERROR_THREAD", "ERROR_NO_RESULT"]:
                            err_msg_detail = kw_data_from_q if 'kw_data_from_q' in locals() and status_kw_extraction == "ERROR_THREAD" else "thread comm error"
                            sys.stderr.write(f"\r{ANSI_EL}Keyword extraction failed ({err_msg_detail}). Using original. ❌\n")
                        else: # PENDING or other unknown
                            sys.stderr.write(f"\r{ANSI_EL}Keyword extraction status unclear or took too long, using original query. ⚠️\n")
                        sys.stderr.flush()
                    else: # LLM for keywords not enabled or not configured
                        # sys.stderr.write("LLM-based keyword extraction skipped.\n") # Optional: can be noisy
                        pass # query_to_search remains original_query
                    
                    sys.stderr.write(f"Searching online for: '{query_to_search}'... ")
                    sys.stderr.flush()

                    search_results_urls = None
                    search_error_msg = None

                    def perform_search_in_thread(q_search_results, search_query_param):
                        nonlocal search_error_msg
                        try:
                            results = list(google_search_func(
                                search_query_param, num_results=DEFAULT_SEARCH_RESULTS_LIMIT,
                                lang='en', sleep_interval=DEFAULT_SEARCH_SLEEP_INTERVAL ))
                            q_search_results.put(results)
                        except Exception as e:
                            search_error_msg = str(e); q_search_results.put(None)

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
                        sys.stderr.write(f"{base_fetch_msg_template} [0/{num_total_results}]...")
                        sys.stderr.flush()
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
            else:
                 processed_lines.append(line) # Malformed @search directive
        else: # Not a directive
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
                        if item_type == 'DONE': break
                        if item_type == 'ERROR':
                            final_status_success = False; stop_spinner(success=False)
                            if (is_reasoning_deepseek or in_think_block_tag) and not hide_reasoning: sys.stdout.write(ANSI_RESET)
                            sys.stdout.flush(); sys.stderr.write(f"\n{data}\n"); return None
                    except queue.Empty:
                        if final_status_success: # Only write error if not already errored out
                            final_status_success = False; stop_spinner(success=False)
                            sys.stderr.write(f"\n{ANSI_EL}Worker thread finished unexpectedly.\n")
                        return None # Return None if worker died and no error was queued by it
                    # If we broke from inner try due to DONE/ERROR after worker died, outer loop will catch DONE
                    # or the error would have been handled and returned.
                    # This break is to exit the `while True` if the worker died and we got a final item.
                    break 

        clear_thinking_indicator_if_on_screen()
        if spinner_active: stop_spinner(success=final_status_success)

        if not hide_reasoning:
            if is_reasoning_deepseek:
                sys.stdout.write(ANSI_RESET + "\n" + REASONING_END_MARKER + "\n")
            if in_think_block_tag: 
                sys.stdout.write(ANSI_RESET)
        sys.stdout.flush()

        processed_response = "".join(processed_response_chunks)
        return processed_response.lstrip('\n') if processed_response else ""

    except KeyboardInterrupt:
        final_status_success = False # Ensure spinner stops with error indication
        sys.stdout.write(ANSI_RESET); sys.stderr.write(ANSI_RESET) 
        sys.stdout.flush(); sys.stderr.flush()
        clear_thinking_indicator_if_on_screen(); stop_spinner(success=False)
        if not hide_reasoning:
            if is_reasoning_deepseek or in_think_block_tag:
                 sys.stdout.write(ANSI_RESET); sys.stdout.flush()
                 # Avoid printing end marker if it was never started or already reset.
                 # If it was truly mid-reasoning, a visual reset is enough.
                 # sys.stderr.write(f"\n{REASONING_END_MARKER if is_reasoning_deepseek else '</think>'}\n(Interrupted during reasoning/thinking)\n")
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
        # Check if spinner_active exists in local scope before trying to clear it
        if 'spinner_active' in locals() and spinner_active : 
             sys.stderr.write(f"\r{ANSI_EL}"); sys.stderr.flush()
        if 'worker_thread' in locals() and worker_thread.is_alive():
            # We should not be joining here if the main loop exited normally,
            # as the worker thread should have put 'DONE' or 'ERROR' and exited.
            # This join is more of a safeguard or for unexpected exits from stream_response.
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
_completion_matches_cache = []

def custom_directive_completer(text: str, state: int) -> Optional[str]:
    global _completion_matches_cache
    current_typing_for_logic = text 
    directive_prefix = "@file("
    if current_typing_for_logic.startswith(directive_prefix):
        if state == 0:
            _completion_matches_cache = []
            path_typed_after_directive = current_typing_for_logic[len(directive_prefix):]
            expanded_partial_path = os.path.expanduser(os.path.expandvars(path_typed_after_directive))
            search_dir = os.path.dirname(expanded_partial_path)
            item_prefix_for_glob = os.path.basename(expanded_partial_path)
            if not search_dir: search_dir = "." 
            glob_pattern = os.path.join(search_dir, item_prefix_for_glob + "*")
            try:
                for path_match_from_glob in glob.glob(glob_pattern):
                    completion_candidate = directive_prefix + path_match_from_glob
                    if os.path.isdir(path_match_from_glob) and not completion_candidate.endswith(os.sep):
                        completion_candidate += os.sep
                    _completion_matches_cache.append(completion_candidate)
            except Exception: pass
            _completion_matches_cache.sort() 
        return _completion_matches_cache[state] if state < len(_completion_matches_cache) else None
    else:
        if state == 0:
            _completion_matches_cache = [token for token in COMPLETION_TOKENS if token.startswith(current_typing_for_logic)]
        return _completion_matches_cache[state] if state < len(_completion_matches_cache) else None


# --- setup_tty_stdin and restore_stdin ---
def setup_tty_stdin() -> Tuple[int, Optional[str]]:
    original_stdin_fd = -1; pipe_input_content = ""
    readline_available_flag = ('readline' in sys.modules and all(hasattr(sys.modules['readline'], attr) for attr in 
        ['read_history_file', 'set_completer', 'parse_and_bind', 'get_line_buffer', 'get_begidx', 'get_endidx', 'get_completer_delims']))
    try:
        original_stdin_fd = os.dup(0)
        if not sys.stdin.isatty():
            pipe_input_content = sys.stdin.read(); sys.stdin.close()
            try:
                new_stdin_fd = os.open('/dev/tty', os.O_RDONLY)
                os.dup2(new_stdin_fd, 0); os.close(new_stdin_fd)
                sys.stdin = open(0, 'r', closefd=False)
            except OSError:
                try: os.dup2(original_stdin_fd, 0); sys.stdin = open(0, 'r', closefd=False)
                except OSError as restore_e: sys.stderr.write(f"CRITICAL_SETUP: Restore stdin fail: {restore_e}\n")
        if sys.stdin.isatty() and readline_available_flag:
            try:
                default_delims = readline.get_completer_delims()
                new_delims = default_delims
                for char_to_remove in ['@', '!', '(', ')', '/', '~', '$', '.']: new_delims = new_delims.replace(char_to_remove, '')
                if new_delims != default_delims: readline.set_completer_delims(new_delims)
                readline.set_completer(custom_directive_completer); readline.parse_and_bind('tab: complete')
                history_file = os.path.join(os.path.expanduser("~"), ".ag_history")
                try: readline.read_history_file(history_file)
                except (FileNotFoundError, Exception): pass
                import atexit
                def save_history():
                    try: readline.write_history_file(history_file)
                    except Exception: pass
                atexit.register(save_history)
            except Exception as e_readline: traceback.print_exc(file=sys.stderr)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        if original_stdin_fd != -1 and original_stdin_fd != 0:
            try: os.close(original_stdin_fd)
            except OSError: pass
            original_stdin_fd = -1
    return original_stdin_fd, pipe_input_content

def restore_stdin(original_stdin_fd: int):
    try:
        if original_stdin_fd != -1:
            try:
                if sys.stdin and not sys.stdin.closed: sys.stdin.close()
            except Exception: pass
            try: os.dup2(original_stdin_fd, 0); sys.stdin = open(0, 'r', closefd=False)
            except Exception: pass
            finally:
                if original_stdin_fd != 0:
                    try: os.close(original_stdin_fd)
                    except OSError: pass
    except Exception: pass

# --- Modified main Function ---
def main():
    parser = argparse.ArgumentParser(description='AG - Ask GPT from CLI.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Main LLM arguments
    parser.add_argument('--api-url', default=DEFAULT_API_URL, help="Base API URL for the main LLM.")
    parser.add_argument('--model', default=DEFAULT_MODEL, help="Model name for the main LLM.")
    parser.add_argument('--model-type', choices=['openai', 'gemini'], default=DEFAULT_MODEL_TYPE, help="Type of the main LLM.")
    parser.add_argument('--api-key', default=os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY), help="API key for the main LLM.")
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens for the main LLM response.")
    parser.add_argument('--frequency-penalty', type=float, default=DEFAULT_FREQUENCY_PENALTY, help="Frequency penalty for the main LLM. Set to -1.0 to disable.")
    parser.add_argument('--presence-penalty', type=float, default=DEFAULT_PRESENCE_PENALTY, help="Presence penalty for the main LLM. Set to -1.0 to disable.")
    
    # Search keyword LLM specific arguments
    parser.add_argument('--search-kw-api-url', default=None, help="API URL for search keyword extraction LLM. Defaults to main API URL if not set.")
    parser.add_argument('--search-kw-model', default=None, help="Model name for search keyword extraction LLM. Defaults to main model if not set.")
    parser.add_argument('--search-kw-api-key', default=None, help="API key for search keyword extraction LLM. Defaults to main API key if not set.")
    parser.add_argument('--search-kw-max-tokens', type=int, default=DEFAULT_SEARCH_KW_MAX_TOKENS, help="Max tokens for search keyword extraction LLM.")
    parser.add_argument('--search-kw-temperature', type=float, default=DEFAULT_SEARCH_KW_TEMPERATURE, help="Temperature for search keyword extraction LLM (0.0-2.0).")

    # Other arguments
    parser.add_argument('-g', '--grep-enabled', action='store_true', help="Enable strict grep-like filtering mode (requires piped input and query text).")
    parser.add_argument('-r', '--hide-reasoning', action='store_true', help="Hide <think> and reasoning content, showing only a spinner.")
    parser.add_argument('raw_text', nargs='?', help="Raw text input for a single query. If provided, script exits after processing.")
    args = parser.parse_args()

    if args.model_type == 'gemini': args.hide_reasoning = True
    freq_penalty = args.frequency_penalty if args.frequency_penalty != -1.0 else None
    pres_penalty = args.presence_penalty if args.presence_penalty != -1.0 else None

    client = None
    try:
        # Initialize main client only if not relying solely on dedicated keyword client for all LLM tasks (though usually main client is expected)
        # This script structure implies a main client is generally expected for chat.
        # Keyword client is created on-demand in process_input_directives if needed.
        client = OpenAI( base_url=args.api_url, api_key=args.api_key )
    except Exception as e:
        # If only raw_text with @search is used, and a dedicated keyword client is specified, this failure might be acceptable.
        # However, for general use, main client initialization is important.
        sys.stderr.write(f"Warning: Error initializing main OpenAI client: {e}\n")
        # Do not exit if user might be using @search with its own client.
        # If client remains None, chat functionalities won't work, but @search with dedicated client might.

    messages: List[Dict[str, str]] = []
    original_stdin_fd, initial_pipe_input_content = setup_tty_stdin()
    is_readline_available_in_main = ('readline' in sys.modules and hasattr(sys.modules['readline'], 'get_line_buffer'))
    original_termios_settings = None
    if TERMIOS_AVAILABLE and sys.stdin.isatty():
        try:
            fd_val = sys.stdin.fileno()
            if fd_val >=0: original_termios_settings = set_terminal_no_echoctl(fd_val)
        except Exception: pass

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
        if args.raw_text:
            user_content = args.raw_text
            if initial_pipe_input_content:
                user_content = f"Context from piped input:\n{initial_pipe_input_content.strip()}\n\nUser Query: {user_content}"
            
            processed_user_content = process_input_directives(
                user_content, client, args.model,
                args.search_kw_api_url, args.search_kw_model, args.search_kw_api_key,
                args.search_kw_max_tokens, args.search_kw_temperature
            )

            if args.grep_enabled:
                if not initial_pipe_input_content: sys.exit("Error: --grep-enabled requires piped input with raw_text.")
                if not client: sys.exit("Error: Main LLM client not initialized, required for --grep-enabled.")
                messages = [{"role": "user", "content": grep_prompt_template.format(pattern=args.raw_text, input_text=initial_pipe_input_content)}]
            else:
                messages = [{"role": "user", "content": processed_user_content}]
            
            if not client and not args.grep_enabled and not ("@search" in user_content or "@file" in user_content or "@url" in user_content):
                 sys.exit("Error: Main LLM client not initialized and no directives found in raw_text input.")
            elif not client and args.grep_enabled: # Should have been caught earlier but as a safeguard
                 sys.exit("Error: Main LLM client not initialized, required for --grep-enabled.")


            # Only call stream_response if not solely relying on directives processed by process_input_directives
            # and if client is available for a chat completion. Grep mode always needs it.
            # If it was just @file/@url/@search, process_input_directives handled it.
            # This path (raw_text) usually implies a direct query to the LLM.
            if client: # If main client is available
                response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
                if response_content is not None and not response_content.endswith('\n'): sys.stdout.write("\n")
            elif not ("@search" in user_content or "@file" in user_content or "@url" in user_content):
                # No client and no directives that could have produced output (like search results to context)
                sys.stderr.write("No main LLM client and no directives to process. Exiting.\n")
            
            sys.stdout.flush(); sys.exit(0)

        if not client:
             sys.exit(f"Error: Main LLM client failed to initialize. Interactive mode requires a functioning main LLM. API URL: {args.api_url}")

        sys.stderr.write("Entering interactive mode. Use Ctrl+D to submit, !cmd, @file, @url, @search, Tab for completion:\n")
        while True:
            sys.stderr.write(f"\n{ANSI_RESET}💥 Ask (Ctrl+D Submit, !cmd, @file, @url, @search, Tab completion):\n")
            current_prompt_accumulator: List[str] = []
            if initial_pipe_input_content and not has_used_initial_pipe_input:
                current_prompt_accumulator.append(f"Context from initial piped input:\n{initial_pipe_input_content.strip()}")
            while True:
                prompt_char = "  " if '\n'.join(current_prompt_accumulator).strip() else "> "
                try:
                    user_line_input = input(prompt_char)
                    stripped_user_line = user_line_input.strip()
                    if stripped_user_line.startswith('!'):
                        command_str = stripped_user_line[1:].strip()
                        if command_str.lower() == 'clear':
                            messages.clear(); current_prompt_accumulator.clear()
                            sys.stdout.write(ANSI_CLEAR_SCREEN); sys.stdout.flush()
                            sys.stderr.write("Conversation history and current input cleared.\n"); break
                        elif command_str:
                            sys.stderr.write(f"\nExecuting: `{command_str}`\n---\n")
                            cmd_output = execute_command(command_str)
                            sys.stderr.write(cmd_output)
                            if not cmd_output.endswith('\n'): sys.stderr.write("\n")
                            sys.stderr.write("---\n"); sys.stderr.flush()
                            current_prompt_accumulator.append(f"--- User executed command `{command_str}` output ---\n{cmd_output.strip()}\n--- End command output ---")
                            sys.stderr.write("Command output added to prompt. Continue or Ctrl+D.\n")
                        else: current_prompt_accumulator.append(user_line_input)
                    else: current_prompt_accumulator.append(user_line_input)
                except EOFError: sys.stderr.write(ANSI_GREEN + "^D EOF!\n" + ANSI_RESET); break
                except KeyboardInterrupt:
                    sys.stderr.write(ANSI_GREEN + "\n^C Cancelled!\n" + ANSI_RESET)
                    readline_buf_non_empty = False
                    if is_readline_available_in_main and 'readline' in sys.modules and hasattr(sys.modules['readline'], 'get_line_buffer'):
                        try:
                            if sys.modules['readline'].get_line_buffer(): readline_buf_non_empty = True
                        except Exception: pass
                    if bool(''.join(current_prompt_accumulator).strip()) or readline_buf_non_empty:
                        current_prompt_accumulator.clear() 
                    else: sys.stderr.write(ANSI_RESET); sys.stderr.flush(); raise
            
            final_prompt_str = '\n'.join(current_prompt_accumulator).strip()
            if not final_prompt_str: continue
            if initial_pipe_input_content and not has_used_initial_pipe_input and \
               any(initial_pipe_input_content.strip() in p for p in current_prompt_accumulator):
                has_used_initial_pipe_input = True
            
            processed_input = process_input_directives(
                final_prompt_str, client, args.model,
                args.search_kw_api_url, args.search_kw_model, args.search_kw_api_key,
                args.search_kw_max_tokens, args.search_kw_temperature
            )
            messages.append({"role": "user", "content": processed_input})
            sys.stdout.write(f"{ANSI_RESET}💡:\n"); sys.stdout.flush()
            response_content = stream_response(client, args.model, args.model_type, messages, args.max_tokens, freq_penalty, pres_penalty, args.hide_reasoning)
            if response_content is not None:
                messages.append({"role": "assistant", "content": response_content})
                if response_content and not response_content.endswith('\n'): sys.stdout.write("\n")
                sys.stdout.flush()
            sys.stdout.write(ANSI_RESET); sys.stdout.flush()
    except KeyboardInterrupt: sys.stderr.write(f"\n{ANSI_RESET}Exiting due to user interrupt.\n")
    except Exception as e:
        sys.stderr.write(f"\n{ANSI_RESET}Unexpected error in main loop: {e}\n"); traceback.print_exc(file=sys.stderr)
    finally:
        if TERMIOS_AVAILABLE and original_termios_settings is not None:
            current_stdin_is_tty = False; current_stdin_fd_val = -1
            try:
                if sys.stdin and not sys.stdin.closed:
                    current_stdin_fd_val = sys.stdin.fileno()
                    if current_stdin_fd_val >=0 and os.isatty(current_stdin_fd_val): current_stdin_is_tty = True
            except Exception: pass
            if current_stdin_is_tty: restore_terminal_settings(current_stdin_fd_val, original_termios_settings)
        restore_stdin(original_stdin_fd)
        sys.stderr.write(ANSI_RESET); sys.stdout.write(ANSI_RESET); sys.stderr.flush(); sys.stdout.flush()

if __name__ == '__main__':
    main()

