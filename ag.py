import argparse
import fileinput
import sys
import os
import re
from openai import OpenAI
import requests
from bs4 import BeautifulSoup


def fetch_webpage(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text().strip()
    except requests.RequestException as e:
        sys.stderr.write(f"Error fetching {url}: {str(e)}\n")
        return None


def process_input(input_str):
    processed_lines = []
    for line in input_str.split('\n'):
        if line.strip().startswith('@file'):
            match = re.match(r'@file\(([^)]+)\)', line.strip())
            if match:
                filename = match.group(1)
                try:
                    with open(filename, 'r') as f:
                        processed_lines.append(f"{filename}: {f.read()}")
                except Exception as e:
                    sys.stderr.write(f"Error reading {filename}: {str(e)}\n")
        elif line.strip().startswith('@url'):
            match = re.match(r'@url\(([^)]+)\)', line.strip())
            if match:
                url = match.group(1)
                webpage_content = fetch_webpage(url)
                if webpage_content:
                    processed_lines.append(f"Webpage: {webpage_content}")
        else:
            processed_lines.append(line)
    return '\n'.join(processed_lines)

def stream_response(client, model, messages, max_tokens):
    full_response = []
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                sys.stdout.write(content)
                sys.stdout.flush()
                full_response.append(content)
        return ''.join(full_response)
    except KeyboardInterrupt:
        sys.stderr.write("\nResponse generation interrupted by user.\n")
        return None
    except Exception as e:
        sys.stderr.write(f"\nAPI Error: {str(e)}\n")
        return None

def main():
    parser = argparse.ArgumentParser(description='AG - Ask GPT from CLI')
    parser.add_argument('--api-url', help='Custom API endpoint', default="http://localhost:11434/v1")
    parser.add_argument('--model', help='Model name', default='qwq:32b')
    parser.add_argument('--api-key', help='API key', default="sk-zoxrdmircomznftiqcpijfrvrrgtzzfsceayptfizlqknhkb")
    parser.add_argument('--max_tokens', help='Maximum tokens', default=4096)
    parser.add_argument('question', nargs='?', help='Direct question to ask GPT')
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Error: API key required. Set OPENAI_API_KEY env var or use --api-key")

    client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key
    )

    messages = []
    pipe_input = ""

    original_stdin_fd = os.dup(0)
    if not sys.stdin.isatty():
        pipe_input = sys.stdin.read()


    try:
        os.close(0)
        tty_fd = os.open('/dev/tty', os.O_RDONLY)
        assert tty_fd == 0
        sys.stdin = open(0, 'r', closefd=False)
        import readline
        readline.parse_and_bind('tab: complete')
    except ImportError:
        pass

    if args.question:
        # Directly handle the question without entering interactive mode
        processed = process_input(pipe_input+args.question)
        pipe_input = ""
        messages.append({"role": "user", "content": processed})
        sys.stdout.write("\n💡:\n")
        response_content = stream_response(client, args.model, messages, args.max_tokens)
        if response_content:
            messages.append({"role": "assistant", "content": response_content})
        sys.stdout.write("\n")

    while True:
        try:
            lines = []

            sys.stderr.write("\n💥 ^D:\n")

            while True:
                try:
                    line = input("> ")
                    lines.append(line)
                except EOFError:
                    break


            user_input = '\n'.join(lines).strip()
            user_input = pipe_input + "\n" + user_input

            if not user_input.strip():
                sys.stderr.write("\nNo input received.\n")
                continue

            processed = process_input(user_input)
            messages.append({"role": "user", "content": processed})

            sys.stdout.write("\n💡:\n")

            response_content = stream_response(client, args.model, messages, args.max_tokens)
            if response_content:
                messages.append({"role": "assistant", "content": response_content})
            sys.stdout.write("\n")

            pipe_input = ""
            

        except KeyboardInterrupt:
            sys.stderr.write("\nExiting...\n")
            os.dup2(original_stdin_fd, 0)
            os.close(original_stdin_fd)
            break

if __name__ == '__main__':
    main()
