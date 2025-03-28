import argparse
import fileinput
import sys
import os
import re
import requests
import subprocess
from openai import OpenAI
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

def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr.decode('utf-8')}"


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


def stream_response(client, model, messages, max_tokens, frequency_penalty, presence_penalty):
    full_response = []  # 收集完整的响应内容（含标签）
    buffer = ''
    content = ''
    reasoning_content = ''
    in_think = False
    sys.stdout.write('\033[s')  # 保存光标起始位置
    sys.stdout.flush()
    try:
        base_params = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "stream": True
        }
        if frequency_penalty != "-1.0":
            base_params["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty != "-1.0":
            base_params["presence_penalty"] = float(presence_penalty)

        # if (frequency_penalty == "-1.0") and (presence_penalty == "-1.0"): # Disable
        #     stream = client.chat.completions.create(
        #         model=model,
        #         messages=messages,
        #         max_tokens=int(max_tokens),
        #         stream=True
        #     )
        # else:
        #     stream = client.chat.completions.create(
        #         model=model,
        #         messages=messages,
        #         max_tokens=int(max_tokens),
        #         frequency_penalty=float(frequency_penalty),
        #         presence_penalty=float(presence_penalty),
        #         stream=True
        #         )
        stream = client.chat.completions.create(**base_params)

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue
            buffer += content

            # reasoning_content += chunk.choices[0].delta.reasoning_content
            # print(reasoning_content)

            while True:
                if not in_think:
                    start_pos = buffer.find('<think>')
                    if start_pos == -1:
                        # 输出非思考内容
                        sys.stdout.write(buffer)
                        sys.stdout.flush()
                        full_response.append(buffer)
                        buffer = ''
                        break
                    else:
                        # 处理<think>前的内容
                        before_think = buffer[:start_pos]
                        sys.stdout.write(before_think)
                        sys.stdout.flush()
                        full_response.append(before_think)
                        # 记录<think>标签
                        full_response.append('<think>')
                        buffer = buffer[start_pos+7:]
                        in_think = True
                        sys.stdout.write('\033[32m')  # 开启绿色
                        sys.stdout.flush()
                else:
                    end_pos = buffer.find('</think>')
                    if end_pos == -1:
                        # 输出思考内容（无闭合标签）
                        sys.stdout.write(buffer)
                        sys.stdout.flush()
                        full_response.append(buffer)
                        buffer = ''
                        break
                    else:
                        # 输出闭合前的思考内容
                        think_content = buffer[:end_pos]
                        sys.stdout.write(think_content)
                        sys.stdout.flush()
                        full_response.append(think_content)
                        # 记录</think>标签
                        full_response.append('</think>')
                        buffer = buffer[end_pos+8:]
                        in_think = False
                        sys.stdout.write('\033[0m')  # 关闭绿色
                        sys.stdout.flush()
        # 处理剩余buffer
        # if buffer:
        #    if in_think:
        #        sys.stdout.write(buffer)
        #        full_response.append(buffer)
        #    else:
        #        sys.stdout.write(buffer)
        #        full_response.append(buffer)
        #    sys.stdout.flush()
        # 生成处理后的响应（移除思考块）
        full_response_str = ''.join(full_response)
        processed_response = re.sub(r'<think>.*?</think>', '', full_response_str, flags=re.DOTALL)
        # 清除原内容并输出处理后的结果
        #sys.stdout.write('\033[u')        # 恢复原始位置
        #sys.stdout.write('\033[2J\033[H') # 全屏清除+光标归位
        #sys.stdout.write(processed_response)
        #sys.stdout.flush()
        return processed_response
    except KeyboardInterrupt:
        sys.stdout.write('\033[0m')  # 关闭绿色
        sys.stdout.flush()
        sys.stderr.write("\nResponse generation interrupted by user.\n")
        return None
    except Exception as e:
        sys.stdout.write('\033[0m')  # 关闭绿色
        sys.stdout.flush()
        sys.stderr.write(f"\nAPI Error: {str(e)}\n")
        return None




def main():
    parser = argparse.ArgumentParser(description='AG - Ask GPT from CLI')
    parser.add_argument('--api-url', help='Custom API endpoint', default="http://localhost:11434/v1")
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--api-key', help='API key')
    parser.add_argument('--max_tokens', help='Maximum tokens', default=4096)
    parser.add_argument('--frequency_penalty', help='Frequency penalty', default=0.5)
    parser.add_argument('--presence_penalty', help='Presence penalty', default=0.5)
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
        sys.stdout.write("\n💡")
        response_content = stream_response(client, args.model, messages, args.max_tokens, args.frequency_penalty, args.presence_penalty)
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
                    # if not line or line.isspace():
                    #    break
                    lines.append(line)
                except EOFError:
                    break


            user_input = '\n'.join(lines).strip()
            user_input = pipe_input + "\n" + user_input

            if not user_input.strip():
                sys.stderr.write("\nNo input received.\n")
                continue


            isContinue = False
            for line in user_input.split('\n'):
                if line.strip().startswith('!'):
                    command_to_execute=line[1:]
                    if command_to_execute.lower() == 'clear':
                        messages.clear()
                        output_from_command=execute_command(command_to_execute)
                        sys.stdout.write(output_from_command)
                        isContinue = True
                    else:
                        sys.stdout.write("\n")
                        output_from_command=execute_command(command_to_execute)
                        sys.stdout.write(output_from_command)
                        isContinue = True

            if isContinue:
                continue


            processed = process_input(user_input)
            messages.append({"role": "user", "content": processed})
            sys.stdout.write("\n💡:\n")



            response_content = stream_response(client, args.model, messages, args.max_tokens, args.frequency_penalty, args.presence_penalty)
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
