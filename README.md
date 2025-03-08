![](https://s2.loli.net/2025/03/08/YLGmvlR7VoEB3tN.png)
![](https://s2.loli.net/2025/03/07/8zOiCVWGXYDUywJ.png) 

# AG - Ask GPT from CLI

**AG** is a command-line interface (CLI) tool that enables interaction with GPT models, including custom models hosted on platforms like SiliconFlow. It supports fetching content from webpages, reading files, and streaming responses in real-time, making it a versatile tool for developers, researchers, and anyone who needs quick access to GPT-powered assistance directly from the terminal. Just as Professor Jiang Yanyan demonstrated in class, simply type `ag "your question"` to effortlessly obtain intelligent responses from GPT, ensuring efficiency and convenience.

## Features

- **Interactive Chat Mode**: Engage in a conversation with the GPT model directly from the terminal.
- **File and URL Integration**: Fetch content from local files (`@file(filename)`) or webpages (`@url(URL)`) to include in your queries.
- **Streaming Responses**: Responses are streamed in real-time for a smooth and interactive experience.
- **Custom API Support**: Easily configure the tool to use a custom API endpoint and model.
- **API Key Management**: Supports API key configuration via command-line arguments or environment variables.

## Usage

### Basic Setup

```bash
git clone https://github.com/moeleak/ag.git
cd ag
pip install -r requirements.txt
```

### Running the Script

```bash
python ag.py --api-key YOUR_API_KEY
```

### Examples

1. **Fetch Content from a File**:
   ```
   > @file(example.txt)
   ```

2. **Fetch Content from a Webpage**:
   ```
   > @url(https://example.com)
   ```

3. **Interactive Chat**:
   ```
   > Tell me a joke.
   ```

### Command-Line Options

- `--api-url`: Custom API endpoint (default: `https://api.siliconflow.cn/v1`)
- `--model`: Model name (default: `deepseek-ai/DeepSeek-V3`)
- `--api-key`: API key (required if not set via environment variable)

## Requirements

- Python 3.6+
- `openai` Python library
- `requests` Python library
- `beautifulsoup4` Python library

