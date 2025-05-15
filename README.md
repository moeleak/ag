# AG - AI Command Line Assistant

[![asciicast](https://asciinema.org/a/R2KOpEIjL7HT4QzxLNuICaDTQ.svg)](https://asciinema.org/a/R2KOpEIjL7HT4QzxLNuICaDTQ)

## Overview
AG is a powerful command-line interface (CLI) tool that allows you to interact with OpenAI-compatible AI models directly from your terminal. It supports both single-query mode and interactive chat sessions, with advanced features like file/URL content integration, command execution, and grep-like filtering.

## Key Features

- **Multi-API Compatibility**: Works with any OpenAI-compatible API (including Ollama, LiteLLM, etc.)
- **Dual Model Support**: Optimized for both OpenAI-style and Gemini-style model behaviors
- **Smart Content Integration**:
  - `@file(path)` - Directly include file contents
  - `@url(URL)` - Fetch and process webpage content
  - `@search(query)` - Web search results integration
- **Built-in Command Execution**:
  - Execute shell commands with `!command` syntax
  - View output directly in your session
- **Intelligent Text Filtering**:
  - AI-powered grep with natural language patterns (`-g` flag)
  - Pipe-aware input handling
- **Enhanced Terminal Experience**:
  - Real-time streaming responses
  - Optional reasoning visibility
  - Full terminal control with proper TTY management

## Installation
```bash
pip install openai requests beautifulsoup4 wcwidth
```

## Usage

### Basic Query
```bash
ag "Your question here"
```

### Interactive Mode
```bash
ag
```

### With Piped Input
```bash
cat file.txt | ag "Your question about the file content"
```

### AI-Powered Grep Filter
```bash
cat logfile.txt | ag -g "error messages related to database connections"
```

### Include File/URL Content / Search from web
```
@file(/path/to/file.txt)
@url(https://example.com)
@search(your search content)
```

## Command Syntax

### Directives
- `@file(path)`: Include content from a file
- `@url(URL)`: Include content from a webpage

### Shell Commands (interactive mode only)
- `!command`: Execute a shell command
- `!clear`: Clear the conversation history

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--api-url` | API endpoint URL | `http://localhost:11434/v1` |
| `--model` | Model name to use | `deepseek-reasoner` |
| `--model-type` | Model API type (`openai` or `gemini`) | `openai` |
| `--api-key` | API key | `sk-...` or `OPENAI_API_KEY` env var |
| `--max-tokens` | Maximum tokens for response | `4096` |
| `--frequency-penalty` | Frequency penalty (0.0-2.0, -1 to disable) | `0.5` |
| `--presence-penalty` | Presence penalty (0.0-2.0, -1 to disable) | `0.5` |
| `-g/--grep-enabled` | Enable AI-powered grep filter mode | `False` |
| `-r/--hide-reasoning` | Hide model reasoning content | `False` |

## Examples

1. Query about local files:
```bash
ag "@file(~/projects/notes.txt) summarize the key points"
```
2. Research a webpage:
```bash
ag "@url(https://en.wikipedia.org/wiki/Machine_learning) what are the three main types of machine learning?"
```

3. Interactive research session:
```
ag
> @url(https://news.ycombinator.com/)
> what are today's top AI stories?
> !curl https://api.example.com/data | jq .stats
> analyze these statistics
```

4. Web search integration:
```
ag "@search(2024 AI trends) summarize top 3 trends"
```

## Keybindings (Interactive Mode)

- **Ctrl+D**: Submit current input
- **Ctrl+C**: Cancel current input or exit if input is empty
- **Tab**: Auto-completion (when readline is available)

