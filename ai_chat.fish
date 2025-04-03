#!/usr/bin/env fish

function agg
    $HOME/miniconda3/bin/python /Users/lolimaster/projects/ag/ag.py $argv \
        --model "models/gemini-2.5-pro-exp-03-25" \
        --model-type gemini \
        --api-url "http://localhost:7001/v1/" \
        --api-key linuxdo \
        --frequency-penalty -1.0 \
        --presence-penalty -1.0 \
        --max-tokens 100000
end

function ags
    $HOME/miniconda3/bin/python /Users/lolimaster/projects/ag/ag.py $argv \
        --model Pro/deepseek-ai/DeepSeek-R1 \
        --api-url "https://api.siliconflow.cn/v1" \
        --max-tokens 16384 \
        --api-key sk-zoxrdmircomznftiqcpijfrvrrgtzzfsceayptfizlqknhkb \
        --frequency-penalty -1.0 \
        --presence-penalty -1.0
end
# --- 配置 ---
# 初始发送给 DeepSeek 的消息
set initial_message 你好

# Gemini 的提问模板， $(last_response) 会被 DeepSeek 的上一句回答替换
set gemini_prompt_template "你要回答deepseek的回答："

# --- 脚本开始 ---

# 检查 agg 命令是否存在
# 假设 ags -r 是一个有效的命令或别名/函数来调用 DeepSeek
# 如果不是，你可能需要将其替换为实际调用 DeepSeek 的 agg 命令，例如：
# alias ags-r="agg deepseek" 或者直接在脚本中使用 agg deepseek

echo "对话开始..."
echo ----------------------------------------

# 1. 让 DeepSeek 先回应初始消息
echo ">>> Me (Initial Prompt to DeepSeek):"
echo $initial_message
echo ""
echo ">>> DeepSeek (ags -r):"
set last_response (ags -r "$initial_message")
if test $status -ne 0
    echo "错误：调用 'ags -r' 失败。" >&2
    exit 1
end
echo $last_response
echo ----------------------------------------
sleep 1 # 短暂暂停

# 2. 开始无限循环对话
while true
    # 2.1 Gemini 回应 DeepSeek
    set current_prompt "$gemini_prompt_template $last_response"
    echo ">>> Gemini (Responding to DeepSeek):"
    # echo "(Prompt: $current_prompt)" # 如果需要，取消注释以查看发送给 Gemini 的完整提示

    # 使用 eval 来确保变量在模板字符串中正确展开 (虽然在这个简单模板中可能不是必须的)
    # 或者直接拼接字符串
    set current_prompt "$gemini_prompt_template $(echo $last_response)"

    set gemini_response (agg "$current_prompt")
    if test $status -ne 0
        echo "错误：调用 'agg' 失败。正在退出..." >&2
        break # 发生错误时退出循环
    end
    if test -z "$gemini_response"
        echo "警告：Gemini 返回了空回复。正在退出..." >&2
        break # 空回复时退出
    end
    echo $gemini_response
    set last_response $gemini_response # 更新最后的回应为 Gemini 的回应
    echo ----------------------------------------
    sleep 1 # 短暂暂停

    # 2.2 DeepSeek 回应 Gemini
    echo ">>> DeepSeek (ags -r) (Responding to Gemini):"
    # echo "(Prompt: $last_response)" # 如果需要，取消注释以查看发送给 DeepSeek 的提示

    set deepseek_response (ags -r "$last_response")
    if test $status -ne 0
        echo "错误：调用 'ags -r' 失败。正在退出..." >&2
        break # 发生错误时退出循环
    end
    if test -z "$deepseek_response"
        echo "警告：DeepSeek 返回了空回复。正在退出..." >&2
        break # 空回复时退出
    end
    echo $deepseek_response
    set last_response $deepseek_response # 更新最后的回应为 DeepSeek 的回应
    echo ----------------------------------------
    sleep 1 # 短暂暂停
end

echo "对话结束。"
