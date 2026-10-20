"""本地大模型命令行助手 —— 通过 HTTP API 调用本地 Ollama

用法:
    python src/ai.py "用一句话解释什么是机器学习"
    python src/ai.py "帮我写一个冒泡排序" --system "你是一个资深 Python 工程师，回答要简洁"
    python src/ai.py "你好" --no-stream                          # 关闭流式输出
    python src/ai.py "你好" --model qwen2.5:0.5b                 # 换一个模型

设计说明:
    1. 全程走 Ollama 的 HTTP 接口（默认 http://127.0.0.1:11434），不依赖 ollama 命令行。
       好处：返回的是结构化 JSON，可以精确控制流式输出；
       以后要换成别的模型服务（OpenAI 格式），基本只需要改这一个地址。
    2. 默认流式输出（stream=True）：模型生成一个词就打印一个词，体验接近 ChatGPT。
    3. 运行前必须保证后台有 Ollama 服务：ollama serve

本机踩坑备忘:
    本机 AMD R7 M260 的 Vulkan 驱动会导致 Ollama 推理引擎崩溃（0xc0000005），
    已通过环境变量 OLLAMA_VULKAN=0 强制走 CPU。完整排查过程见 logs/2026-10-09.md
"""
import argparse
import json
import sys

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:1.5b"


def build_payload(prompt: str, model: str, system: str | None, stream: bool) -> dict:
    """把命令行参数打包成 Ollama 接口需要的 JSON。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if system:
        payload["system"] = system
    return payload


def ask_stream(prompt: str, model: str, system: str | None) -> None:
    """流式提问：模型每生成一小段，就立刻打印出来。"""
    with requests.post(
        OLLAMA_URL,
        json=build_payload(prompt, model, system, stream=True),
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                print(f"\n[模型服务报错] {chunk['error']}", file=sys.stderr)
                sys.exit(1)
            print(chunk.get("response", ""), end="", flush=True)
    print()  # 全部生成完，补一个换行


def ask_once(prompt: str, model: str, system: str | None) -> None:
    """非流式提问：等模型全部生成完，再一次性打印。"""
    resp = requests.post(
        OLLAMA_URL,
        json=build_payload(prompt, model, system, stream=False),
        timeout=300,
    )
    resp.raise_for_status()
    print(resp.json()["response"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="本地大模型命令行助手（基于本地 Ollama）",
        epilog='示例: python src/ai.py "用一句话解释什么是机器学习"',
    )
    parser.add_argument("prompt", help="你想问的问题")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名，默认 {DEFAULT_MODEL}")
    parser.add_argument("--system", default=None, help="系统提示词，用来设定角色或回答风格")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出，等全部生成完再显示")
    args = parser.parse_args()

    try:
        if args.no_stream:
            ask_once(args.prompt, args.model, args.system)
        else:
            ask_stream(args.prompt, args.model, args.system)
    except requests.exceptions.ConnectionError:
        print("连不上本地 Ollama。请先启动服务：ollama serve", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"请求出错：{exc}", file=sys.stderr)
        print("提示：模型名是否写对？可用 ollama list 查看已下载的模型。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
