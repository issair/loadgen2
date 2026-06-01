#!/usr/bin/env python3
"""
将 extracted_files/ 下的多轮对话 JSONL 拆分为独立请求文件。

每个文件包含一条完整对话（含 system + 多轮 user/assistant/tool）。
拆分为：每个 assistant 响应之前的消息组成一个独立请求。

拆分规则：
  - 找到所有 role == "assistant" 的消息位置
  - 对于每个 assistant 位置 i（i > 0），生成请求：messages[:i]
  - 即：请求 = 发送给模型的所有历史消息（不含模型的响应）
  - 最后追加完整的 messages[:] 作为最终请求

命名规则：
  - 输出到 extracted_files_split/ 目录
  - 文件名 = 原文件名.stem + "_" + 两位顺序编号 + ".jsonl"
  - 例如：20260129000158fc281d78bbb64e07_01.jsonl

使用方法：
    python scripts/split_conversations.py
    python scripts/split_conversations.py --input-dir extracted_files --output-dir extracted_files_split
"""

import argparse
import json
import os
from pathlib import Path


def load_jsonl(file_path: Path) -> list:
    """读取 JSONL 文件，返回所有 JSON 对象的列表。"""
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return [data]


def split_requests(messages: list, tools: list) -> list:
    """根据 assistant 消息位置拆分请求。

    对于每个 assistant 消息位置 i（i > 0），生成请求：messages[:i]
    即该请求是发送给模型以获取该 assistant 响应的完整上下文。
    最后，追加完整的 messages[:] 作为最终请求。
    """
    assistant_positions = [
        i
        for i, msg in enumerate(messages)
        if isinstance(msg, dict) and msg.get("role") == "assistant"
    ]

    requests = []
    for pos in assistant_positions:
        if pos > 0:
            req = {
                "messages": messages[:pos],
                "tools": tools,
            }
            requests.append(req)

    # Append the full message list
    requests.append({
        "messages": messages[:],
        "tools": tools,
    })

    return requests


def process_file(file_path: Path, output_dir: Path, starting_suffix: int = 1) -> int:
    """处理单个文件，拆分并写入输出。返回生成的请求数。"""
    records = load_jsonl(file_path)
    if not records:
        return 0

    base_name = file_path.stem
    suffix = starting_suffix
    total_written = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        messages = rec.get("messages", [])
        tools = rec.get("tools", [])

        requests = split_requests(messages, tools)
        for req in requests:
            out_file = output_dir / f"{base_name}_{suffix:03d}.jsonl"
            with out_file.open("w", encoding="utf-8") as f:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
            suffix += 1
            total_written += 1

    return total_written


def main():
    parser = argparse.ArgumentParser(description="拆分多轮对话 JSONL 为独立请求文件")
    parser.add_argument(
        "--input-dir",
        "-i",
        default="extracted_files",
        help="输入目录（默认: extracted_files）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="extracted_files_split",
        help="输出目录（默认: extracted_files_split）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(list(input_dir.glob("*.jsonl")) + list(input_dir.glob("*.json")))
    if not input_files:
        print(f"未在 {input_dir}/ 找到 .jsonl 或 .json 文件")
        return

    total_requests = 0
    total_files = 0
    for f in input_files:
        nb = process_file(f, output_dir)
        total_requests += nb
        total_files += 1
        print(f"  {f.name}: {nb} requests")

    print(f"\n完成。处理 {total_files} 个文件，生成 {total_requests} 个请求文件。")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
