#!/usr/bin/env python3
"""
将 codex_swebenchpro_format.json 中的每条 conversation 拆分为独立 JSON 文件，
并转换为大模型 chat template 格式（user/assistant 角色）。

输入格式：
  [
    {
      "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt",   "value": "..."},
        ...
      ]
    },
    ...
  ]

输出格式（每个文件一条 record）：
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
  }

使用方法：
    python scripts/split_codex_swebenchpro.py \
        --input codex_swebenchpro_format.json \
        --output-dir conversations_split
"""

import argparse
import json
import os
from pathlib import Path

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
}


def convert_conversation(record: dict) -> dict:
    """将单条 record 的 conversations 转换为 chat template 格式。"""
    messages = []
    for turn in record.get("conversations", []):
        role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        messages.append({
            "role": role,
            "content": turn.get("value", ""),
        })
    return {"messages": messages}


def process_file(input_path: Path, output_dir: Path) -> int:
    """读取大 JSON 文件，逐条拆分并写入输出目录。返回写入的文件数。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"期望 JSON 顶层为数组，实际为 {type(data).__name__}")

    count = 0
    total = len(data)
    digits = len(str(total))

    for i, record in enumerate(data):
        converted = convert_conversation(record)
        out_path = output_dir / f"conversation_{i:0{digits}d}.json"
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(converted, out_f, ensure_ascii=False)
        count += 1

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  进度: {i + 1}/{total}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="将 codex_swebenchpro_format.json 拆分为独立 chat template 文件"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入 JSON 文件路径",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="输出目录路径",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return

    output_dir = Path(args.output_dir)

    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")

    count = process_file(input_path, output_dir)

    print(f"\n完成。共生成 {count} 个文件。")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
