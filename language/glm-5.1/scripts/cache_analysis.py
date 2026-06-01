#!/usr/bin/env python3
"""
分析批量 JSONL 数据集的 LLM Prompt Caching 缓存命中率。

使用方法：
    python scripts/cache_analysis.py
    python scripts/cache_analysis.py --data-dir . --output result.csv
    python scripts/cache_analysis.py --max-files 100

统计方式：
    - 将每个 JSONL 文件视为一次完整的 API 请求
    - 通过 Trie 前缀树匹配跨请求的共同 token 前缀
    - Shortest-First 顺序发送（system prompt 分组后按总 token 升序）
    - 模拟无缓存过期的理想命中率
    - 按文件名前缀分组，组内共享 Trie 状态
"""

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 设置国内镜像
HF_ENDPOINT = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# 进度条
# ============================================================
class ProgressTracker:
    """简单的控制台进度条，打印到 stderr 不干扰正常输出。"""

    def __init__(self, total: int, label: str, interval: int = 50):
        self.total = total
        self.label = label
        self.interval = interval
        self.current = 0
        self.start_time = time.time()
        self.last_print = 0

    def update(self, n: int = 1) -> None:
        self.current += n
        if (
            self.current - self.last_print >= self.interval
            or self.current == self.total
        ):
            self.last_print = self.current
            elapsed = time.time() - self.start_time
            pct = self.current / self.total if self.total > 0 else 0
            speed = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / speed if speed > 0 else 0
            bar_len = 20
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stderr.write(
                f"\r{self.label}: [{bar}] {pct:5.1%} "
                f"({self.current}/{self.total}) "
                f"{speed:.1f}it/s ETA:{eta:.0f}s"
            )
            sys.stderr.flush()
        if self.current >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()


# ============================================================
# Trie 前缀树
# ============================================================
class TrieNode:
    __slots__ = ("children",)

    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}


class PrefixTrie:
    """用 Trie 存储所有已见过的 token 序列，支持最长前缀匹配。"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens: List[int]) -> None:
        node = self.root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                child = TrieNode()
                node.children[token] = child
            node = child

    def longest_prefix_length(self, tokens: List[int]) -> int:
        """返回 tokens 与 Trie 中任一序列的最长公共前缀长度。"""
        node = self.root
        length = 0
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                break
            node = child
            length += 1
        return length


# ============================================================
# Tokenizer 相关
# ============================================================
def setup_tokenizer(model_name: str):
    import os as _os

    _os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    from transformers import AutoTokenizer

    logger.info(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.model_max_length = int(1e9)  # 消除超长序列警告
    logger.info("Tokenizer 加载成功")
    return tokenizer


def fix_tool_call_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """修复 tool_calls 中 arguments 的双重 JSON 编码问题。"""
    fixed = []
    for msg in messages:
        m = dict(msg)
        tool_calls = m.get("tool_calls", [])
        if tool_calls:
            fixed_tcs = []
            for tc in tool_calls:
                ftc = dict(tc)
                func = ftc.get("function", {})
                if isinstance(func, dict):
                    func = dict(func)
                    if isinstance(func.get("arguments"), str):
                        parsed = func["arguments"]
                        while isinstance(parsed, str):
                            try:
                                parsed = json.loads(parsed)
                            except (json.JSONDecodeError, TypeError, ValueError):
                                break
                        func["arguments"] = parsed
                    ftc["function"] = func
                fixed_tcs.append(ftc)
            m["tool_calls"] = fixed_tcs
        fixed.append(m)
    return fixed


def render_tokens(
    data: Dict[str, Any], tokenizer
) -> Tuple[List[int], int, Optional[str]]:
    """渲染单条请求的完整 token 序列。

    Returns:
        (tokens, static_prefix_length, error)
    """
    messages = data.get("messages", [])
    tools = data.get("tools", [])

    if not messages:
        return [], 0, "empty messages"

    try:
        fixed = fix_tool_call_arguments(messages)
        tokens = tokenizer.apply_chat_template(
            fixed,
            tools=tools if tools else None,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template 失败: {e}")
        # 回退：使用 fix 后但不传 tools 参数
        try:
            fixed = fix_tool_call_arguments(messages)
            tokens = tokenizer.apply_chat_template(
                fixed,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        except Exception as e2:
            logger.warning(f"回退也失败: {e2}")
            return [], 0, str(e2)

    # 检测 static prefix 边界：找到 message[1]（第一个非 system 消息）
    # 在 token 序列中定位 <|user|>（154827）或 <|assistant|>（154828）的位置
    # 排除 tools 渲染中的 user 指示（如果有）
    # 方法：取 messages[1] 在模板中渲染的开始位置
    try:
        # 只渲染前 1 条消息（system） + tools，作为 static prefix 的近似
        static_fixed = fix_tool_call_arguments(messages[:1])
        if tools:
            static_tokens = tokenizer.apply_chat_template(
                static_fixed,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        else:
            static_tokens = tokenizer.apply_chat_template(
                static_fixed,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        static_prefix_len = len(static_tokens)
    except Exception:
        # 回退：在 token 序列中找第一个 <|observation|> 或 <|user|>
        TOKEN_USER = 154827
        TOKEN_ASSISTANT = 154828
        static_prefix_len = 0
        for t in tokens:
            if t in (TOKEN_USER, TOKEN_ASSISTANT):
                break
            static_prefix_len += 1

    return tokens, static_prefix_len, None


def get_system_content(data: Dict[str, Any]) -> str:
    """提取 system prompt 文本，用于分组。"""
    messages = data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            return msg.get("content", "")
    return messages[0].get("content", "")


def get_system_group_key(data: Dict[str, Any]) -> str:
    """生成分组 key：system content + tools hash。"""
    system = get_system_content(data)
    tools = data.get("tools", [])
    tools_str = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    return hashlib.md5((system + tools_str).encode("utf-8")).hexdigest()


def get_file_group_key(file_path: str) -> str:
    """提取文件名前缀作为分组 key，用于跨文件 cache 隔离。"""
    name = Path(file_path).stem
    # 文件名前缀：去掉末尾数字后缀（处理 input-1.jsonl → input; r001.json → r）
    import re

    return re.sub(r"[_-]?\d+$", "", name) or name


def group_files_by_prefix(file_paths: List[str]) -> Dict[str, List[str]]:
    """按文件名前缀分组。"""
    groups = defaultdict(list)
    for fp in file_paths:
        key = get_file_group_key(fp)
        groups[key].append(fp)
    return dict(groups)


# ============================================================
# Worker 全局变量（用于多进程初始化）
# ============================================================
_worker_tokenizer = None
_worker_verbose = False


def _init_worker(tokenizer_name: str, verbose: bool):
    """进程初始化函数：每个 Worker 进程只加载一次 tokenizer。"""
    global _worker_tokenizer, _worker_verbose
    import os as _os

    _os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    from transformers import AutoTokenizer

    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=True
    )
    _worker_tokenizer.model_max_length = int(1e9)  # 消除超长序列警告
    _worker_verbose = verbose


def _process_group_worker(args: Tuple) -> List[Dict[str, Any]]:
    """Worker 函数：处理一个文件前缀组的完整分析（包括 Trie 匹配）。"""
    group_key, file_paths = args
    results = []

    trie = PrefixTrie()

    for file_path in file_paths:
        json_records = load_jsonl(file_path)
        for json_data in json_records:
            tokens, static_len, error = render_tokens(json_data, _worker_tokenizer)
            total = len(tokens)

            if error or total == 0:
                results.append(
                    {
                        "file": file_path,
                        "total_tokens": 0,
                        "hit_tokens": 0,
                        "static_prefix_tokens": 0,
                        "cache_hit_ratio": 0.0,
                        "static_prefix_ratio": 0.0,
                        "group": get_system_group_key(json_data),
                        "error": error or "",
                    }
                )
                continue

            hit_len = trie.longest_prefix_length(tokens)
            trie.insert(tokens)

            hit_ratio = hit_len / total if total > 0 else 0
            static_ratio = static_len / total if total > 0 else 0

            results.append(
                {
                    "file": file_path,
                    "total_tokens": total,
                    "hit_tokens": hit_len,
                    "static_prefix_tokens": static_len,
                    "cache_hit_ratio": round(hit_ratio, 4),
                    "static_prefix_ratio": round(static_ratio, 4),
                    "group": get_system_group_key(json_data),
                    "error": "",
                }
            )
    return results


# ============================================================
# 文件扫描与加载
# ============================================================
def scan_jsonl_files(data_dir: str, max_files: Optional[int] = None) -> List[str]:
    """扫描目录下所有 jsonl/json 文件（含子目录）。"""
    path = Path(data_dir)
    files = []

    def _scan_dir(directory: Path):
        for f in sorted(directory.glob("*.jsonl")):
            files.append(str(f))
        for f in sorted(directory.glob("*.json")):
            files.append(str(f))

    # 根目录
    _scan_dir(path)

    # 子目录
    for sub_name in ("extracted_files", "request_params"):
        sub_dir = path / sub_name
        if sub_dir.exists():
            _scan_dir(sub_dir)

    if max_files:
        files = files[:max_files]

    logger.info(f"扫描到 {len(files)} 个 jsonl/json 文件")
    return files


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件，返回所有 JSON 对象的列表。"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [data]


# ============================================================
# 缓存命中率计算
# ============================================================
class FileRecord:
    """单条请求的渲染结果。"""

    __slots__ = (
        "file_path",
        "file_group_key",
        "tokens",
        "static_prefix_len",
        "total_tokens",
        "group_key",
        "error",
    )

    def __init__(
        self,
        file_path: str,
        file_group_key: str,
        tokens,
        static_prefix_len: int,
        total_tokens: int,
        group_key: str,
        error: Optional[str] = None,
    ):
        self.file_path = file_path
        self.file_group_key = file_group_key
        self.tokens = tokens
        self.static_prefix_len = static_prefix_len
        self.total_tokens = total_tokens
        self.group_key = group_key
        self.error = error


def _analyze_group(
    group_key: str,
    file_records: List[FileRecord],
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    处理单个文件前缀组的 Trie 匹配。

    Returns:
        (results, group_hit_tokens, group_static_tokens, group_total_tokens)
    """
    trie = PrefixTrie()
    results = []
    group_hit_tokens = 0
    group_static_tokens = 0
    group_total_tokens = 0
    hit_ratios = []

    # Shortest-First within group
    file_records.sort(key=lambda r: r.total_tokens)

    for rec in file_records:
        if rec.error or rec.total_tokens == 0:
            results.append(
                {
                    "file": rec.file_path,
                    "total_tokens": 0,
                    "hit_tokens": 0,
                    "static_prefix_tokens": 0,
                    "cache_hit_ratio": 0.0,
                    "static_prefix_ratio": 0.0,
                    "group": rec.group_key,
                    "error": rec.error or "",
                }
            )
            continue

        hit_len = trie.longest_prefix_length(rec.tokens)
        trie.insert(rec.tokens)

        hit_ratio = hit_len / rec.total_tokens if rec.total_tokens > 0 else 0
        static_ratio = (
            rec.static_prefix_len / rec.total_tokens if rec.total_tokens > 0 else 0
        )
        hit_ratios.append(hit_ratio)

        group_hit_tokens += hit_len
        group_static_tokens += rec.static_prefix_len
        group_total_tokens += rec.total_tokens

        results.append(
            {
                "file": rec.file_path,
                "total_tokens": rec.total_tokens,
                "hit_tokens": hit_len,
                "static_prefix_tokens": rec.static_prefix_len,
                "cache_hit_ratio": round(hit_ratio, 4),
                "static_prefix_ratio": round(static_ratio, 4),
                "group": rec.group_key,
                "error": "",
            }
        )

        if verbose and hit_ratio > 0:
            logger.debug(
                f"  [{group_key}] {Path(rec.file_path).name}: "
                f"total={rec.total_tokens}, hit={hit_len}, ratio={hit_ratio:.2%}"
            )

    return results, group_hit_tokens, group_static_tokens, group_total_tokens


def analyze(
    file_paths: List[str],
    tokenizer,
    verbose: bool = False,
    n_workers: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    主分析流程（多进程版本）。

    步骤：
    1. 按文件名前缀分组
    2. 每组独立渲染 token（多进程并行）
    3. 组内 Trie 匹配（Shortest-First）
    4. 汇总统计
    """
    # Step 1: 按文件名前缀分组
    logger.info("按文件名前缀分组中...")
    prefix_groups = group_files_by_prefix(file_paths)
    logger.info(f"共 {len(prefix_groups)} 个文件前缀组")

    # Step 2: 多进程渲染 token
    logger.info("启动多进程渲染 token...")
    if n_workers is None:
        n_workers = min(cpu_count(), len(prefix_groups))

    # 准备 worker 参数：(group_key, [file_paths], tokenizer_name, verbose)
    worker_args = [
        (group_key, sorted(file_list)) for group_key, file_list in prefix_groups.items()
    ]

    progress = ProgressTracker(len(prefix_groups), "渲染 tokens (多进程)")

    results = []
    total_hit_tokens = 0
    total_static_tokens = 0
    total_all_tokens = 0
    total_valid_requests = 0
    render_errors = 0
    all_records: List[FileRecord] = []

    if n_workers <= 1:
        # 单进程模式
        for group_key, file_paths_list in worker_args:
            for fp in file_paths_list:
                json_records = load_jsonl(fp)
                for json_data in json_records:
                    tokens, static_len, error = render_tokens(json_data, tokenizer)
                    group_key_for_rec = get_system_group_key(json_data)
                    all_records.append(
                        FileRecord(
                            file_path=fp,
                            file_group_key=group_key,
                            tokens=tokens,
                            static_prefix_len=static_len,
                            total_tokens=len(tokens),
                            group_key=group_key_for_rec,
                            error=error,
                        )
                    )
            progress.update()

        # Step 3: 按 system group key 分组进行 Trie 匹配
        logger.info("Trie 匹配中（按 system group key 分组）...")
        file_group_map: Dict[str, List[FileRecord]] = defaultdict(list)
        for rec in all_records:
            if not rec.error:
                file_group_map[rec.group_key].append(rec)

        for group_key in sorted(file_group_map.keys()):
            group_records = file_group_map[group_key]
            gresults, gh, gs, gt = _analyze_group(group_key, group_records, verbose)
            results.extend(gresults)
            total_hit_tokens += gh
            total_static_tokens += gs
            total_all_tokens += gt
            total_valid_requests += len(group_records)

        for rec in all_records:
            if rec.error:
                results.append(
                    {
                        "file": rec.file_path,
                        "total_tokens": 0,
                        "hit_tokens": 0,
                        "static_prefix_tokens": 0,
                        "cache_hit_ratio": 0.0,
                        "static_prefix_ratio": 0.0,
                        "group": rec.group_key,
                        "error": rec.error,
                    }
                )
                render_errors += 1
    else:
        # 多进程模式：每个 worker 独立完成渲染 + Trie 匹配
        with Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(tokenizer.name_or_path, verbose),
        ) as pool:
            for group_results in pool.imap_unordered(
                _process_group_worker, worker_args
            ):
                for r in group_results:
                    if r["error"]:
                        render_errors += 1
                    if r["total_tokens"] > 0:
                        total_hit_tokens += r["hit_tokens"]
                        total_static_tokens += r["static_prefix_tokens"]
                        total_all_tokens += r["total_tokens"]
                        total_valid_requests += 1
                    results.append(r)
                progress.update()

    # 汇总
    overall_hit_rate = (
        total_hit_tokens / total_all_tokens if total_all_tokens > 0 else 0
    )
    avg_static_ratio = (
        total_static_tokens / total_all_tokens if total_all_tokens > 0 else 0
    )

    hit_ratios = [float(r["cache_hit_ratio"]) for r in results if r["total_tokens"] > 0]
    threshold = 0.9
    above_threshold = sum(1 for r in hit_ratios if r >= threshold)

    summary = {
        "total_files": len(file_paths),
        "total_valid_requests": total_valid_requests,
        "render_errors": render_errors,
        "total_all_tokens": total_all_tokens,
        "total_hit_tokens": total_hit_tokens,
        "overall_cache_hit_rate": round(overall_hit_rate, 4),
        "avg_static_prefix_ratio": round(avg_static_ratio, 4),
        "threshold_90pct": threshold,
        "requests_above_threshold": above_threshold,
        "requests_above_threshold_pct": round(
            above_threshold / len(hit_ratios) * 100 if hit_ratios else 0, 2
        ),
        "unique_groups": len(prefix_groups),
        "total_groups": len(prefix_groups),
    }

    return summary, results


# ============================================================
# 输出
# ============================================================
def print_summary(summary: Dict[str, Any]):
    """打印汇总报告。"""
    print()
    print("=" * 60)
    print("  缓存命中率分析报告")
    print("=" * 60)
    print(f"  文件总数:            {summary['total_files']:>10,}")
    print(f"  有效请求数:          {summary['total_valid_requests']:>10,}")
    print(f"  渲染失败:            {summary['render_errors']:>10,}")
    print(f"  唯一 group 数:       {summary['unique_groups']:>10,}")
    print("-" * 60)
    print(f"  总 tokens:       {summary['total_all_tokens']:>12,}")
    print(f"  命中 tokens:         {summary['total_hit_tokens']:>12,}")
    print(f"  重复率:          {summary['overall_cache_hit_rate']:>11.2%}")
    print(f"  Static Prefix 占比:  {summary['avg_static_prefix_ratio']:>11.2%}")
    print("-" * 60)
    print(
        f"  满足 {summary['threshold_90pct']:.0%} 阈值的请求:  "
        f"{summary['requests_above_threshold']:>6,} / "
        f"{summary['total_valid_requests']:>6,}  "
        f"({summary['requests_above_threshold_pct']:.1f}%)"
    )
    print("=" * 60)
    print()


def save_csv(results: List[Dict[str, Any]], output_path: str):
    """保存详细结果到 CSV。"""
    fieldnames = [
        "file",
        "total_tokens",
        "hit_tokens",
        "static_prefix_tokens",
        "cache_hit_ratio",
        "static_prefix_ratio",
        "group",
        "error",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"详细结果已保存到: {output_path}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="分析 LLM Prompt Caching 缓存命中率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default=".",
        help="数据目录（默认: 当前目录）",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        default="zai-org/GLM-5.1",
        help="Tokenizer 模型名称（默认: zai-org/GLM-5.1）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="cache_analysis_result.csv",
        help="输出 CSV 文件路径（默认: cache_analysis_result.csv）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="最多处理的文件数（默认: 全部）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="多进程渲染的 worker 数量（默认: CPU 核心数）",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start = time.time()

    # 1. 加载 tokenizer
    tokenizer = setup_tokenizer(args.tokenizer)

    # 2. 扫描文件
    files = scan_jsonl_files(args.data_dir, max_files=args.max_files)
    if not files:
        logger.error("未找到 jsonl 文件")
        sys.exit(1)

    # 3. 分析
    summary, results = analyze(
        files, tokenizer, verbose=args.verbose, n_workers=args.workers
    )

    # 4. 输出
    print_summary(summary)
    save_csv(results, args.output)

    elapsed = time.time() - start
    logger.info(f"耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
