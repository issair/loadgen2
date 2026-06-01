#!/usr/bin/env python3
"""MLPerf inference runner for GLM-5.1 with OpenAI backend."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import mlperf_loadgen as lg
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mlperf import (
    OfflineSUT,
    QuerySampleLibrary,
    ServerSUT,
)
from utils import (
    create_base_argument_parser,
    generate_timestamped_filename,
    get_backend_instance,
    handle_runner_error,
    setup_output_paths,
    validate_runner_args,
    validate_runner_for_backend,
)
from utils.data_utils import DiskCacheRequestStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = create_base_argument_parser(
        "Run MLPerf inference benchmarks for GLM-5.1 with OpenAI backend"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "server", "interactive"],
        help="MLPerf mode",
    )
    parser.add_argument(
        "--mlperf-conf",
        type=str,
        default="/inference/mlperf.conf",
        help="Path to MLPerf configuration file",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="mlperf/user.conf",
        help="Path to user configuration file",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy mode instead of performance",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mlperf_results",
        help="Directory for MLPerf output logs",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="Directory for detailed logs"
    )
    parser.add_argument(
        "--max-vu",
        type=int,
        default=None,
        help="Maximum number of VUs (truncates sample_pool)",
    )
    return parser


def configure_loadgen(
    mode, accuracy_mode, mlperf_conf=None, user_conf=None, model_name="glm-5.1"
):
    settings = lg.TestSettings()
    if mode == "offline":
        settings.scenario = lg.TestScenario.Offline
        config_scenario = "Offline"
    elif mode in ("server", "interactive"):
        settings.scenario = lg.TestScenario.Server
        config_scenario = "Server"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "interactive":
        model_name = f"{model_name}-interactive"
    if accuracy_mode:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
    if mlperf_conf and Path(mlperf_conf).exists():
        settings.FromConfig(mlperf_conf, model_name, config_scenario, 2)
    if user_conf and Path(user_conf).exists():
        settings.FromConfig(user_conf, model_name, config_scenario, 1)
    return settings


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    try:
        validate_runner_args(args, "mlperf")
        backend_name = validate_runner_for_backend("mlperf")
        mode = args.mode

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = (
            Path(args.log_dir)
            if args.log_dir
            else (output_dir / mode / ("accuracy" if args.accuracy else "performance"))
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        _, output_file_base = setup_output_paths(args, mode=mode)
        if args.output_file is None:
            mode_str = "accuracy" if args.accuracy else "performance"
            output_file_base = str(
                log_dir / f"{backend_name}_mlperf_{mode}_{mode_str}_output.pkl"
            )

        actual_output_file = generate_timestamped_filename(
            output_file_base, add_timestamp=True
        )
        Path(actual_output_file).parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("MLPerf Inference Benchmark Runner - GLM-5.1")
        logger.info("=" * 80)
        logger.info(f"Backend: {backend_name}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Accuracy: {args.accuracy}")
        logger.info(f"Input file: {args.input_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output file: {actual_output_file}")
        logger.info("=" * 80)

        # ----------------------------------------------------------------
        #  Load preprocessed conversation data (output of split_conversation.py)
        # ----------------------------------------------------------------
        logger.info("Loading preprocessed conversation data...")
        with open(args.input_file, "r", encoding="utf-8") as f:
            preprocessed = json.load(f)

        cache_path: str = preprocessed["llm_requests_cache"]
        llm_requests_count: int = preprocessed["llm_requests_count"]
        sample_pool: List[int] = preprocessed["sample_pool"]
        if args.max_vu is not None:
            sample_pool = sample_pool[: args.max_vu]
            logger.info(f"Truncated sample_pool to {args.max_vu} entries (--max-vu)")
        dependencies: List[List[int]] = preprocessed["dependencies"]

        # Load requests lazily from diskcache (each request fetched on demand)
        llm_requests = DiskCacheRequestStore(cache_path)

        logger.info(
            f"Loaded: {llm_requests_count} llm_requests (lazy via diskcache), "
            f"{len(sample_pool)} sample_pool entries, "
            f"cache path: {cache_path}"
        )

        logger.info(f"Initializing {backend_name} backend...")
        backend = get_backend_instance(backend_name)

        with backend:
            # Build QSL with dependency chain support
            qsl = QuerySampleLibrary(
                llm_requests=llm_requests,
                dependencies=dependencies,
                sample_pool=sample_pool,
            )

            # Build SUT with dependency replay
            sut_cls = OfflineSUT if args.mode == "offline" else ServerSUT
            sut = sut_cls(
                backend=backend,
                llm_requests=llm_requests,
                sample_pool=sample_pool,
                dependencies=dependencies,
                name=f"{backend_name}_{args.mode}_sut",
            )

            settings = configure_loadgen(
                mode=args.mode,
                accuracy_mode=args.accuracy,
                mlperf_conf=args.mlperf_conf,
                user_conf=args.user_conf,
            )
            # Adjust query counts based on sample_pool size
            pool_size = len(sample_pool)
            if args.mode == "offline":
                settings.max_query_count = pool_size
                settings.min_query_count = pool_size
            settings.use_token_latencies = True
            settings.enable_dependency_replay = True
            # server_coalesce_queries must be OFF with dependency replay:
            # coalescing moves samples between queries which breaks the
            # dependency chain expansion and causes LoadGen to wait forever.
            settings.server_coalesce_queries = False

            log_settings = lg.LogSettings()
            log_settings.log_output.outdir = str(log_dir)
            log_settings.log_output.copy_summary_to_stdout = False
            log_settings.enable_trace = False

            logger.info("Starting SUT...")
            sut.start()
            try:
                lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
                sut_results = sut.get_results()
                logger.info(f"Retrieved {len(sut_results)} dependency results from SUT")
            finally:
                sut.stop()
                if qsl.qsl is not None:
                    lg.DestroyQSL(qsl.qsl)
                    qsl.qsl = None

            if sut_results:
                # Build a DataFrame from all dependency results
                import pandas as pd

                df_output = pd.DataFrame(sut_results)

                # Add ratio columns for _print_token_usage_stats
                df_output["cached_ratio"] = (
                    df_output["cached_tokens"]
                    / df_output["prompt_tokens"].replace(0, 1)
                    * 100
                )
                df_output["reasoning_ratio"] = (
                    df_output["reasoning_tokens"]
                    / df_output["completion_tokens"].replace(0, 1)
                    * 100
                )

                # Print token usage statistics
                _print_token_usage_stats(df_output, log_dir)

                if args.accuracy:
                    mlperf_log_file = log_dir / "mlperf_log_accuracy.json"
                    if mlperf_log_file.exists():
                        from eval_accuracy import process_mlperf_log_accuracy

                        checkpoint_path = backend.config.get(
                            "tokenizer", backend.config.get("model", "THUDM/glm-5-1")
                        )
                        process_mlperf_log_accuracy(
                            mlperf_log_file=mlperf_log_file,
                            dataset_file=args.input_file,
                            checkpoint_path=checkpoint_path,
                            output_dir=log_dir,
                        )
                    else:
                        from eval_accuracy import (
                            evaluate_dataframe,
                            print_evaluation_results,
                        )

                        df_eval = evaluate_dataframe(df_output)
                        print_evaluation_results(df_eval)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        handle_runner_error(e, "run_mlperf.py")


def _print_token_usage_stats(df: pd.DataFrame, log_dir: Path) -> None:
    """Print token usage statistics and LoadGen performance summary.

    Args:
        df: Output dataframe with token usage columns.
        log_dir: Directory containing loadgen log files.
    """
    import re

    # ── LoadGen Performance Summary ──────────────────────────────────
    summary_file = log_dir / "mlperf_log_summary.txt"
    if summary_file.exists():
        logger.info("")
        logger.info("=" * 64)
        logger.info("LoadGen Performance Summary")
        logger.info("=" * 64)
        with open(summary_file, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                # Convert nanosecond values to milliseconds
                if "ns" in line:
                    # Match the number after colon (the time value in ns)
                    converted = re.sub(
                        r":\s*(\d+(?:\.\d+)?)",
                        lambda m: f": {float(m.group(1)) / 1e6:.3f}",
                        line,
                    )
                    converted = converted.replace("ns", "ms")
                else:
                    converted = line
                logger.info(converted)
        logger.info("=" * 64)

    # Build a stats helper
    def _stats(series):
        vals = series.dropna()
        if len(vals) == 0:
            return {
                "max": 0,
                "min": 0,
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p97": 0.0,
                "p99": 0.0,
                "p999": 0.0,
            }
        return {
            "max": int(vals.max()),
            "min": int(vals.min()),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
            "p95": float(vals.quantile(0.95)),
            "p97": float(vals.quantile(0.97)),
            "p99": float(vals.quantile(0.99)),
            "p999": float(vals.quantile(0.999)),
        }

    def _ratio_stats(series):
        vals = series.dropna()
        if len(vals) == 0:
            return {
                "max": 0.0,
                "min": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p97": 0.0,
                "p99": 0.0,
                "p999": 0.0,
            }
        return {
            "max": float(vals.max()),
            "min": float(vals.min()),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
            "p95": float(vals.quantile(0.95)),
            "p97": float(vals.quantile(0.97)),
            "p99": float(vals.quantile(0.99)),
            "p999": float(vals.quantile(0.999)),
        }

    logger.info("")
    logger.info("=" * 64)
    logger.info("Token Usage Statistics (SUT-reported)")
    logger.info("=" * 64)

    # 1. prompt_tokens
    s = _stats(df["prompt_tokens"])
    logger.info("Min Prompt tokens                     : {:>10.0f}".format(s["min"]))
    logger.info("Max Prompt tokens                     : {:>10.0f}".format(s["max"]))
    logger.info("Mean Prompt tokens                    : {:>10.1f}".format(s["mean"]))
    logger.info("Median Prompt tokens                  : {:>10.1f}".format(s["median"]))
    logger.info(
        "Percentiles Prompt tokens (P50/P75/P90/P95/P97/P99/P999): {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f}".format(
            s["p50"], s["p75"], s["p90"], s["p95"], s["p97"], s["p99"], s["p999"]
        )
    )

    # 2. cached_tokens
    s = _stats(df["cached_tokens"])
    logger.info("Min Cached tokens                     : {:>10.0f}".format(s["min"]))
    logger.info("Max Cached tokens                     : {:>10.0f}".format(s["max"]))
    logger.info("Mean Cached tokens                    : {:>10.1f}".format(s["mean"]))
    logger.info("Median Cached tokens                  : {:>10.1f}".format(s["median"]))
    logger.info(
        "Percentiles Cached tokens (P50/P75/P90/P95/P97/P99/P999): {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f}".format(
            s["p50"], s["p75"], s["p90"], s["p95"], s["p97"], s["p99"], s["p999"]
        )
    )

    # 3. cached_ratio (percentage)
    s = _ratio_stats(df["cached_ratio"])
    logger.info("Min Cache hit ratio (%)               : {:>7.2f}%".format(s["min"]))
    logger.info("Max Cache hit ratio (%)               : {:>7.2f}%".format(s["max"]))
    logger.info("Mean Cache hit ratio (%)              : {:>7.2f}%".format(s["mean"]))
    logger.info("Median Cache hit ratio (%)            : {:>7.2f}%".format(s["median"]))
    logger.info(
        "Percentiles Cache hit ratio (P50/P75/P90/P95/P97/P99/P999): {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}%".format(
            s["p50"], s["p75"], s["p90"], s["p95"], s["p97"], s["p99"], s["p999"]
        )
    )

    # 4. completion_tokens
    s = _stats(df["completion_tokens"])
    logger.info("Min Completion tokens                 : {:>10.0f}".format(s["min"]))
    logger.info("Max Completion tokens                 : {:>10.0f}".format(s["max"]))
    logger.info("Mean Completion tokens                : {:>10.1f}".format(s["mean"]))
    logger.info("Median Completion tokens              : {:>10.1f}".format(s["median"]))
    logger.info(
        "Percentiles Completion tokens (P50/P75/P90/P95/P97/P99/P999): {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f}".format(
            s["p50"], s["p75"], s["p90"], s["p95"], s["p97"], s["p99"], s["p999"]
        )
    )

    # 5. reasoning_tokens
    s = _stats(df["reasoning_tokens"])
    logger.info("Min Reasoning tokens                  : {:>10.0f}".format(s["min"]))
    logger.info("Max Reasoning tokens                  : {:>10.0f}".format(s["max"]))
    logger.info("Mean Reasoning tokens                 : {:>10.1f}".format(s["mean"]))
    logger.info("Median Reasoning tokens               : {:>10.1f}".format(s["median"]))
    logger.info(
        "Percentiles Reasoning tokens (P50/P75/P90/P95/P97/P99/P999): {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f} / {:>10.1f}".format(
            s["p50"], s["p75"], s["p90"], s["p95"], s["p97"], s["p99"], s["p999"]
        )
    )

    # 6. reasoning_ratio (percentage of output that is reasoning)
    s = _ratio_stats(df["reasoning_ratio"])
    logger.info("Min Reasoning ratio (%)               : {:>7.2f}%".format(s["min"]))
    logger.info("Max Reasoning ratio (%)               : {:>7.2f}%".format(s["max"]))
    logger.info("Mean Reasoning ratio (%)              : {:>7.2f}%".format(s["mean"]))
    logger.info("Median Reasoning ratio (%)            : {:>7.2f}%".format(s["median"]))
    logger.info(
        "Percentiles Reasoning ratio (P50/P75/P90/P95/P97/P99/P999): {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}% / {:>7.2f}%".format(
            s["p50"], s["p75"], s["p90"], s["p95"], s["p97"], s["p99"], s["p999"]
        )
    )

    logger.info("=" * 64)


if __name__ == "__main__":
    main()
