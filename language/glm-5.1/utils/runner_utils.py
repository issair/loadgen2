"""
Common utilities for all runners.
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


def create_base_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create argument parser with common arguments."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/final_output.pkl",
        help="Input pickle file with prompts",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output pickle file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process from dataset",
    )
    parser.add_argument(
        "--skip-samples",
        type=int,
        default=0,
        help="Number of samples to skip from the beginning",
    )
    return parser


def print_runner_header(
    runner_name: str,
    backend_name: Optional[str] = None,
    args: argparse.Namespace = None,
) -> None:
    """Print standardized header for runners."""
    if backend_name is None:
        from utils.backend_registry import detect_backend

        backend_name = detect_backend()

    print("=" * 80)
    print(f"{runner_name}")
    print("=" * 80)
    print(f"Backend: {backend_name}")
    if args:
        print(f"Input file: {args.input_file}")
        if hasattr(args, "output_file") and args.output_file:
            print(f"Output file: {args.output_file}")
        if hasattr(args, "num_samples") and args.num_samples is not None:
            print(f"Number of samples: {args.num_samples}")
        if hasattr(args, "skip_samples") and args.skip_samples > 0:
            print(f"Skipping samples: {args.skip_samples}")
    print("=" * 80)


def setup_output_paths(
    args: argparse.Namespace,
    backend_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> Tuple[Path, str]:
    """Set up output directories and file paths."""
    if backend_name is None:
        from utils.backend_registry import detect_backend

        backend_name = detect_backend()

    if hasattr(args, "output_dir") and args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if mode:
            output_dir = Path(f"outputs/{backend_name}/{mode}")
        else:
            output_dir = Path(f"outputs/{backend_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if hasattr(args, "num_samples") and args.num_samples:
            suffix = f"_{args.num_samples}samples"
        else:
            suffix = "_full"
        if mode:
            output_file = str(
                output_dir / f"{backend_name}_{mode}_output_{timestamp}{suffix}.pkl"
            )
        else:
            output_file = str(
                output_dir / f"{backend_name}_output_{timestamp}{suffix}.pkl"
            )

    return output_dir, output_file
