#!/usr/bin/env python3
#  Copyright (C) 2025. Hao Zheng
#  All rights reserved.

"""Prepare the vendored whisper.cpp CLI and default model files."""

from __future__ import annotations

import argparse
from pathlib import Path

from openlrc.setup.whisper_cpp import DEFAULT_MODEL, DEFAULT_VAD_MODEL, default_model_dir, setup_whisper_cpp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vendored whisper.cpp and download default OpenLRC models.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"whisper.cpp model name, default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--vad-model", default=DEFAULT_VAD_MODEL, help=f"whisper.cpp VAD model name, default: {DEFAULT_VAD_MODEL}"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=default_model_dir(), help="Directory for downloaded model files."
    )
    parser.add_argument("--skip-build", action="store_true", help="Initialize the submodule but skip CMake build.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = setup_whisper_cpp(
        model=args.model,
        vad_model=args.vad_model,
        model_dir=args.model_dir,
        skip_build=args.skip_build,
        skip_models=args.skip_models,
    )

    print("\nwhisper.cpp setup complete.")
    if result.cli_path:
        print(f"CLI: {result.cli_path}")
    if result.whisper_model:
        print(f"Whisper model: {result.whisper_model}")
    if result.vad_model:
        print(f"VAD model: {result.vad_model}")
    print("\nRuntime overrides:")
    print("  OPENLRC_WHISPER_CLI=/path/to/whisper-cli")
    print("  OPENLRC_WHISPER_MODEL=/path/to/ggml-base.bin")
    print("  OPENLRC_WHISPER_VAD_MODEL=/path/to/ggml-silero-v6.2.0.bin")
    print("  OPENLRC_WHISPER_MODEL_DIR=/path/to/models")


if __name__ == "__main__":
    main()
