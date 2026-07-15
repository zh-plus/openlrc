#!/usr/bin/env python3
#  Copyright (C) 2026. Hao Zheng
#  All rights reserved.

"""Prepare the vendored llama.cpp server and default local LLM model."""

from __future__ import annotations

import argparse
from pathlib import Path

from openlrc.llama_resources import (
    DEFAULT_LLAMA_MODEL_FILE,
    DEFAULT_LLAMA_MODEL_REPO,
    get_local_llm_profile,
    user_llm_model_dir,
)
from openlrc.setup.llama_cpp import setup_llama_cpp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vendored llama.cpp and download the default local LLM model.")
    parser.add_argument(
        "--local-model-profile", default=None, help="Registered local LLM profile to download, e.g. hy-mt2-7b."
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_LLAMA_MODEL_REPO,
        help=f"Hugging Face GGUF model repo, default: {DEFAULT_LLAMA_MODEL_REPO}",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_LLAMA_MODEL_FILE,
        help=f"GGUF model filename, default: {DEFAULT_LLAMA_MODEL_FILE}",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision to download, default: main")
    parser.add_argument("--model-url", default=None, help="Direct GGUF download URL. Overrides --model-repo.")
    parser.add_argument(
        "--model-dir", type=Path, default=user_llm_model_dir(), help="Directory for downloaded GGUF model files."
    )
    parser.add_argument("--skip-build", action="store_true", help="Initialize the submodule but skip CMake build.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads.")
    parser.add_argument("--force", action="store_true", help="Re-download the model even if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_model_profile and not args.skip_models:
        profile = get_local_llm_profile(args.local_model_profile)
        if not profile.model_repo or not profile.model_file:
            raise SystemExit(
                f"{profile.name} has no downloadable default GGUF. Place a converted model locally instead."
            )
        if args.model_repo == DEFAULT_LLAMA_MODEL_REPO:
            args.model_repo = profile.model_repo
        if args.model_file == DEFAULT_LLAMA_MODEL_FILE:
            args.model_file = profile.model_file

    result = setup_llama_cpp(
        model_repo=args.model_repo,
        model_file=args.model_file,
        revision=args.revision,
        model_url=args.model_url,
        model_dir=args.model_dir,
        skip_build=args.skip_build,
        skip_models=args.skip_models,
        force=args.force,
    )

    print("\nllama.cpp setup complete.")
    if result.server_path:
        print(f"Server: {result.server_path}")
    if result.cli_path:
        print(f"CLI: {result.cli_path}")
    if result.model_path:
        print(f"LLM model: {result.model_path}")
    print("\nRuntime overrides:")
    print("  OPENLRC_LLAMA_SERVER=/path/to/llama-server")
    print("  OPENLRC_LLAMA_CLI=/path/to/llama-cli")
    print("  OPENLRC_LLAMA_MODEL=/path/to/Qwen3.5-9B-Q4_K_M.gguf")
    print("  OPENLRC_LLAMA_MODEL_DIR=/path/to/llm-models")


if __name__ == "__main__":
    main()
