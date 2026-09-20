#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_INSTRUCTION = """Rank the candidate file path by how important it is for an AI coding agent trying to understand the repository's core architecture and behavior.

You are given the repository's complete tracked file tree as context, but no file contents. Judge only from path names and the surrounding repository structure.

Prefer paths that are likely to contain:
- main entry points
- core business or domain logic
- central architecture
- important interfaces and abstractions
- major configuration that defines application behavior

Deprioritize paths that are likely to be:
- generated artifacts
- vendored dependencies
- lock files
- trivial utilities
- examples
- tests, unless they appear central to understanding behavior
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank tracked file paths in a git repository with Qwen3-Reranker."
    )
    parser.add_argument("repo", type=Path, help="Path to a git repository")
    parser.add_argument("--top", type=int, default=20, help="Number of results to print")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only rank candidate paths matching this glob. Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Skip candidate paths matching this glob. Repeatable.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Write full ranking as JSON")
    return parser.parse_args()


def git_files(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return sorted(item.decode("utf-8") for item in proc.stdout.split(b"\0") if item)


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()

    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit(f"not a git repository: {repo}")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    all_paths = git_files(repo)
    candidates = [
        path
        for path in all_paths
        if (not args.include or matches(path, args.include))
        and not (args.exclude and matches(path, args.exclude))
    ]

    if not candidates:
        raise SystemExit("no candidate paths matched the requested filters")

    tree = "\n".join(f"- {path}" for path in all_paths)
    query = f"""Repository: {repo.name}

Goal:
Understand what this repository does, how it is structured, and where its core behavior is implemented.

Complete tracked file tree:
{tree}
"""

    print(f"Repository context: {len(all_paths)} tracked paths", flush=True)
    print(f"Candidate paths: {len(candidates)}", flush=True)
    print(f"Loading {args.model} on {args.device} ...", flush=True)

    model = CrossEncoder(
        args.model,
        prompts={"repo-path-importance": DEFAULT_INSTRUCTION.strip()},
        default_prompt_name="repo-path-importance",
        max_length=args.max_length,
        device=args.device,
    )

    print("Ranking paths only; file contents are never read.", flush=True)
    scores = model.predict(
        [(query, f"Candidate path: {path}") for path in candidates],
        batch_size=args.batch_size,
        show_progress_bar=True,
        activation_fn=torch.nn.Sigmoid(),
    )
    scores = np.asarray(scores).reshape(-1)

    results = [
        {"path": path, "score": float(score)}
        for score, path in zip(scores, candidates, strict=True)
    ]
    results.sort(key=lambda item: item["score"], reverse=True)

    print()
    for item in results[: args.top]:
        print(f'{item["score"]:.6f}  {item["path"]}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "repository": str(repo),
            "mode": "path-only",
            "instruction": DEFAULT_INSTRUCTION.strip(),
            "max_length": args.max_length,
            "repository_path_count": len(all_paths),
            "candidate_count": len(results),
            "query": query,
            "results": results,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
