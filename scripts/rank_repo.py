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
DEFAULT_INSTRUCTION = """Judge whether the document is one of the most important files for an AI coding agent trying to understand the repository's core architecture and behavior.

Prefer:
- main entry points
- core business or domain logic
- central architecture
- important interfaces and abstractions
- major configuration that defines application behavior

Deprioritize:
- generated files
- vendored dependencies
- lock files
- trivial utilities
- examples
- tests, unless they are essential for understanding behavior
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank tracked files in a git repository with Qwen3-Reranker."
    )
    parser.add_argument("repo", type=Path, help="Path to a git repository")
    parser.add_argument("--top", type=int, default=20, help="Number of results to print")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only rank paths matching this glob. Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Skip paths matching this glob. Repeatable.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum characters retained from each text file",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=512 * 1024,
        help="Skip files larger than this before decoding",
    )
    parser.add_argument("--batch-size", type=int, default=2)
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
    return [item.decode("utf-8") for item in proc.stdout.split(b"\0") if item]


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def read_text(path: Path, max_file_bytes: int, max_chars: int) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if len(data) > max_file_bytes or b"\0" in data[:8192]:
        return None

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    if not text.strip():
        return None

    if len(text) <= max_chars:
        return text

    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n\n...[truncated]...\n\n" + text[-tail:]


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()

    if not (repo / ".git").exists():
        raise SystemExit(f"not a git repository: {repo}")

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    candidates: list[tuple[str, str]] = []
    for relative in git_files(repo):
        if args.include and not matches(relative, args.include):
            continue
        if args.exclude and matches(relative, args.exclude):
            continue

        text = read_text(repo / relative, args.max_file_bytes, args.max_chars)
        if text is None:
            continue

        document = f"Path: {relative}\n\n{text}"
        candidates.append((relative, document))

    if not candidates:
        raise SystemExit("no rankable text files matched the requested paths")

    query = (
        f"Repository: {repo.name}\n\n"
        "Goal: understand what this repository does, how it is structured, "
        "and where its core behavior is implemented."
    )

    print(f"Loading {args.model} on {args.device} ...", flush=True)
    model = CrossEncoder(
        args.model,
        prompts={"repo-importance": DEFAULT_INSTRUCTION.strip()},
        default_prompt_name="repo-importance",
        max_length=args.max_length,
        device=args.device,
    )

    print(f"Ranking {len(candidates)} files ...", flush=True)
    scores = model.predict(
        [(query, document) for _, document in candidates],
        batch_size=args.batch_size,
        show_progress_bar=True,
        activation_fn=torch.nn.Sigmoid(),
    )
    scores = np.asarray(scores).reshape(-1)

    results = [
        {"path": path, "score": float(score)}
        for score, (path, _) in zip(scores, candidates, strict=True)
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
            "instruction": DEFAULT_INSTRUCTION.strip(),
            "query": query,
            "candidate_count": len(results),
            "results": results,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
