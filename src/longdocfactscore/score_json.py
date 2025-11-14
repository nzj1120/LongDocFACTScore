"""Utility script to score pairs of JSON files with LongDocFACTScore.

The script expects two JSON files that share the same set of keys.
Each value should be a string containing the source document or its
corresponding summary. The script outputs per-section scores, an overall
average score, a progress bar (unless disabled) and the average
character counts of the source documents and summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .ldfacts import LongDocFACTScore


@dataclass
class ScoreResult:
    """Container for JSON scoring results."""

    scores: Dict[str, float]
    average_source_length: float
    average_summary_length: float


def _load_json(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"All keys must be strings, but key {key!r} is {type(key).__name__}")
        if not isinstance(value, str):
            raise ValueError(
                f"All values must be strings, but value for key {key!r} is {type(value).__name__}"
            )
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score summaries stored in JSON files.")
    parser.add_argument("source_json", type=Path, help="Path to the source/original JSON file")
    parser.add_argument("summary_json", type=Path, help="Path to the summary JSON file")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run on, e.g. 'cuda:0'. Defaults to automatic selection.",
    )
    parser.add_argument(
        "--sent-model",
        dest="sent_model",
        default="uer/sbert-base-chinese-nli",
        help="SentenceTransformer model path or name for retrieval.",
    )
    parser.add_argument(
        "--bart-model",
        dest="bart_model",
        default="fnlp/bart-large-chinese",
        help="BART model path or name for BARTScore computation.",
    )
    parser.add_argument(
        "--bart-tokenizer",
        dest="bart_tokenizer",
        default=None,
        help="Optional tokenizer path/name for the BART model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the per-section scores as JSON.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar during scoring.",
    )
    return parser


def score_json(
    source_json: Path,
    summary_json: Path,
    *,
    device: Optional[str] = None,
    sent_model: str = "uer/sbert-base-chinese-nli",
    bart_model: str = "fnlp/bart-large-chinese",
    bart_tokenizer: Optional[str] = None,
    progress: bool = False,
) -> ScoreResult:
    """Load JSON files and return per-section scores and length statistics."""

    src_data = _load_json(source_json)
    hyp_data = _load_json(summary_json)

    if set(src_data.keys()) != set(hyp_data.keys()):
        missing_in_summary = sorted(set(src_data.keys()) - set(hyp_data.keys()))
        missing_in_source = sorted(set(hyp_data.keys()) - set(src_data.keys()))
        error_parts: List[str] = []
        if missing_in_summary:
            error_parts.append(f"missing in summary JSON: {missing_in_summary}")
        if missing_in_source:
            error_parts.append(f"missing in source JSON: {missing_in_source}")
        raise ValueError(
            "Source and summary JSON files must have identical keys; " + "; ".join(error_parts)
        )

    sections = list(src_data.keys())
    src_docs = [src_data[section] for section in sections]
    hyp_docs = [hyp_data[section] for section in sections]

    src_lengths = [len(doc) for doc in src_docs]
    hyp_lengths = [len(doc) for doc in hyp_docs]

    scorer = LongDocFACTScore(
        device=device,
        sent_model_name_or_path=sent_model,
        bart_model_name_or_path=bart_model,
        bart_tokenizer_name_or_path=bart_tokenizer,
    )
    scores = scorer.score_src_hyp_long(src_docs, hyp_docs, progress=progress)

    average_source_length = sum(src_lengths) / len(src_lengths) if src_lengths else float("nan")
    average_summary_length = sum(hyp_lengths) / len(hyp_lengths) if hyp_lengths else float("nan")

    return ScoreResult(
        scores=dict(zip(sections, scores)),
        average_source_length=average_source_length,
        average_summary_length=average_summary_length,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = score_json(
        args.source_json,
        args.summary_json,
        device=args.device,
        sent_model=args.sent_model,
        bart_model=args.bart_model,
        bart_tokenizer=args.bart_tokenizer,
        progress=not args.no_progress,
    )

    for section, score in result.scores.items():
        print(f"{section}: {score:.4f}")

    average_score = (
        sum(result.scores.values()) / len(result.scores) if result.scores else float("nan")
    )
    print(f"Average score: {average_score:.4f}")
    print(f"原文平均字数: {result.average_source_length:.1f}")
    print(f"缩写平均字数: {result.average_summary_length:.1f}")

    if args.output:
        output_payload = {
            "scores": result.scores,
            "average_score": average_score,
            "average_source_length": result.average_source_length,
            "average_summary_length": result.average_summary_length,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

