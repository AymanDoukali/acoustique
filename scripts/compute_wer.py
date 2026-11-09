"""Calcule le WER des transcriptions par rapport aux références de metadata.json.

Les références sont reconstruites depuis timeline_segments : les segments sont
triés chronologiquement (start_time_sec), puis leurs champs text sont concaténés.
Chaque fichier *.txt sous hypothesis-root/case_*/sim_*/ est évalué séparément.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from jiwer import process_words

logger = logging.getLogger("compute_wer")


def normalize_text(text: str) -> str:
    """Minuscules, ponctuation remplacée par des espaces, espaces normalisés."""
    text = text.lower()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def load_reference(metadata_path: Path) -> str:
    """Reconstruit le texte de référence à partir de metadata.json."""
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)

    segments = metadata.get("timeline_segments")
    if not isinstance(segments, list):
        raise ValueError(f"Champ timeline_segments absent ou invalide : {metadata_path}")

    # Le tri est stable : en cas d'égalité, l'ordre présent dans le JSON est conservé.
    segments = sorted(
        segments,
        key=lambda item: float(item.get("start_time_sec", 0.0)) if isinstance(item, dict) else 0.0,
    )
    texts = [
        str(segment.get("text", "")).strip()
        for segment in segments
        if isinstance(segment, dict) and segment.get("text")
        and str(segment.get("text")).strip()
    ]
    return " ".join(texts)


def score_transcript(reference: str, hypothesis: str) -> dict:
    """Retourne les comptes d'erreurs et le WER pour une paire de textes."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    ref_words = len(ref.split())
    hyp_words = len(hyp.split())

    # jiwer fournit les alignements et les comptes S/D/I.
    result = process_words(ref, hyp)
    wer = float(result.wer) if ref_words else math.nan
    return {
        "reference_words": ref_words,
        "hypothesis_words": hyp_words,
        "substitutions": int(result.substitutions),
        "deletions": int(result.deletions),
        "insertions": int(result.insertions),
        "wer": wer,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Écrit un CSV, en créant le dossier comme evaluate.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("Aucun résultat à écrire dans %s", path)
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: round(value, 4) if isinstance(value, float) and math.isfinite(value)
                else ("" if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in row.items()
            })


def corpus_summary(rows: list[dict]) -> list[dict]:
    """Agrège S/D/I par nom de transcription (WER corpus, pas moyenne des WER)."""
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"simulations": 0, "reference_words": 0, "hypothesis_words": 0,
                 "substitutions": 0, "deletions": 0, "insertions": 0}
    )
    for row in rows:
        item = totals[row["hypothesis_file"]]
        item["simulations"] += 1
        for key in ("reference_words", "hypothesis_words",
                    "substitutions", "deletions", "insertions"):
            item[key] += row[key]

    summary = []
    for name, counts in sorted(totals.items()):
        errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
        denominator = counts["reference_words"]
        summary.append({
            "hypothesis_file": name,
            **counts,
            "wer_corpus": errors / denominator if denominator else math.nan,
        })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--metadata-root", type=Path, default=Path("data/generated/dataset"))
    parser.add_argument("--hypothesis-root", type=Path, default=Path("data/generated/transcript"))
    parser.add_argument("--result-dir", type=Path, default=Path("data/generated/result"))
    parser.add_argument(
        "--hypothesis-name", default=None,
        help="nom précis du .txt à évaluer (défaut : tous les .txt de chaque simulation)",
    )
    parser.add_argument("--output-name", default="wer_results.csv",
                        help="nom du CSV détaillé (défaut : wer_results.csv)")
    parser.add_argument("--summary-name", default="wer_summary.csv",
                        help="nom du CSV agrégé par type de transcription")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    metadata_files = sorted(args.metadata_root.glob("case_*/sim_*/metadata.json"))
    if not metadata_files:
        logger.error("Aucun metadata.json sous %s", args.metadata_root)
        return 1

    rows: list[dict] = []
    failures = 0
    for metadata_path in metadata_files:
        case, sim = metadata_path.parent.parent.name, metadata_path.parent.name
        hypothesis_dir = args.hypothesis_root / case / sim
        if args.hypothesis_name:
            hypotheses = [hypothesis_dir / args.hypothesis_name]
        else:
            hypotheses = sorted(hypothesis_dir.glob("*.txt")) if hypothesis_dir.is_dir() else []

        if not hypotheses:
            logger.warning("Aucune hypothèse .txt pour %s/%s (%s)", case, sim, hypothesis_dir)
            continue

        try:
            reference = load_reference(metadata_path)
            for hypothesis_path in hypotheses:
                if not hypothesis_path.is_file():
                    logger.warning("Hypothèse absente : %s", hypothesis_path)
                    continue
                try:
                    hypothesis = hypothesis_path.read_text(encoding="utf-8-sig")
                    scores = score_transcript(reference, hypothesis)
                    row = {
                        "case": case,
                        "sim": sim,
                        "hypothesis_file": hypothesis_path.name,
                        **scores,
                    }
                    rows.append(row)
                    wer_text = f"{scores['wer']:.3f}" if math.isfinite(scores["wer"]) else "N/A"
                    logger.info("%s/%s/%s | WER=%s | S=%d D=%d I=%d",
                                case, sim, hypothesis_path.name, wer_text,
                                scores["substitutions"], scores["deletions"],
                                scores["insertions"])
                except Exception:
                    failures += 1
                    logger.exception("Échec sur l'hypothèse %s", hypothesis_path)
        except Exception:
            failures += 1
            logger.exception("Échec sur les métadonnées %s", metadata_path)

    if rows:
        detailed_path = args.result_dir / args.output_name
        summary_path = args.result_dir / args.summary_name
        write_csv(detailed_path, rows)
        summaries = corpus_summary(rows)
        write_csv(summary_path, summaries)
        for summary in summaries:
            wer = summary["wer_corpus"]
            wer_text = f"{wer:.4f}" if math.isfinite(wer) else "N/A"
            logger.info("WER corpus %-24s : %s (%d mots de référence, %d simulations)",
                        summary["hypothesis_file"], wer_text,
                        summary["reference_words"], summary["simulations"])
        logger.info("CSV détaillé : %s\nCSV agrégé : %s", detailed_path, summary_path)
    else:
        logger.error("Aucune transcription évaluée.")
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
