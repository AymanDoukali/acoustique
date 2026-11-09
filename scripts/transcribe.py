"""Transcription (OWSM-CTC v4) du signal amélioré et du mélange brut.

Pour chaque case_*/sim_* :
    data/generated/speech/case_i/sim_j/enhanced_best.wav   -> transcript/case_i/sim_j/enhanced_best.txt
    data/generated/dataset/case_i/sim_j/mixture_mic_all.wav -> transcript/case_i/sim_j/mixture_mic_all.txt

Le mélange est multicanal : on transcrit un seul canal (--mixture-channel, 0 par défaut)
ou la moyenne des canaux (--mixture-channel mean). Les transcriptions déjà présentes sont
conservées (relance possible après interruption) ; --overwrite les refait.

    python transcribe.py
    python transcribe.py --enhanced-name enhanced_ch0.wav --overwrite
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logger = logging.getLogger("transcribe")

MODEL_TAG = "espnet/owsm_ctc_v4_1B"
TARGET_FS = 16_000


@dataclass(frozen=True)
class Job:
    root: Path
    filename: str
    channel: int | None   # None : moyenne des canaux


def load_audio(path: Path, channel: int | None) -> np.ndarray:
    """Lit un WAV et renvoie un signal mono float32 à 16 kHz."""
    x, fs = sf.read(path, dtype="float32", always_2d=True)   # (n_samples, n_channels)
    if channel is None:
        mono = x.mean(axis=1)
    else:
        if channel >= x.shape[1]:
            raise ValueError(f"{path} : canal {channel} demandé, {x.shape[1]} canal(aux) disponible(s).")
        mono = x[:, channel]
    if fs != TARGET_FS:
        g = gcd(fs, TARGET_FS)
        mono = resample_poly(mono, TARGET_FS // g, fs // g).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def build_model(model_tag: str, lang: str, device: str | None):
    import torch
    from espnet2.bin.s2t_inference import Speech2Text

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Chargement de %s sur %s (langue %s)", model_tag, device, lang)
    return Speech2Text.from_pretrained(
        model_tag,
        device=device,
        use_flash_attn=False,
        generate_interctc_outputs=False,
        lang_sym=lang,
        task_sym="<asr>",
    )


def transcribe(s2t, audio: np.ndarray, batch_size: int, context_len_in_secs: int) -> str:
    """Texte complet. Le modèle CTC renvoie un seul segment, sans horodatage."""
    segments = s2t.decode_long(
        audio, batch_size=batch_size, context_len_in_secs=context_len_in_secs
    )
    return " ".join(text.strip() for _, _, text in segments).strip()


def output_path(output_root: Path, audio_path: Path) -> Path:
    case, sim = audio_path.parts[-3], audio_path.parts[-2]
    return output_root / case / sim / f"{audio_path.stem}.txt"


def parse_channel(value: str) -> int | None:
    return None if value == "mean" else int(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--speech-root", type=Path, default=Path("data/generated/speech"))
    p.add_argument("--dataset-root", type=Path, default=Path("data/generated/dataset"))
    p.add_argument("--output-root", type=Path, default=Path("data/generated/transcript"))
    p.add_argument("--enhanced-name", default="enhanced_best.wav")
    p.add_argument("--mixture-name", default="mixture_mic_all.wav")
    p.add_argument("--mixture-channel", default="0",
                   help="canal du mélange à transcrire (0, 1, 2…) ou 'mean'")
    p.add_argument("--model", default=MODEL_TAG)
    p.add_argument("--lang", default="<eng>", help="symbole de langue du modèle")
    p.add_argument("--device", default=None, help="cuda / cpu (défaut : auto)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--context-len", type=int, default=4, help="contexte entre blocs (s)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    jobs = [
        Job(args.speech_root, args.enhanced_name, None),
        Job(args.dataset_root, args.mixture_name, parse_channel(args.mixture_channel)),
    ]
    todo: list[tuple[Path, Path, Job]] = []
    for job in jobs:
        found = sorted(job.root.glob(f"case_*/sim_*/{job.filename}"))
        if not found:
            logger.warning("Aucun %s sous %s", job.filename, job.root)
        for path in found:
            out = output_path(args.output_root, path)
            if out.exists() and not args.overwrite:
                logger.info("Déjà fait, ignoré : %s", out)
                continue
            todo.append((path, out, job))

    if not todo:
        logger.info("Rien à transcrire.")
        return 0

    s2t = build_model(args.model, args.lang, args.device)

    failures = 0
    for i, (path, out, job) in enumerate(todo, 1):
        logger.info("[%d/%d] %s", i, len(todo), path)
        try:
            audio = load_audio(path, job.channel)
            text = transcribe(s2t, audio, args.batch_size, args.context_len)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
        except Exception:
            failures += 1
            logger.exception("Échec sur %s", path)

    logger.info("%d transcription(s) écrite(s), %d échec(s). Dossier : %s",
                len(todo) - failures, failures, args.output_root)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
