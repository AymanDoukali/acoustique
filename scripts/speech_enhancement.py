"""Pipeline : amélioration multi-canal (ESPnet) puis sélection du meilleur micro.

Exemple :
    python speech_enhancement.py --input-root data/generated/dataset \
                                 --output-root data/generated/speech
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from espnet2.bin.enh_inference import SeparateSpeech
from espnet_model_zoo.downloader import ModelDownloader

from channel_selection import (
    SelectionConfig,
    channel_similarity,
    select_best_microphone,
)

logger = logging.getLogger("speech_enhancement")

DEFAULT_MODEL_TAG = "lichenda/chime4_fasnet_dprnn_tac"
N_MICS = 3
INPUT_NAME = "mixture_mic_all.wav"
SIMILARITY_WARNING_THRESHOLD = 1e-3


def build_enhancers(model_tag: str = DEFAULT_MODEL_TAG, n_mics: int = N_MICS):
    """Un modèle par canal de référence (le modèle est téléchargé une seule fois).

    normalize_output_wav=False : la normalisation de crête par sortie donnerait
    un gain différent à chaque canal et créerait des sauts de niveau lors de la
    commutation. Les sorties conservent ainsi leurs niveaux relatifs.
    """
    files = ModelDownloader().download_and_unpack(model_tag)
    return [
        SeparateSpeech(
            train_config=files["train_config"],
            model_file=files["model_file"],
            normalize_segment_scale=False,
            show_progressbar=True,
            ref_channel=ref,
            normalize_output_wav=False,
        )
        for ref in range(n_mics)
    ]


def enhance_all(enhancers, audio: np.ndarray, fs: int) -> list[np.ndarray]:
    """Applique chaque modèle à l'enregistrement (n_samples, n_mics)."""
    batch = audio[np.newaxis, ...]
    return [np.asarray(enh(batch, fs)[0]).squeeze() for enh in enhancers]


def process_file(
    input_path: Path, output_dir: Path, enhancers, base_cfg: SelectionConfig
) -> None:
    audio, fs = sf.read(input_path, dtype="float32")
    if audio.ndim != 2 or audio.shape[1] != len(enhancers):
        raise ValueError(f"{input_path} : attendu {len(enhancers)} canaux, "
                         f"reçu shape {audio.shape}.")

    enhanced = enhance_all(enhancers, audio, fs)

    stats = channel_similarity(enhanced)
    if all(s["relative_rms_diff"] < SIMILARITY_WARNING_THRESHOLD for s in stats):
        logger.warning("%s : les %d signaux améliorés sont quasi identiques ; "
                       "la sélection n'a pas de diversité à exploiter.",
                       input_path, len(enhanced))

    result = select_best_microphone(enhanced, replace(base_cfg, fs=fs))

    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "enhanced_best.wav", result.signal, fs, subtype="FLOAT")
    for m, sig in enumerate(enhanced):
        sf.write(output_dir / f"enhanced_ch{m}.wav", sig, fs, subtype="FLOAT")
    np.save(output_dir / "selected_channels.npy", result.selected_channels)
    np.save(output_dir / "snr_matrix.npy", result.snr_db)
    np.save(output_dir / "noise_power.npy", result.noise_power)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, default=Path("data/generated/dataset"))
    p.add_argument("--output-root", type=Path, default=Path("data/generated/speech"))
    p.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    p.add_argument("--noise-duration", type=float, default=3.0)
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--hysteresis-db", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    files = sorted(args.input_root.glob(f"case_*/sim_*/{INPUT_NAME}"))
    if not files:
        logger.error("Aucun fichier %s sous %s", INPUT_NAME, args.input_root)
        return 1

    cfg = SelectionConfig(
        noise_duration=args.noise_duration,
        window_sec=args.window_sec,
        hysteresis_db=args.hysteresis_db,
    )
    enhancers = build_enhancers(args.model_tag)

    failures = 0
    for path in files:
        case, sim = path.parents[1].name, path.parent.name
        logger.info("Traitement : %s/%s", case, sim)
        try:
            process_file(path, args.output_root / case / sim, enhancers, cfg)
        except Exception:
            failures += 1
            logger.exception("Échec sur %s", path)

    logger.info("%d fichier(s) traité(s), %d échec(s).", len(files) - failures, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
