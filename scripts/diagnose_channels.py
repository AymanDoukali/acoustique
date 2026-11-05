"""Diagnostic : pourquoi les SNR des trois micros sont-ils identiques ?

Compare les canaux ORIGINAUX puis les canaux AMÉLIORÉS (mêmes modèles que le
pipeline) : corrélation, écart relatif, puissances de bruit, SNR par fenêtre.

    python diagnose_channels.py data/generated/dataset/case_0/sim_0/mixture_mic_all.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from channel_selection import (
    SelectionConfig,
    channel_similarity,
    compute_snr_db,
    estimate_noise_power,
    stack_signals,
)
from speech_enhancement import DEFAULT_MODEL_TAG, N_MICS, build_enhancers, enhance_all


def report(title: str, signals, cfg: SelectionConfig) -> None:
    x = stack_signals(signals)
    noise_power = estimate_noise_power(x, cfg.noise_samples)
    snr_db = compute_snr_db(x, noise_power, cfg.window_samples)

    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    for m in range(x.shape[0]):
        print(f"Mic {m} | puissance totale={np.mean(x[m] ** 2):.4e} "
              f"| puissance bruit (0-{cfg.noise_duration:.0f}s)={noise_power[m]:.4e}")

    print("\nSimilarité entre canaux (relative_rms_diff ~ 0 => canaux identiques) :")
    for s in channel_similarity(signals):
        i, j = s["pair"]
        print(f"  Mic {i} vs {j} | corr={s['correlation']:.6f} "
              f"| écart RMS relatif={s['relative_rms_diff']:.4e}")

    print("\nSNR (dB) par fenêtre, après le bruit seul :")
    with np.printoptions(precision=3, suppress=True):
        for k in range(cfg.noise_windows, snr_db.shape[0]):
            spread = snr_db[k].max() - snr_db[k].min()
            print(f"  Fenêtre {k}: {snr_db[k]} | écart max-min={spread:.4f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    args = parser.parse_args()

    audio, fs = sf.read(args.input, dtype="float32")
    if audio.ndim != 2 or audio.shape[1] != N_MICS:
        raise SystemExit(f"Attendu {N_MICS} canaux, reçu shape {audio.shape}.")
    cfg = SelectionConfig(fs=fs)

    report("SIGNAUX ORIGINAUX", [audio[:, m] for m in range(N_MICS)], cfg)

    enhancers = build_enhancers(args.model_tag)
    report("SIGNAUX AMÉLIORÉS", enhance_all(enhancers, audio, fs), cfg)


if __name__ == "__main__":
    main()
