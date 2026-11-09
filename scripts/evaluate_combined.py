"""Évaluation TAC avec une référence vocale unique (somme des locuteurs).

Les fichiers speaker_*.wav sont des pistes sources synchronisées : chaque piste
est principalement silencieuse lorsque le locuteur correspondant ne parle pas.
On les additionne donc échantillon par échantillon pour obtenir une référence
vocale globale. 

Métriques calculées :
- SDR : fidélité de la sortie par rapport à la référence vocale combinée,
  avec un filtre de projection BSS-Eval.
- SI-SDR : fidélité après compensation d'un gain global, après estimation
  d'un seul retard entre la sortie et la référence combinée.
- SNR entrée/sortie et delta SNR : estimation du bruit sur les premières
  `noise_duration` secondes, supposées sans parole.
- SI-SDR de chaque microphone et gain par rapport à leur moyenne.



Exemples :
    python evaluate_combined.py
    python evaluate_combined.py --enhanced-name enhanced_best.wav
    python evaluate_combined.py --ref-offset 3.0
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import LinAlgError, solve, toeplitz

logger = logging.getLogger("evaluate_combined")

EPS = 1e-12
MIXTURE_NAME = "mixture_mic_all.wav"
BSS_FILTER_LENGTH = 512


class Projector:
    """Projection LS d'un signal sur une référence et ses retards (BSS-Eval)."""

    def __init__(self, refs: np.ndarray, flen: int = BSS_FILTER_LENGTH):
        self.flen = flen
        self.n_src, self.n = refs.shape
        self.nfft = 1 << int(np.ceil(np.log2(self.n + flen)))
        self.spec = np.fft.rfft(refs, self.nfft, axis=1)
        lags_neg = (-np.arange(flen)) % self.nfft
        self.gram = np.zeros((self.n_src * flen, self.n_src * flen))

        for i in range(self.n_src):
            for j in range(i, self.n_src):
                r = np.fft.irfft(np.conj(self.spec[i]) * self.spec[j], self.nfft)
                block = toeplitz(r[:flen], r[lags_neg])
                self.gram[i * flen:(i + 1) * flen,
                          j * flen:(j + 1) * flen] = block
                self.gram[j * flen:(j + 1) * flen,
                          i * flen:(i + 1) * flen] = block.T

    def project(self, est: np.ndarray) -> np.ndarray:
        if len(est) != self.n:
            raise ValueError(f"est a {len(est)} échantillons, attendu {self.n}.")
        spec_est = np.fft.rfft(est, self.nfft)
        rhs = np.fft.irfft(np.conj(self.spec[0]) * spec_est, self.nfft)[:self.flen]
        gram = self.gram.copy()
        gram.flat[:: gram.shape[0] + 1] += (
            1e-10 * np.trace(gram) / gram.shape[0]
        )
        try:
            h = solve(gram, rhs, assume_a="pos")
        except (LinAlgError, ValueError):
            h = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        projected_spec = np.fft.rfft(h, self.nfft) * self.spec[0]
        return np.fft.irfft(projected_spec, self.nfft)[:self.n + self.flen - 1]


def db_ratio(num: float, den: float) -> float:
    return float(10.0 * np.log10((num + 1e-20) / (den + 1e-20)))


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, fs = sf.read(path, dtype="float64")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, fs


def load_combined_reference(
    sim_dir: Path, n_samples: int, fs: int, ref_offset: float
) -> tuple[list[str], np.ndarray]:
    """Additionne les pistes speaker synchronisées, sans recalage individuel."""
    files = sorted(sim_dir.glob("speaker_*.wav"))
    if not files:
        raise FileNotFoundError(f"Aucun speaker_*.wav dans {sim_dir}")

    names: list[str] = []
    combined = np.zeros(n_samples, dtype=np.float64)
    offset = int(round(ref_offset * fs))

    for path in files:
        source, source_fs = read_mono(path)
        if source_fs != fs:
            raise ValueError(f"{path} : {source_fs} Hz, attendu {fs} Hz")
        source = np.pad(source, (offset, 0))
        source = fit_length(source, n_samples)
        combined += source
        names.append(path.stem)

    return names, combined


def estimate_lag(est: np.ndarray, ref: np.ndarray, max_lag: int) -> int:
    """Retard global de est par rapport à ref, estimé par corrélation croisée."""
    nfft = 1 << int(np.ceil(np.log2(len(est) + len(ref))))
    corr = np.fft.irfft(
        np.fft.rfft(est, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft
    )
    lags = np.concatenate([
        np.arange(0, max_lag + 1),
        np.arange(-max_lag, 0)
    ])
    return int(lags[np.argmax(corr[lags % nfft])])


def shift_signal(x: np.ndarray, lag: int) -> np.ndarray:
    if lag > 0:
        return np.concatenate([np.zeros(lag), x[:-lag]])
    if lag < 0:
        return np.concatenate([x[-lag:], np.zeros(-lag)])
    return x.copy()


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    scale = (est @ ref) / (ref @ ref + EPS)
    target = scale * ref
    error = est - target
    return db_ratio(float(target @ target), float(error @ error))


def aligned_si_sdr(
    est: np.ndarray, ref: np.ndarray, start: int, max_lag: int
) -> tuple[float, int]:
    lag = estimate_lag(est[start:], ref[start:], max_lag)
    aligned = shift_signal(ref, lag)
    return si_sdr(est[start:], aligned[start:]), lag


def single_reference_sdr(est: np.ndarray, projector: Projector) -> float:
    """SDR BSS-Eval pour une cible unique (tous les locuteurs additionnés)."""
    projected = projector.project(est)
    est_padded = np.concatenate([est, np.zeros(projector.flen - 1)])
    error = est_padded - projected
    return db_ratio(float(projected @ projected), float(error @ error))


def speech_to_noise_db(
    signal: np.ndarray, projector: Projector, start: int
) -> float:
    if start <= 0 or start >= len(signal):
        raise ValueError(
            f"noise_duration doit être entre 0 et la durée du signal "
            f"(indice {start}, longueur {len(signal)})."
        )
    noise_power = max(float(np.mean(signal[:start] ** 2)), EPS)
    projected = projector.project(signal[start:])
    speech_power = float(projected @ projected) / (len(signal) - start)
    return db_ratio(speech_power, noise_power)


def evaluate_sim(sim_dir: Path, enhanced_path: Path, args) -> dict:
    mix, fs = sf.read(sim_dir / MIXTURE_NAME, dtype="float64")
    if mix.ndim == 1:
        mix = mix[:, None]

    enhanced, enhanced_fs = read_mono(enhanced_path)
    if enhanced_fs != fs:
        raise ValueError(f"{enhanced_path} : {enhanced_fs} Hz, attendu {fs} Hz")

    n = len(enhanced)
    names, reference = load_combined_reference(
        sim_dir, n, fs, args.ref_offset
    )

    start = int(round(args.noise_duration * fs))
    if start >= n:
        raise ValueError("Le signal est plus court que noise_duration.")

    # Référence unique : somme synchronisée de toutes les pistes de locuteurs.
    ref_speech = reference[start: n]
    projector = Projector(ref_speech[None, :])

    sdr = single_reference_sdr(enhanced[start:], projector)
    si_sdr_enhanced, lag_enhanced = aligned_si_sdr(
        enhanced, reference, start, int(round(args.max_lag * fs))
    )

    snr_out = speech_to_noise_db(enhanced, projector, start)
    mic_si_sdr = []
    mic_snr = []
    for channel in range(mix.shape[1]):
        mic = fit_length(mix[:, channel], n)
        score, _ = aligned_si_sdr(
            mic, reference, start, int(round(args.max_lag * fs))
        )
        mic_si_sdr.append(score)
        mic_snr.append(speech_to_noise_db(mic, projector, start))

    snr_in = float(np.mean(mic_snr))
    mean_mic_si_sdr = float(np.mean(mic_si_sdr))

    row = {
        "case": sim_dir.parent.name,
        "sim": sim_dir.name,
        "n_speakers": len(names),
        "reference": "sum_speakers",
        "sdr_combined_db": sdr,
        "si_sdr_combined_db": si_sdr_enhanced,
        "lag_enhanced_samples": lag_enhanced,
        "snr_in_mean_mics_db": snr_in,
        "snr_out_db": snr_out,
        "delta_snr_db": snr_out - snr_in,
        "si_sdr_mic0_db": mic_si_sdr[0] if len(mic_si_sdr) > 0 else np.nan,
        "si_sdr_mic1_db": mic_si_sdr[1] if len(mic_si_sdr) > 1 else np.nan,
        "si_sdr_mic2_db": mic_si_sdr[2] if len(mic_si_sdr) > 2 else np.nan,
        "mean_mic_si_sdr_db": mean_mic_si_sdr,
        "gain_si_sdr_vs_mean_mics_db": si_sdr_enhanced - mean_mic_si_sdr,
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: round(value, 4)
                if isinstance(value, (float, np.floating)) and np.isfinite(value)
                else value
                for key, value in row.items()
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path("data/generated/dataset"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/generated/speech"))
    parser.add_argument("--result-dir", type=Path,
                        default=Path("data/generated/result"))
    parser.add_argument("--enhanced-name", default="enhanced_best.wav")
    parser.add_argument("--noise-duration", type=float, default=3.0)
    parser.add_argument("--max-lag", type=float, default=0.1)
    parser.add_argument("--ref-offset", type=float, default=0.0)
    parser.add_argument("--output-name", default="evaluate_combined.csv")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    mix_files = sorted(
        args.input_root.glob(f"case_*/sim_*/{MIXTURE_NAME}")
    )
    if not mix_files:
        logger.error("Aucun %s sous %s", MIXTURE_NAME, args.input_root)
        return 1

    rows = []
    failures = 0
    for mix_path in mix_files:
        sim_dir = mix_path.parent
        enhanced_path = (
            args.output_root / sim_dir.parent.name / sim_dir.name
            / args.enhanced_name
        )
        try:
            row = evaluate_sim(sim_dir, enhanced_path, args)
            rows.append(row)
            print(
                f"{row['case']}/{row['sim']} | "
                f"SDR={row['sdr_combined_db']:.2f} dB | "
                f"SI-SDR={row['si_sdr_combined_db']:.2f} dB | "
                f"ΔSNR={row['delta_snr_db']:+.2f} dB",
                flush=True
            )
        except Exception:
            failures += 1
            logger.exception("Échec sur %s", sim_dir)

    if rows:
        output = args.result_dir / args.output_name
        write_csv(output, rows)
        print(f"\nÉvaluations réussies : {len(rows)}")
        print(f"Échecs : {failures}")
        print(
            "Moyennes : "
            f"SDR={np.nanmean([r['sdr_combined_db'] for r in rows]):.2f} dB, "
            f"SI-SDR={np.nanmean([r['si_sdr_combined_db'] for r in rows]):.2f} dB, "
            f"gain SNR={np.nanmean([r['delta_snr_db'] for r in rows]):+.2f} dB"
        )
        print(f"CSV : {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
