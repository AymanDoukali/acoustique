"""Évaluation du signal amélioré : SDR / SIR / SAR (BSS-Eval), gain de SNR et SI-SDR.

Référence = fichiers propres speaker_*.wav du dossier de la simulation.

* BSS-Eval (Vincent et al. 2006, comme mir_eval.separation.bss_eval_sources) :
  l'estimation est projetée sur les versions retardées (0..511 échantillons, soit
  32 ms à 16 kHz) des références. Les retards de propagation de chaque locuteur sont
  donc absorbés par les filtres, sans recalage préalable.
    SDR : cible vs (interférences + artefacts)
    SIR : cible vs interférences (les AUTRES LOCUTEURS)
    SAR : (cible + interférences) vs artefacts (bruit résiduel, distorsion)
  Avec des locuteurs successifs tous conservés, SDR et SIR sont tirés vers le bas
  (SIR ~ 0 dB pour 2 locuteurs, ~ -3 dB pour 3) : ils ne mesurent pas la qualité.
* Gain de SNR : SNR = puissance de la parole (projection sur les références, après
  le bruit seul) / puissance du bruit (les `noise_duration` premières secondes).
  Le gain = SNR de sortie - SNR moyen des micros bruts. Indépendant du gain du modèle.
  Hypothèse : le bruit est stationnaire (celui des 3 premières secondes est
  représentatif de celui pendant la parole).
* SI-SDR : comparé aux micros bruts, avec recalage par locuteur.

    python evaluate.py                          # CSV dans data/generated/result/
    python evaluate.py --enhanced-name enhanced_ch0.wav
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.linalg import LinAlgError, solve, toeplitz

logger = logging.getLogger("evaluate")

EPS = 1e-12
MIXTURE_NAME = "mixture_mic_all.wav"
LENGTH_WARNING_SEC = 0.5   # écart toléré (queue de réverbération) avant avertissement
BSS_FILTER_LENGTH = 512    # même valeur que mir_eval


# --------------------------------------------------------------------------
# Projection sur les références retardées (cœur de BSS-Eval)
# --------------------------------------------------------------------------

class Projector:
    """Projection par moindres carrés d'un signal sur les versions retardées (0..flen-1)
    des références. La matrice de Gram ne dépend que des références : elle est calculée
    une fois, puis réutilisée pour chaque signal projeté.
    """

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
                block = toeplitz(r[:flen], r[lags_neg])   # G[k, m] = r_ij(k - m)
                self.gram[i * flen:(i + 1) * flen, j * flen:(j + 1) * flen] = block
                self.gram[j * flen:(j + 1) * flen, i * flen:(i + 1) * flen] = block.T

    def project(self, est: np.ndarray, idx: list[int] | None = None) -> np.ndarray:
        """Projection de `est` (longueur n) sur les références `idx` (défaut : toutes).

        Renvoie un signal de longueur n + flen - 1.
        """
        if len(est) != self.n:
            raise ValueError(f"est a {len(est)} échantillons, attendu {self.n}.")
        idx = list(range(self.n_src)) if idx is None else idx
        flen = self.flen
        spec_est = np.fft.rfft(est, self.nfft)

        sl = np.concatenate([np.arange(i * flen, (i + 1) * flen) for i in idx])
        rhs = np.concatenate(
            [np.fft.irfft(np.conj(self.spec[i]) * spec_est, self.nfft)[:flen] for i in idx])
        g = self.gram[np.ix_(sl, sl)].copy()
        g.flat[:: g.shape[0] + 1] += 1e-10 * np.trace(g) / g.shape[0]  # régularisation
        try:
            h = solve(g, rhs, assume_a="pos")
        except (LinAlgError, ValueError):
            h = np.linalg.lstsq(g, rhs, rcond=None)[0]
        h = h.reshape(len(idx), flen)
        spec_p = sum(np.fft.rfft(h[k], self.nfft) * self.spec[i] for k, i in enumerate(idx))
        return np.fft.irfft(spec_p, self.nfft)[: self.n + flen - 1]


def _db(num: float, den: float) -> float:
    return float(10 * np.log10((num + 1e-20) / (den + 1e-20)))


def bss_eval(
    est: np.ndarray,
    refs: np.ndarray,
    flen: int = BSS_FILTER_LENGTH,
    projector: Projector | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SDR, SIR, SAR (dB) de `est` pour chaque référence de `refs` (n_src, n_samples).

    Pour la référence j : s_filt = projection de est sur les versions retardées de
    refs[j] ; p_all = projection sur toutes les références. Alors
        SDR = |s_filt|² / |est - s_filt|²
        SIR = |s_filt|² / |p_all - s_filt|²   (nan s'il n'y a qu'une référence)
        SAR = |p_all|²  / |est - p_all|²
    """
    n_src = refs.shape[0]
    proj = projector or Projector(refs, flen)
    est_p = np.concatenate([est, np.zeros(proj.flen - 1)])

    p_all = proj.project(est)
    e_artif = est_p - p_all
    sdr, sir, sar = (np.full(n_src, np.nan) for _ in range(3))
    for j in range(n_src):
        if refs[j] @ refs[j] < EPS:
            logger.warning("Référence %d silencieuse sur la zone évaluée.", j)
            continue
        p_j = p_all if n_src == 1 else proj.project(est, [j])
        sdr[j] = _db(p_j @ p_j, (est_p - p_j) @ (est_p - p_j))
        if n_src > 1:
            e_interf = p_all - p_j
            sir[j] = _db(p_j @ p_j, e_interf @ e_interf)
        sar[j] = _db(p_all @ p_all, e_artif @ e_artif)
    return sdr, sir, sar


def speech_to_noise_db(signal: np.ndarray, proj: Projector, start: int) -> float:
    """SNR (dB) : parole après `start` (projection sur les références) / bruit avant `start`.

    `proj` doit avoir été construit sur les références tronquées à partir de `start`.
    """
    noise_power = max(float(np.mean(signal[:start] ** 2)), EPS)
    p = proj.project(signal[start:])
    speech_power = float(p @ p) / (len(signal) - start)
    return _db(speech_power, noise_power)


# --------------------------------------------------------------------------
# SI-SDR avec recalage par locuteur
# --------------------------------------------------------------------------

def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR en dB (invariant à l'échelle). Signaux 1D de même longueur."""
    est = est - est.mean()
    ref = ref - ref.mean()
    target = (est @ ref) / (ref @ ref + EPS) * ref
    noise = est - target
    return float(10 * np.log10((target @ target + EPS) / (noise @ noise + EPS)))


def estimate_lag(est: np.ndarray, ref: np.ndarray, max_lag: int) -> int:
    """Retard (en échantillons) de `est` par rapport à `ref`, par corrélation croisée.

    lag > 0 : est est en retard sur ref.
    """
    nfft = 1 << int(np.ceil(np.log2(len(est) + len(ref))))
    corr = np.fft.irfft(np.fft.rfft(est, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    return int(lags[np.argmax(corr[lags % nfft])])


def delay(x: np.ndarray, lag: int) -> np.ndarray:
    """Retarde x de `lag` échantillons (avance si négatif), même longueur, complété de zéros."""
    if lag > 0:
        return np.concatenate([np.zeros(lag), x[:-lag]])
    if lag < 0:
        return np.concatenate([x[-lag:], np.zeros(-lag)])
    return x.copy()


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def aligned_reference(
    est: np.ndarray, sources: list[np.ndarray], start: int, max_lag: int
) -> tuple[np.ndarray, list[int]]:
    """Somme des locuteurs, chacun recalé sur `est` par son propre retard."""
    n = len(est)
    ref = np.zeros(n)
    lags = []
    for src in sources:
        src = fit_length(src, n)
        lag = estimate_lag(est[start:], src[start:], max_lag)
        lags.append(lag)
        ref += delay(src, lag)
    return ref, lags


def scored_si_sdr(
    est: np.ndarray, sources: list[np.ndarray], start: int, max_lag: int
) -> tuple[float, list[int]]:
    """SI-SDR après exclusion du début et recalage par locuteur. Renvoie (SI-SDR, retards)."""
    est = est.astype(np.float64)
    ref, lags = aligned_reference(est, sources, start, max_lag)
    return si_sdr(est[start:], ref[start:]), lags


# --------------------------------------------------------------------------
# Lecture des fichiers
# --------------------------------------------------------------------------

def read_mono(path: Path) -> tuple[np.ndarray, int]:
    x, fs = sf.read(path, dtype="float64")
    return (x if x.ndim == 1 else x.mean(axis=1)), fs


def load_sources(
    sim_dir: Path, n_samples: int, fs: int, offset_sec: float
) -> tuple[list[str], list[np.ndarray]]:
    files = sorted(sim_dir.glob("speaker_*.wav"))
    if not files:
        raise FileNotFoundError(f"Aucun speaker_*.wav dans {sim_dir}")
    names, sources = [], []
    for f in files:
        x, f_fs = read_mono(f)
        if f_fs != fs:
            raise ValueError(f"{f} : {f_fs} Hz, attendu {fs} Hz")
        x = np.pad(x, (int(round(offset_sec * fs)), 0))
        if abs(len(x) - n_samples) > LENGTH_WARNING_SEC * fs:
            logger.warning("%s : %d échantillons contre %d pour le mélange "
                           "(utilise --ref-offset si la parole démarre à 3 s).",
                           f.name, len(x), n_samples)
        names.append(f.stem)
        sources.append(x)
    return names, sources


# --------------------------------------------------------------------------
# Évaluation d'une simulation
# --------------------------------------------------------------------------

def evaluate_sim(sim_dir: Path, enhanced_path: Path, args) -> tuple[dict, list[dict]]:
    """Renvoie (ligne par simulation, lignes par locuteur)."""
    mix, fs = sf.read(sim_dir / MIXTURE_NAME, dtype="float64")
    enhanced, fs_e = read_mono(enhanced_path)
    if fs_e != fs:
        raise ValueError(f"{enhanced_path} : {fs_e} Hz, attendu {fs} Hz")

    names, sources = load_sources(sim_dir, mix.shape[0], fs, args.ref_offset)
    start = int(round(args.noise_duration * fs))
    max_lag = int(round(args.max_lag * fs))
    case, sim = sim_dir.parent.name, sim_dir.name

    n = len(enhanced)
    refs = np.stack([fit_length(s, n) for s in sources])[:, start:]
    proj = Projector(refs)

    # BSS-Eval sur la zone après le bruit seul
    sdr, sir, sar = bss_eval(enhanced[start:], refs, projector=proj)
    per_speaker = [
        {"case": case, "sim": sim, "speaker": name,
         "sdr_db": sdr[k], "sir_db": sir[k], "sar_db": sar[k]}
        for k, name in enumerate(names)
    ]

    # Gain de SNR (bruit estimé sur les premières secondes)
    snr_out = speech_to_noise_db(enhanced, proj, start)
    snr_in = float(np.mean([speech_to_noise_db(fit_length(mix[:, m], n), proj, start)
                            for m in range(mix.shape[1])]))

    row = {"case": case, "sim": sim, "n_speakers": len(names),
           "sdr_db": np.nanmean(sdr), "sir_db": np.nanmean(sir), "sar_db": np.nanmean(sar),
           "snr_in_db": snr_in, "snr_out_db": snr_out, "delta_snr_db": snr_out - snr_in}

    row["si_sdr_enhanced"], row["lags_enhanced"] = scored_si_sdr(enhanced, sources, start, max_lag)
    mic_scores = []
    for m in range(mix.shape[1]):
        row[f"si_sdr_mic{m}"], _ = scored_si_sdr(mix[:, m], sources, start, max_lag)
        mic_scores.append(row[f"si_sdr_mic{m}"])
    row["gain_si_sdr_vs_mean_mic"] = row["si_sdr_enhanced"] - float(np.mean(mic_scores))
    return row, per_speaker


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", type=Path, default=Path("data/generated/dataset"))
    p.add_argument("--output-root", type=Path, default=Path("data/generated/speech"))
    p.add_argument("--result-dir", type=Path, default=Path("data/generated/result"),
                   help="dossier où sont écrits les CSV")
    p.add_argument("--enhanced-name", default="enhanced_best.wav")
    p.add_argument("--noise-duration", type=float, default=3.0)
    p.add_argument("--max-lag", type=float, default=0.1,
                   help="retard max (s) pour le recalage du SI-SDR")
    p.add_argument("--ref-offset", type=float, default=0.0,
                   help="secondes de silence à ajouter au début des fichiers speaker_*.wav")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sims = sorted(args.input_root.glob(f"case_*/sim_*/{MIXTURE_NAME}"))
    if not sims:
        logger.error("Aucun %s sous %s", MIXTURE_NAME, args.input_root)
        return 1

    rows, speaker_rows, failures = [], [], 0
    for mix_path in sims:
        sim_dir = mix_path.parent
        enhanced_path = args.output_root / sim_dir.parent.name / sim_dir.name / args.enhanced_name
        try:
            row, per_speaker = evaluate_sim(sim_dir, enhanced_path, args)
        except Exception:
            failures += 1
            logger.exception("Échec sur %s", sim_dir)
            continue
        rows.append(row)
        speaker_rows.extend(per_speaker)
        print(f"{row['case']}/{row['sim']} | SDR={row['sdr_db']:6.2f} SIR={row['sir_db']:6.2f} "
              f"SAR={row['sar_db']:6.2f} dB | SNR {row['snr_in_db']:5.1f} -> {row['snr_out_db']:5.1f} dB "
              f"(gain {row['delta_snr_db']:+.1f}) | SI-SDR={row['si_sdr_enhanced']:6.2f} dB",
              flush=True)

    if rows:
        stem = Path(args.enhanced_name).stem
        sim_csv = args.result_dir / f"bss_eval_{stem}.csv"
        spk_csv = args.result_dir / f"bss_eval_{stem}_par_locuteur.csv"
        write_csv(sim_csv, rows)
        write_csv(spk_csv, speaker_rows)
        print(f"\nMoyenne sur {len(rows)} simulation(s) : "
              f"SDR={np.nanmean([r['sdr_db'] for r in rows]):.2f} dB, "
              f"SIR={np.nanmean([r['sir_db'] for r in rows]):.2f} dB, "
              f"SAR={np.nanmean([r['sar_db'] for r in rows]):.2f} dB, "
              f"gain de SNR={np.mean([r['delta_snr_db'] for r in rows]):+.2f} dB")
        print(f"CSV : {sim_csv}\n      {spk_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
