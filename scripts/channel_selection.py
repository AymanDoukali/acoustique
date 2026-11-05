"""Sélection adaptative du meilleur microphone, fenêtre par fenêtre.

Hypothèse : les `noise_duration` premières secondes ne contiennent que du
bruit (les locuteurs commencent à parler ensuite). La puissance de bruit de
chaque canal est estimée sur ce segment, puis un SNR est calculé par fenêtre.

Ce module ne dépend ni d'ESPnet ni du disque : il travaille sur des tableaux
numpy, ce qui le rend testable seul.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


@dataclass(frozen=True)
class SelectionConfig:
    fs: int = 16_000
    noise_duration: float = 3.0   # secondes de bruit seul au début
    window_sec: float = 1.0       # taille des fenêtres de décision
    hysteresis_db: float = 1.0    # gain minimal (dB) pour quitter le canal courant
    crossfade_sec: float = 0.01   # fondu enchaîné lors d'un changement de canal
    default_channel: int = 0      # canal utilisé pendant le bruit seul

    @property
    def noise_samples(self) -> int:
        return int(round(self.noise_duration * self.fs))

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.fs))

    @property
    def noise_windows(self) -> int:
        """Nombre de fenêtres recouvrant le segment de bruit seul (arrondi au-dessus)."""
        return -(-self.noise_samples // self.window_samples)


@dataclass(frozen=True)
class SelectionResult:
    signal: np.ndarray            # (n_samples,) signal final, float32
    selected_channels: np.ndarray  # (n_windows,) ; -1 = fenêtre de bruit seul
    snr_db: np.ndarray            # (n_windows, n_mics)
    noise_power: np.ndarray       # (n_mics,)


# --------------------------------------------------------------------------
# Briques élémentaires
# --------------------------------------------------------------------------

def stack_signals(signals: Sequence[np.ndarray]) -> np.ndarray:
    """Convertit une liste de signaux mono en tableau (n_mics, n_samples)."""
    arrays = [np.asarray(s, dtype=np.float64).squeeze() for s in signals]
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Chaque signal doit être mono (1D après squeeze).")
    lengths = {len(a) for a in arrays}
    n = min(lengths)
    if len(lengths) > 1:
        logger.warning("Longueurs différentes %s : troncature à %d échantillons.",
                       sorted(lengths), n)
    return np.stack([a[:n] for a in arrays])


def estimate_noise_power(x: np.ndarray, noise_samples: int) -> np.ndarray:
    """Puissance moyenne par canal sur le segment de bruit seul."""
    if x.shape[1] < noise_samples:
        raise ValueError("Signal plus court que le segment de bruit seul.")
    return np.mean(x[:, :noise_samples] ** 2, axis=1)


def compute_snr_db(
    x: np.ndarray, noise_power: np.ndarray, window_samples: int
) -> np.ndarray:
    """SNR estimé (dB) par fenêtre et par canal -> (n_windows, n_mics).

    SNR = (P_fenêtre - P_bruit) / P_bruit, avec plancher EPS.
    """
    n_mics, n_samples = x.shape
    n_windows = n_samples // window_samples
    frames = x[:, : n_windows * window_samples].reshape(n_mics, n_windows, window_samples)
    total_power = np.mean(frames ** 2, axis=2).T                 # (n_windows, n_mics)
    speech_power = np.maximum(total_power - noise_power[None, :], EPS)
    return 10.0 * np.log10(speech_power / np.maximum(noise_power, EPS)[None, :])


def select_channels(
    snr_db: np.ndarray, n_noise_windows: int, hysteresis_db: float
) -> np.ndarray:
    """Meilleur canal par fenêtre, avec hystérésis.

    On ne quitte le canal courant que si un autre le dépasse de plus de
    `hysteresis_db`. Sans cela, des écarts de quelques millièmes de dB
    feraient basculer le choix au hasard à chaque fenêtre.
    """
    selected = np.full(snr_db.shape[0], -1, dtype=int)
    current = None
    for k in range(n_noise_windows, snr_db.shape[0]):
        best = int(np.argmax(snr_db[k]))
        if current is None or snr_db[k, best] > snr_db[k, current] + hysteresis_db:
            current = best
        selected[k] = current
    return selected


def build_output(x: np.ndarray, selected: np.ndarray, cfg: SelectionConfig) -> np.ndarray:
    """Assemble le signal final avec un fondu enchaîné entre canaux.

    Le signal garde la durée complète de l'entrée (la fin, plus courte qu'une
    fenêtre, reprend le dernier canal choisi).
    """
    n_mics, n_samples = x.shape
    w = cfg.window_samples

    labels = np.where(selected < 0, cfg.default_channel, selected)
    channel_per_sample = np.empty(n_samples, dtype=int)
    covered = len(labels) * w
    channel_per_sample[:covered] = np.repeat(labels, w)
    channel_per_sample[covered:] = labels[-1] if len(labels) else cfg.default_channel

    masks = (channel_per_sample[None, :] == np.arange(n_mics)[:, None]).astype(np.float64)

    fade = int(round(cfg.crossfade_sec * cfg.fs))
    if fade > 1:
        kernel = np.full(fade, 1.0 / fade)
        left = fade // 2
        padded = np.pad(masks, ((0, 0), (left, fade - 1 - left)), mode="edge")
        masks = np.stack([np.convolve(m, kernel, mode="valid") for m in padded])
        # Les masques lissés somment toujours à 1 : pas de saut de niveau.

    return np.sum(masks * x, axis=0)


# --------------------------------------------------------------------------
# API principale
# --------------------------------------------------------------------------

def select_best_microphone(
    signals: Sequence[np.ndarray], config: SelectionConfig | None = None
) -> SelectionResult:
    cfg = config or SelectionConfig()
    x = stack_signals(signals)

    if not 0 <= cfg.default_channel < x.shape[0]:
        raise ValueError("default_channel hors limites.")

    noise_power = estimate_noise_power(x, cfg.noise_samples)
    snr_db = compute_snr_db(x, noise_power, cfg.window_samples)
    if snr_db.shape[0] <= cfg.noise_windows:
        raise ValueError("Signal trop court : aucune fenêtre après le bruit seul.")

    selected = select_channels(snr_db, cfg.noise_windows, cfg.hysteresis_db)
    signal = build_output(x, selected, cfg).astype(np.float32)

    logger.debug("Puissances de bruit : %s", noise_power)
    return SelectionResult(signal, selected, snr_db, noise_power)


# --------------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------------

def channel_similarity(signals: Sequence[np.ndarray]) -> list[dict]:
    """Corrélation et écart RMS relatif entre chaque paire de canaux.

    Si les canaux sont quasi identiques, leurs SNR le seront aussi : ce n'est
    alors pas un bug de calcul, mais un manque de diversité entre les canaux.
    """
    x = stack_signals(signals)
    results = []
    for i, j in itertools.combinations(range(x.shape[0]), 2):
        a, b = x[i], x[j]
        rms_a = max(float(np.sqrt(np.mean(a ** 2))), EPS)
        results.append({
            "pair": (i, j),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
            "relative_rms_diff": float(np.sqrt(np.mean((a - b) ** 2)) / rms_a),
        })
    return results
