import numpy as np
import pytest

from channel_selection import (
    SelectionConfig,
    channel_similarity,
    select_best_microphone,
)

FS = 16_000


def make_mics(gains_by_segment, seconds=10, noise_std=0.01, seed=0):
    """3 micros : bruit indépendant, parole (bruit gaussien) dès 3 s avec gains par segment."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    mics = [noise_std * rng.standard_normal(n) for _ in range(3)]
    speech = 0.1 * rng.standard_normal(n)
    for (t0, t1), gains in gains_by_segment.items():
        for m in range(3):
            mics[m][t0 * FS:t1 * FS] += gains[m] * speech[t0 * FS:t1 * FS]
    return mics


def test_selects_expected_channel_and_keeps_length():
    mics = make_mics({(3, 6): (1, 2, 1), (6, 10): (1, 1, 2)})
    res = select_best_microphone(mics, SelectionConfig(fs=FS))
    assert list(res.selected_channels[:3]) == [-1, -1, -1]
    assert list(res.selected_channels[3:6]) == [1, 1, 1]
    assert list(res.selected_channels[6:]) == [2, 2, 2, 2]
    assert len(res.signal) == len(mics[0])


def test_identical_channels_give_identical_output():
    base = make_mics({(3, 10): (1, 1, 1)})[0]
    res = select_best_microphone([base, base, base], SelectionConfig(fs=FS))
    assert np.allclose(res.signal, base, atol=1e-6)  # le fondu ne change pas le niveau


def test_hysteresis_prevents_flipping_on_tiny_differences():
    mics = make_mics({(3, 10): (1.0, 1.03, 1.0)})  # ~0,25 dB d'écart
    res = select_best_microphone(mics, SelectionConfig(fs=FS, hysteresis_db=1.0))
    assert len(set(res.selected_channels[3:])) == 1


def test_too_short_signal_raises():
    with pytest.raises(ValueError):
        select_best_microphone([np.zeros(2 * FS)] * 3, SelectionConfig(fs=FS))


def test_channel_similarity_flags_identical_channels():
    x = np.random.default_rng(1).standard_normal(FS)
    stats = channel_similarity([x, x, x])
    assert all(s["relative_rms_diff"] < 1e-12 for s in stats)
