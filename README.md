# Acoustique

Room-acoustics experiments that generate multi-speaker signals, add optional
noise, and simulate recordings from a three-microphone array with
[pyroomacoustics](https://github.com/LCAV/pyroomacoustics).

## 1. Download the required data

The speech generator uses the **LibriSpeech `test-clean` dataset**. Download
the archive from [OpenSLR 12](https://www.openslr.org/12), then extract it so
that the repository contains this directory:

```text
data/raw/LibriSpeech/test-clean/
```

The directory must contain the LibriSpeech speaker folders (for example
`61/70968/`) and their `.flac` audio files and transcript files.

The default configuration also expects a fan-noise file at:

```text
data/noises/fan.wav
```

Place a WAV recording of fan noise at that location, or update the
`raw_path`/`noise_type` configuration before running the generator.

## 2. Set up the environment

Python 3.12 or newer is required. From the repository root, create and
activate a virtual environment:

### Linux and macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project also includes a `pyproject.toml` and `uv.lock` for users who
prefer [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## 3. Generate the dataset

Run the command from the repository root so that the relative `data/` paths
resolve correctly:

```bash
python src/main.py
```

If the current directory is already `src/`, the equivalent command is:

```bash
python main.py
```

Generated files are written under `data/generated/final/`. Each simulation
directory contains:

- the raw speaker tracks;
- `noise_fan.wav` when noise is enabled;
- `mixture_mic_1.wav`, `mixture_mic_2.wav`, and `mixture_mic_3.wav`;
- `mixture_mic_all.wav`, a multichannel WAV containing all microphones;
- `metadata.json`, containing the generation and simulation metadata.

## Development

Do not commit `.venv/` or generated data. To add a dependency, update
`requirements.txt` and install it with:

```bash
python -m pip install -r requirements.txt
```
