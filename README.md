# acoustique

Room acoustics experiments with [pyroomacoustics](https://github.com/LCAV/pyroomacoustics).

## Environment setup

The project uses [uv](https://docs.astral.sh/uv/) to manage Python (3.12) and dependencies.
The exact versions are locked in `uv.lock`, so everyone gets the same environment.

1. Install uv (once): https://docs.astral.sh/uv/getting-started/installation/
   - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Clone the repo and create the environment:
   ```
   git clone https://github.com/AymanDoukali/acoustique.git
   cd acoustique
   uv sync
   ```
   This downloads Python 3.12 if needed, creates `.venv/` and installs the locked dependencies.

## Using the environment

Either let uv run things inside the venv for you (no activation needed):

```
uv run python my_script.py
```

Or activate it manually:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Windows (cmd): `.venv\Scripts\activate.bat`
- macOS / Linux: `source .venv/bin/activate`

In VS Code, select the interpreter `.venv` (Ctrl+Shift+P → "Python: Select Interpreter").

## Adding a dependency

```
uv add <package>
```

Then commit the updated `pyproject.toml` and `uv.lock`. Collaborators just run `uv sync` after pulling.
Do not commit `.venv/`.

# Warning

You have to install LibriSpeech in data/raw/
