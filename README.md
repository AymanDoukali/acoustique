# Acoustique — amélioration, transcription et évaluation WER

Ce projet traite des scènes acoustiques simulées avec deux ou trois locuteurs,
trois microphones et un bruit de ventilateur optionnel. Le pipeline comprend
l'amélioration audio (ESPnet), la transcription ASR et l'évaluation des textes
reconnus par rapport aux références conservées dans `metadata.json`.

## Structure des données

```text
data/generated/dataset/
└── case_0/
    └── sim_0/
        ├── mixture_mic_1.wav
        ├── mixture_mic_2.wav
        ├── mixture_mic_3.wav
        ├── mixture_mic_all.wav
        ├── speaker_*.wav
        └── metadata.json
```

Chaque simulation possède un `metadata.json`. Le champ `timeline_segments`
contient les segments de parole, notamment `text`, `start_time_sec`,
`end_time_sec`, `speaker_index` et `speaker_id`. Pour le WER, les segments sont
triés par `start_time_sec`, puis les textes non vides sont concaténés dans
l'ordre chronologique. Les segments sans texte sont ignorés.

Selon `data_explanation.txt`, le dataset comporte normalement 12 configurations
(`case_0` à `case_11`) et 10 simulations par configuration. Les audios ont un
taux d'échantillonnage nominal de 16 kHz et une durée cible de 60 secondes.

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

ESPnet, PyTorch et leurs dépendances peuvent être longs à installer. Le choix
de la version de PyTorch dépend notamment de la plateforme et du support CPU
ou CUDA ; consulter les instructions PyTorch adaptées à la machine en cas
de difficulté.

## 1. Amélioration audio

Le script `speech_enhancement.py` traite les fichiers `mixture_mic_all.wav`
et écrit les résultats sous `data/generated/speech/case_i/sim_j/`.

```bash
python speech_enhancement.py \
  --model tac \
  --input-root data/generated/dataset \
  --output-root data/generated/speech
```

Deux modèles sont disponibles :

- `tac` (par défaut) : FaSNet-TAC combine les microphones en une seule sortie ;
  il n'y a donc pas de sélection ultérieure entre canaux.
- `mvdr` : le beamformer est exécuté avec chaque microphone comme canal de
  référence, puis le meilleur canal est sélectionné fenêtre par fenêtre.

Exemple MVDR :

```bash
python speech_enhancement.py --model mvdr --hysteresis-db 1.0
```

Options principales : `--noise-duration` (3 s), `--window-sec` (1 s),
`--hysteresis-db` (1 dB), `--input-root` et `--output-root`.

Selon le modèle, les sorties comprennent :

- `enhanced_best.wav` : signal final ;
- `enhanced_ch0.wav`, `enhanced_ch1.wav`, `enhanced_ch2.wav` : sorties par
  canal de référence (produites par le mode MVDR) ;
- `selected_channels.npy`, `snr_matrix.npy`, `noise_power.npy` : diagnostics
  de sélection (mode MVDR).

**Hypothèse de sélection :** les premières secondes (`--noise-duration`)
sont supposées ne contenir que du bruit. Cette hypothèse doit être vérifiée
pour les données utilisées.

## 2. Diagnostic des canaux

```bash
python diagnose_channels.py \
  data/generated/dataset/case_0/sim_0/mixture_mic_all.wav

python diagnose_channels.py \
  data/generated/dataset/case_0/sim_0/mixture_mic_all.wav \
  --model mvdr
```

## 3. Transcription

`transcribe.py` utilise le modèle ESPnet `espnet/owsm_ctc_v4_1B`.
Il transcrit par défaut le signal amélioré `enhanced_best.wav` et le mélange
brut `mixture_mic_all.wav`.

```bash
python transcribe.py
```

Les fichiers texte sont écrits sous `data/generated/transcript/` :

```text
data/generated/transcript/
└── case_0/
    └── sim_0/
        ├── enhanced_best.txt
        └── mixture_mic_all.txt
```

Pour le mélange multicanal, le canal 0 est utilisé par défaut. Il est possible
de choisir un autre canal ou la moyenne des canaux :

```bash
python transcribe.py --mixture-channel 1
python transcribe.py --mixture-channel mean
```

Les transcriptions déjà présentes sont conservées. Pour les recalculer :

```bash
python transcribe.py --overwrite
```

Autres options : `--speech-root`, `--dataset-root`, `--output-root`,
`--enhanced-name`, `--mixture-name`, `--model`, `--lang`, `--device`,
`--batch-size` et `--context-len`.

## 4. Calcul du WER

`compute_wer.py` compare chaque fichier `.txt` généré par `transcribe.py` au
texte de référence reconstruit depuis le `metadata.json` correspondant.
Par défaut, tous les fichiers `.txt` présents dans chaque dossier de simulation
sont évalués séparément.

```bash
python compute_wer.py
```

Les chemins par défaut sont :

- références : `data/generated/dataset` ;
- hypothèses : `data/generated/transcript` ;
- résultats : `data/generated/result`.

Pour évaluer un seul type de transcription :

```bash
python compute_wer.py --hypothesis-name enhanced_best.txt
python compute_wer.py --hypothesis-name mixture_mic_all.txt
```

Pour personnaliser les dossiers :

```bash
python compute_wer.py \
  --metadata-root data/generated/dataset \
  --hypothesis-root data/generated/transcript \
  --result-dir data/generated/result
```

Deux CSV sont produits :

- `wer_results.csv` : une ligne par transcription, avec `case`, `sim`,
  `hypothesis_file`, nombre de mots de référence et d'hypothèse, substitutions,
  suppressions, insertions et WER ;
- `wer_summary.csv` : agrégation par nom de fichier de transcription, avec le
  WER corpus calculé par `(S + D + I) / nombre total de mots de référence`.

Le WER est calculé après passage en minuscules, retrait de la ponctuation et
normalisation des espaces. Il peut être supérieur à 1 lorsque le nombre
d'erreurs dépasse le nombre de mots de référence. Si une référence ne contient
aucun mot, son WER est non défini et la cellule correspondante reste vide dans
le CSV.

Les fichiers sont écrits dans `data/generated/result/`, comme les résultats
de `evaluate.py`. Les noms peuvent être modifiés avec `--output-name` et
`--summary-name`.

## 5. Évaluation audio

`evaluate.py` calcule les métriques de séparation/amélioration (SDR, SIR, SAR),
le gain de SNR et le SI-SDR. Les références audio sont les fichiers
`speaker_*.wav` de chaque simulation.

```bash
python evaluate.py
python evaluate.py --enhanced-name enhanced_ch0.wav
```

Les CSV sont écrits dans `data/generated/result/`. Cette évaluation audio est
distincte du WER : le WER mesure la proximité entre la transcription reconnue
et le texte de référence de `metadata.json`.

## Fichiers générés

Les données, résultats et modèles téléchargés ne sont pas à versionner.
Éviter de committer notamment `.venv/`, `data/` et les caches de modèles.
