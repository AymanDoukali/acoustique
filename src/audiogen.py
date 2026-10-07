import glob
import json
import os
import random
import uuid
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import datetime


N_SPEAKERS = 10


class audiogen:

    def __init__(
        self,
        raw_path=Path("data/raw/LibriSpeech/test-clean"),
        output_base_dir=Path("data/generated/speech"),
    ):
        # pathlib gère les séparateurs de chemins sous Linux et Windows.
        self.raw_path = Path(raw_path)
        self.output_base_dir = Path(output_base_dir)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def _load_transcripts(self, trans_file_path):
        """Lit un fichier .trans.txt et extrait un dictionnaire {id_audio: texte}"""
        transcripts = {}
        if os.path.exists(trans_file_path):
            with open(trans_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcripts[utt_id] = text
        return transcripts

    def get_random_speakers(self, num_speakers=10):
        """Sélectionne aléatoirement un nombre donné de speakers à partir du répertoire raw_path."""
        speaker_dirs = [
            d
            for d in os.listdir(self.raw_path)
            if os.path.isdir(os.path.join(self.raw_path, d))
        ]
        if len(speaker_dirs) < num_speakers:
            raise ValueError(
                f"Pas assez de speakers dans {self.raw_path} (minimum {num_speakers} requis)."
            )
        return random.sample(speaker_dirs, num_speakers)
    
    def generate(self, target_duration_sec=60.0, n_speakers = 3, speakers=None, custom_track_id=None, normalize_audio=True):
        """Génère 3 fichiers audio de 1 minute synchronisés.

        Une seule personne parle à la fois : la piste du speaker actif contient de l'audio,
        tandis que les deux autres pistes contiennent du silence.
        """
        track_id = custom_track_id or f"track_{uuid.uuid4().hex[:8]}"
        track_dir = os.path.join(self.output_base_dir, track_id)
        os.makedirs(track_dir, exist_ok=True)
    
        if speakers is None:
            speakers = self.get_random_speakers(n_speakers)

        speaker_dirs = [
            d
            for d in os.listdir(self.raw_path)
            if os.path.isdir(os.path.join(self.raw_path, d))
        ]

        for speaker in speakers:
            if speaker not in speaker_dirs:
                raise ValueError(
                    f"Le speaker {speaker} n'existe pas dans {self.raw_path}."
                )

        selected_speakers = random.sample(speakers, n_speakers)

        sr = 16000
        total_samples = int(target_duration_sec * sr)

        audio_tracks = [
            np.zeros(total_samples, dtype=np.float32) for _ in range(3)
        ]

        current_time = 0.0
        speaker_idx = 0
        segments_info = []

        while current_time < target_duration_sec:
            active_idx = speaker_idx % n_speakers
            speaker_id = selected_speakers[active_idx]
            speaker_path = os.path.join(self.raw_path, speaker_id)

            flac_files = glob.glob(
                os.path.join(speaker_path, "**", "*.flac"), recursive=True
            )

            if not flac_files:
                speaker_idx += 1
                continue

            audio_file = random.choice(flac_files)
            audio_data, _ = librosa.load(audio_file, sr=sr)
            duration = librosa.get_duration(y=audio_data, sr=sr)

            if current_time + duration > target_duration_sec:
                duration = target_duration_sec - current_time
                max_samples_clip = int(duration * sr)
                audio_data = audio_data[:max_samples_clip]

            start_sample = int(current_time * sr)
            end_sample = start_sample + len(audio_data)

            audio_tracks[active_idx][start_sample:end_sample] = audio_data

            chapter_dir = os.path.dirname(audio_file)
            trans_files = glob.glob(os.path.join(chapter_dir, "*.trans.txt"))

            speech_text = ""
            if trans_files:
                transcripts = self._load_transcripts(trans_files[0])
                utt_id = os.path.splitext(os.path.basename(audio_file))[0]
                speech_text = transcripts.get(utt_id, "")

            segments_info.append(
                {
                    "speaker_index": active_idx + 1,
                    "speaker_id": speaker_id,
                    "audio_file": os.path.basename(audio_file),
                    "text": speech_text,
                    "start_time_sec": round(current_time, 2),
                    "end_time_sec": round(current_time + duration, 2),
                    "duration_sec": round(duration, 2),
                }
            )

            current_time += duration
            speaker_idx += 1

        files_metadata = []
        for i, (speaker_id, track_signal) in enumerate(
            zip(selected_speakers, audio_tracks), start=1
        ):
            file_name = f"speaker_{i}_{speaker_id}.wav"
            wav_path = os.path.join(track_dir, file_name)

            if normalize_audio:
                track_signal = librosa.util.normalize(track_signal)

            sf.write(wav_path, track_signal, sr)

            files_metadata.append(
                {
                    "speaker_index": i,
                    "speaker_id": speaker_id,
                    "file_name": file_name,
                    "duration_sec": target_duration_sec,
                }
            )

        # Enregistrement des métadonnées
        metadata_path = os.path.join(track_dir, "metadata.json")
        json_data = {
            "track_id": track_id,
            "total_duration_sec": target_duration_sec,
            "speakers_count": len(selected_speakers),
            "speakers": selected_speakers,
            "normalize_audio": normalize_audio,
            "audios": files_metadata,
            "timeline_segments": segments_info,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        return track_id, audio_tracks



def generate_batch(n_gen=1, target_duration_sec=60.0, n_speakers=3):
    """Génère un lot de pistes audio synchronisées."""
    PATH = Path(f"data/generated/speech/batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    gen_audio = audiogen(output_base_dir = PATH)
    speakers = gen_audio.get_random_speakers(N_SPEAKERS)
    track_ids = []
    for _ in range(n_gen):
        track_id, _ = gen_audio.generate(
            target_duration_sec=target_duration_sec, n_speakers=n_speakers, speakers=speakers
        )
        track_ids.append(track_id)
    return track_ids
    
if __name__ == "__main__":
    #audio_generator = audiogen()
    #track_id = audio_generator.generate()
    print(generate_batch(n_gen=60, target_duration_sec=60.0, n_speakers=3))
    print(generate_batch(n_gen=60, target_duration_sec=60.0, n_speakers=2))