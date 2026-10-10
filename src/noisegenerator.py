import json
import os
from audiogen import audiogen
import librosa
import numpy as np
import soundfile as sf


class NoiseGenerator:

    def __init__(
        self,
        track_id,
        raw_path="data/noises",
        output_base_dir="data/generated/speech",
    ):
        self.raw_path = raw_path
        self.output_base_dir = output_base_dir
        self.track_id = track_id
        # Même dossier que audiogen : output_base_dir/track_id
        self.track_dir = os.path.join(output_base_dir, str(track_id))

    def generate(self, target_duration_sec=60.0, noise_type="fan", sr=16000):
        """Génère un fichier .wav de bruit de durée exacte target_duration_sec
        dans le dossier de la piste, et retourne le signal."""
        audio_file = os.path.join(self.raw_path, f"{noise_type}.wav")
        noise, _ = librosa.load(audio_file, sr=sr)

        total_samples = int(target_duration_sec * sr)

        # Répète le bruit autant de fois que nécessaire, puis coupe à la longueur exacte
        n_repeats = int(np.ceil(total_samples / len(noise)))
        noise = np.tile(noise, n_repeats)[:total_samples].astype(np.float32)

        os.makedirs(self.track_dir, exist_ok=True)
        file_name = f"noise_{noise_type}.wav"
        wav_path = os.path.join(self.track_dir, file_name)
        sf.write(wav_path, noise, sr)

        self._update_metadata(file_name, noise_type, target_duration_sec)
        return noise

    def _update_metadata(self, file_name, noise_type, duration_sec):
        """Ajoute le bruit au metadata.json créé par audiogen (s'il existe)."""
        metadata_path = os.path.join(self.track_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["noise"] = {
            "file_name": file_name,
            "noise_type": noise_type,
            "duration_sec": duration_sec,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    audio_generator = audiogen()
    track_id, _ = audio_generator.generate()

    noisegen = NoiseGenerator(track_id)
    noisegen.generate(noise_type="fan")