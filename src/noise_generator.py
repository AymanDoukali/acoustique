import os
import librosa
import numpy as np


class NoiseGenerator:

    def __init__(
        self,
        raw_path="data/noises",
    ):
        self.raw_path = raw_path

    def generate(self, target_duration_sec=60.0, noise_type="fan", sr=16000):
        """Génère un bruit de durée exacte et ses métadonnées."""
        audio_file = os.path.join(self.raw_path, f"{noise_type}.wav")
        noise, _ = librosa.load(audio_file, sr=sr)

        total_samples = int(target_duration_sec * sr)

        # Répète le bruit autant de fois que nécessaire, puis coupe à la longueur exacte
        n_repeats = int(np.ceil(total_samples / len(noise)))
        noise = np.tile(noise, n_repeats)[:total_samples].astype(np.float32)

        metadata = {
            "noise_type": noise_type,
            "duration_sec": target_duration_sec,
            "sample_rate": sr,
        }
        return noise, metadata


if __name__ == "__main__":
    noisegen = NoiseGenerator()
    noise, metadata = noisegen.generate(noise_type="fan")
    print(f"Generated {len(noise)} samples: {metadata}")