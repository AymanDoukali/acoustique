import os
import numpy as np
import soundfile as sf
from audio_generator import AudioGenerator
from noise_generator import NoiseGenerator
from room_simulator import GenerationResultat  
import json

def genrate_data(output_dir=os.path.join("data", "generated", "final"), fs=16000, target_duration=60, absorbtions=[0.1,0.2,0.3], noise_gain=0.1, n_sim=10, noise_type="fan"):
    # 10 speakers signals and choose among them
    gen_audio = AudioGenerator()
    gen_noise = NoiseGenerator()
    NOISE, noise_metadata = gen_noise.generate(
        target_duration_sec=target_duration, noise_type=noise_type, sr=fs
    )
    case_ = 0
    root_dir = output_dir
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    for absorbtion in absorbtions:
        for noise in [True, False]:
            for n_speaker in [2,3]:
                case_dir = os.path.join(root_dir, "case_" + str(case_))
                if not os.path.exists(case_dir):
                    os.mkdir(case_dir)
                for j in range(n_sim):
                    signals, audio_metadata = gen_audio.generate(
                        target_duration_sec=target_duration, n_speakers=n_speaker
                    )
                    nois_signal = None if noise == False else NOISE
                    simu = GenerationResultat(signals, noise_signal=nois_signal, fs=fs, noise_gain=noise_gain, absorption=absorbtion)
                    mics = simu.get_result()         

                    track_dir = os.path.join(case_dir, "sim_"+str(j))
                    if not os.path.exists(track_dir):
                        os.mkdir(track_dir)
                    for i, signal in enumerate(signals, start=1):
                        file_name = audio_metadata["audios"][i - 1]["file_name"]
                        sf.write(os.path.join(track_dir, file_name), signal, fs)
                    if noise:
                        sf.write(os.path.join(track_dir, "noise_fan.wav"), NOISE, fs)
                    mics = mics / np.abs(mics).max()  
                    for i in range(mics.shape[0]):
                        sf.write(os.path.join(track_dir, f"mixture_mic_{i+1}.wav"), mics[i], fs)
                    sf.write(os.path.join(track_dir, "mixture_mic_all.wav"), mics.T, fs)

                    metadata_path = os.path.join(track_dir, "metadata.json")
                    json_data = {
                        **audio_metadata,
                        "noise_metadata": {
                            **noise_metadata,
                            "file_name": "noise_fan.wav",
                        },
                        "absorbtion": absorbtion,
                        "speakers_count": n_speaker,
                        "noise": noise,

                    }
            
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=4, ensure_ascii=False)
                case_ += 1
