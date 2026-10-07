import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from audiogen import audiogen                   
from generationresultat import GenerationResultat  
import random
import json

FS = 16000
# 10 speakers signals and choose among them
NOISE = np.random.normal(0, 1, 60*16000)
absorbtions = [0.1,0.2,0.3]
max_order = 10
noise_gain = 0.1
n_sim = 10
gen_audio = audiogen()
case_ = 0

root_dir = os.path.join("data", "generated", "final")
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

for absorbtion in absorbtions:
    for noise in [True, False]:
        for n_speaker in [2,3]:
            case_dir = os.path.join(root_dir, "case_" + str(case_))
            if not os.path.exists(case_dir):
                os.mkdir(case_dir)
            for j in range(n_sim):
                track_id, signals = gen_audio.generate(target_duration_sec=60.0, n_speakers=n_speaker)
                nois_signal = None if noise == False else NOISE
                simu = GenerationResultat(signals, noise_signal=nois_signal, fs=FS, noise_gain=noise_gain, absorption=absorbtion)
                mics = simu.get_result()         

                track_dir = os.path.join(case_dir, "sim_"+str(j))
                if not os.path.exists(track_dir):
                    os.mkdir(track_dir)
                mics = mics / np.abs(mics).max()  
                for i in range(mics.shape[0]):
                    sf.write(os.path.join(track_dir, f"mixture_mic_{i+1}.wav"), mics[i], FS)

                metadata_path = os.path.join(track_dir, "metadata.json")
                json_data = {
                    "track_id": track_id,
                    "absorbtion": absorbtion,
                    "speakers_count": n_speaker,
                    "noise": noise,

                }
        
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)
            case_ += 1

