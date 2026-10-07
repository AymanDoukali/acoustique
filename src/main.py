import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from audiogen import audiogen                   
from generationresultat import GenerationResultat  

FS = 16000


gen_audio = audiogen()
track_id, signals = gen_audio.generate(target_duration_sec=60.0)

noise = np.random.normal(0, 1, len(signals[0]))

simu = GenerationResultat(signals, noise_signal=noise, fs=FS)
mics = simu.get_result()         

track_dir = os.path.join(gen_audio.output_base_dir, track_id)
mics = mics / np.abs(mics).max()  
for i in range(mics.shape[0]):
    sf.write(os.path.join(track_dir, f"mixture_mic_{i+1}.wav"), mics[i], FS)

sf.write(os.path.join(track_dir, "noise.wav"), noise, FS)

plt.figure()
plt.plot(mics[0], label='Microphone 1')
plt.plot(mics[1], label='Microphone 2')
plt.plot(mics[2], label='Microphone 3')
plt.legend()
plt.title('Signaux captés par les microphones')
plt.xlabel('Échantillons')
plt.ylabel('Amplitude')
plt.show()