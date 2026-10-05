import numpy as np
import pyroomacoustics as pra


class GenerationResultat:
    def __init__(self, signals, noise_signal=None, fs=16000,
                 absorption=0.3, max_order=10, noise_gain=0.1):
        """
        signals: liste de signaux (numpy arrays) à mélanger, un par personne, sans recouvrement temporel

        noise_signal: signal de bruit à ajouter (numpy array), si on en veut un, qui doit faire la même longueur que les signaux
        fs: fréquence d'échantillonnage
        absorption: coefficient d'absorption des murs
        max_order: ordre maximum de réflexion
        noise_gain: gain du signal de bruit
        """

        n_speakers = len(signals)
        assert n_speakers in (2, 3)
        assert len({len(s) for s in signals}) == 1, "les signaux doivent avoir la même longueur"
        assert noise_signal is None or len(noise_signal) == len(signals[0]), "le signal de bruit doit avoir la même longueur que les signaux"

        self.fs = fs
        self.room = pra.ShoeBox([4, 6], fs=fs, materials=pra.Material(absorption),
                                max_order=max_order)

        centre = np.array([2.0, 3.0])
        self.positions = [centre + 1.5 * np.array([np.cos(2*np.pi*i/n_speakers),
                                                   np.sin(2*np.pi*i/n_speakers)])
                          for i in range(n_speakers)]


        for pos, x in zip(self.positions, signals):
            self.room.add_source(pos, signal=x)

        # Bruit 
        if noise_signal is not None:
            n = np.asarray(noise_signal, dtype=np.float64)
            n = noise_gain * n / np.abs(n).max()  
            self.room.add_source([3.9, 5.9], signal=n)

        # 3 micros en cercle, rayon 10 cm, à 0°, 120°, 240°
        R = pra.circular_2D_array(centre, 3, 0.0, 0.1)
        self.room.add_microphone_array(pra.MicrophoneArray(R, fs))

    def get_result(self):
        self.room.simulate()
        return self.room.mic_array.signals        # shape (3, n_samples)

if __name__ == "__main__":  
    #juste un test de la classe GenerationResultat avant d'avoir l'autre partie du code
    import matplotlib.pyplot as plt

    fs = 16000
    t = np.linspace(0, 1, fs)
    s1 = np.sin(2 * np.pi * 440 * t)  # signal 1: sinusoïde à 440 Hz
    s2 = np.sin(2 * np.pi * 550 * t)  # signal 2: sinusoïde à 550 Hz
    noise = np.random.normal(0, 1, fs)  # bruit blanc

    gen = GenerationResultat([s1, s2], noise_signal=noise, fs=fs)
    result = gen.get_result()

    plt.figure()
    plt.plot(result[0], label='Microphone 1')
    plt.plot(result[1], label='Microphone 2')
    plt.plot(result[2], label='Microphone 3')
    plt.legend()
    plt.title('Signaux captés par les microphones')
    plt.xlabel('Échantillons')
    plt.ylabel('Amplitude')
    plt.show()