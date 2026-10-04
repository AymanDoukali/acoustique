import numpy as np

import pyroomacoustics as pra

class generation_resultat:
    def __init__(self, number_of_spekers: int, Noise: bool, RIR, signal_noise, speechs_path :list):

        self.signal_noise = signal_noise
        self.speechs_path = speechs_path
        self.RIR = RIR
        self.number_of_spekers = number_of_spekers

        # Create a 4 by 6 metres shoe box room
        self.room = pra.ShoeBox([4,6])
        self.room.rir = self.RIR


        # Add a number_of_speaker source 1.5m auround the center of the room, equaly reparted
        self.sound_sources_positions = [[2 + 1.5 * np.cos(2 * np.pi * i / number_of_spekers), 3 + 1.5 * np.sin(2 * np.pi * i / number_of_spekers)] for i in range(number_of_spekers)]
        self.sound_signal=[pra.normalize(pra.load(speech_path)) for speech_path in self.speechs_path]
        for i in range(number_of_spekers):
            self.room.add_sound_source(self.sound_sources_positions[i], signal=self.sound_signal[i])

        # Add a noise if noise == True at the corner of the room
        if Noise:
            self.room.add_sound_source([3.9,5.9], signal=self.signal_noise)

        # Create an array beamformer with 3 microphones
        # with angle 0, 120 and 240 degrees and distance to the center of the room 10 cm
        R = pra.linear_2D_array([2, 3], 3, [0, 120, 240], 0.1)
        self.room.add_microphone_array(pra.Beamformer(R, self.room.fs))
        

    def get_result(self):
        # Compute the resulting signal
        self.room.simulate()
        return self.room.mic_array.signals