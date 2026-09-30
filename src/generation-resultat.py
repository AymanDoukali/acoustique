import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra

class generation_resultat:
    def __init__(self, number_of_spekers: int, Noise: bool, RIR, signal_noise, speech_path):

        self.signal_noise = signal_noise
        self.speech_path = speech_path
        self.RIR = RIR

        # Create a 4 by 6 metres shoe box room
        self.room = pra.ShoeBox([4,6])


        # Add a number_of_speaker source 1.5m auround the center of the room, equaly reparted
        self.room.add_source([2 + 1.5 * np.cos(2 * np.pi * i / number_of_spekers), 3 + 1.5 * np.sin(2 * np.pi * i / number_of_spekers)] for i in range(number_of_spekers))

        # Add a noise if noise == True at the corner of the room
        if Noise:
            self.room.add_sound_source([3.9,5.9], signal=self.signal_noise)

        # Create an array beamformer with 3 microphones
        # with angle 0, 120 and 240 degrees and distance to the center of the room 10 cm
        R = pra.linear_2D_array([2, 3], 3, [0, 120, 240], 0.1)
        self.room.add_microphone_array(pra.Beamformer(R, self.room.fs))

        # Now compute the delay and sum weights for the beamformer
        self.room.mic_array.rake_delay_and_sum_weights(self.room.sources[0][:1])

    def plot_room(self):
        # plot the room and resulting beamformer
        self.room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
        plt.show()