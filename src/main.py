import os
from data_generator import genrate_data
import os


if __name__ =="__main__":
    genrate_data(output_dir=os.path.join("data", "generated", "dataset"), 
                 fs=16000, target_duration=60, absorbtions=[0.1,0.2,0.3], 
                 noise_gain=0.1, n_sim=10, 
                 noise_type="fan")
