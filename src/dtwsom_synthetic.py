import random
import numpy as np
import dtwsom
from pyclustering.nnet.som import type_conn


# Functions
def gen_noisy_sine_list(f0, fs, mean_dur, size):
    final_list = []
    for i in range(size):
        dur = random.sample([mean_dur-1, mean_dur, mean_dur+1], 1)[0]
        t = np.arange(dur)
        sinusoid = np.sin(2*np.pi*t*(f0/fs))
        noise = np.random.normal(0,0.3, dur)
        noisy_sinusoid = noise + sinusoid
        final_list.append(noisy_sinusoid)
    return final_list


def gen_noisy_list(mean_dur, size):
    final_list = []
    for i in range(size):
        dur = random.sample([mean_dur-1, mean_dur, mean_dur+1], 1)[0]
        noise = np.random.normal(0,0.3, dur)+10
        final_list.append(noise)
    return final_list


# Generate dummy dataset
sin_dataset = gen_noisy_sine_list(1, 10, 25, 50) + gen_noisy_list(20, 50)
anchors = [sin_dataset[0], sin_dataset[-1]]
random.shuffle(sin_dataset)
len(sin_dataset)


# Set DTW-SOM network
rows = 3
cols = 3
structure = type_conn.grid_four
network = dtwsom.DtwSom(rows, cols, structure)

# Train DTW-SOM
network.train(sin_dataset, 20)

# Visualise network
network.show_distance_matrix()
network.show_winner_matrix()

