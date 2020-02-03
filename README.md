# DTW-SOM: Self-organizing map for time-series data

The method DTW-SOM (Dynamic Time Warping Self-Organizing Map) was built for the paper 
*Exploring time-series motifs through DTW-SOM*. In the [github repository](https://github.com/misilva73/dtw_som), you'll
 find all the work presented in the paper.
 
DTW-SOM is a vanilla Self-Organizing Map with three main differences, namely (1) the use the Dynamic Time Warping 
distance instead of the Euclidean distance, (2) the adoption of two new network initialization routines (a random sample 
initialization and an anchor initialization) and (3) the adjustment of the Adaptation phase of the training to work with 
variable-length time-series sequences.
 
In the paper, we argue that visually exploring time-series motifs computed by motif discovery algorithms can be useful
 to understand and debug results and we propose the use of DTW-SOM on the list of motif’s centers to conduct these 
explorations. We then test DTW-SOM in a synthetic motif dataset and a real time-series dataset called GunPoint. After an 
exploration of results, we conclude that DTW-SOM is capable of extracting relevant information from a set of motifs and 
display it in a visualization that is space-efficient.

## Github project structure

    ├── notebooks          <- Jupyter notebooks use to test dtwsom and to run the anlsysis for the paper (including the plots)
    ├── paper              <- Folder the the PDF and latex project for the paper
    ├── src                <- Folder with the dtwsom module
    ├── README.md          <- The top-level README for this project
    ├── requirements.txt   <- The requirements file for reproducing the environment used in the paper

## Prerequisites

    dtaidistance==1.2.3
    matplotlib==3.1.2
    numpy==1.18.1
    pyclustering==0.9.3.1
    scipy==1.4.1

## Installing

This packages is available on PyPI and thus can be directly installed with pip:

```bash
pip install dtw_som
```

Alternatively, this package can installed from source by cloning this repository and installing it manually with the 
command:

```bash
python setup.py install
```

## Example Code

Import packages and generate a dummy dataset with 2 clusters, a noisy sine curve and a noise line centered at 10:

```python
import dtw_som
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pyclustering.nnet.som import type_conn


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


sin_dataset = gen_noisy_sine_list(1, 10, 25, 50) + gen_noisy_list(20, 50)
random.shuffle(sin_dataset)
```

Define and train the network:

```python
rows = 3
cols = 3
structure = type_conn.grid_four
network = dtw_som.DtwSom(rows, cols, structure)

network.train(sin_dataset, 20)
```

After training, you can visualise the U-matrix the Winner matrix:

```python
network.show_distance_matrix()
network.show_winner_matrix()
```

Finally, you can also visualize the each unit as a time-series:

```python
n_neurons = network._size
fig, axs = plt.subplots(3,3,figsize=(20, 10), sharey=True)
for neuron_index in range(n_neurons):
    col = math.floor(neuron_index/3)
    row = neuron_index % 3
    neuron_weights = network._weights[neuron_index]
    axs[row, col].plot(np.arange(len(neuron_weights)), neuron_weights, label=str(neuron_index))
    axs[row, col].set_ylabel("Neuron: "+str(neuron_index))
plt.show()
```

To confirm the output of this example, check the following 
[notebook](https://github.com/misilva73/dtw_som/blob/master/notebooks/testing_dtwsom.ipynb).