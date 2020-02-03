# DTW-SOM: Self-organizing map for time-series data

This repository aims to support the work presented in paper *Exploring time-series motifs through DTW-SOM*.

In this paper, we argue that visually exploring time-series motifs computed by motif discovery algorithms can be useful
to understand and debug results and we propose the use of an adapted Self-Organizing Map, the DTWSOM, on the list of 
motif’s centers to conduct these explorations.

DTW-SOM is a vanilla Self-Organizing Map with three main differences, namely (1) the use the Dynamic Time Warping 
distance instead of the Euclidean distance, (2) the adoption of two new network initialization routines (a random sample 
initialization and an anchor initialization) and (3) the adjustment of the Adaptation phase of the training to work with 
variable-length time-series sequences.

We test DTW-SOM in a synthetic motif dataset and a real time-series dataset called GunPoint. After an exploration of 
results, we conclude that DTW-SOM is capable of extracting relevant information from a set of motifs and display it in a 
visualization that is space-efficient.

## Project structure

    ├── notebooks          <- Jupyter notebooks use to test dtwsom and to run the anlsysis for the paper (including the plots)
    ├── paper              <- Folder the the PDF and latex project for the paper
    ├── src                <- Folder with the dtwsom module
    ├── README.md          <- The top-level README for this project
    ├── requirements.txt   <- The requirements file for reproducing the environment used in the paper
