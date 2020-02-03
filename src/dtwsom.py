import math
import random
import numpy as np
import warnings
from dtaidistance import dtw_ndim, dtw
from scipy.spatial import distance

try:
    import matplotlib.pyplot as plt
except Exception as error_instance:
    warnings.warn(
        "Impossible to import matplotlib (please, install 'matplotlib'), pyclustering's visualization "
        "functionality is not available (details: '%s')." % str(error_instance)
    )

import pyclustering.core.som_wrapper as wrapper
from pyclustering.nnet.som import type_conn, type_init, som, som_parameters
from pyclustering.utils.dimension import dimension_info
from enum import IntEnum


class DtwTypeInit(IntEnum):
    """!
    @brief Enumeration of initialization types for DTW SOM.

    @see som

    """

    ## Neurons are initialized as a random sample of the input data
    random_sample = 0

    ## Anchors?
    anchors = 1


class DtwSomParameters:
    """!
    @brief Represents DTW SOM parameters.

    """

    def __init__(self):
        """!
        @brief Constructor container of SOM parameters.

        """

        ## Type of initialization of initial neuron weights (random, random in center of the input data, random
        # distributed in data, ditributed in line with uniform grid).
        self.init_type = DtwTypeInit.random_sample

        ## Initial radius (if not specified then will be calculated by SOM).
        self.init_radius = None

        ## Rate of learning.
        self.init_learn_rate = 0.1

        ## Condition when learining process should be stoped. It's used when autostop mode is used.
        self.adaptation_threshold = 0.001


class DtwParameters:
    """!
    @brief Represents main parameters for DTW distance.
    """

    def __init__(self, window=None, max_dist=None, max_step=None, max_length_diff=None):
        """!
        @brief Constructor container of DTW parameters.

        """
        self.window = window
        self.max_dist = max_dist
        self.max_step = max_step
        self.max_length_diff = max_length_diff


class DtwSom(som):
    """!
        @brief Represents self-organized feature map (SOM) with the Dynamic Time Warping (DTW) distance.
        @details The self-organizing feature map (SOM) method is a powerful tool for the visualization of
                 of high-dimensional data. It converts complex, nonlinear statistical relationships between
                 high-dimensional data into simple geometric relationships on a low-dimensional display.
                 The DTW-SOM is then an adaptation of SOM for time-series subsequences with variable lengths
                 which used the Dynamic Time Warping (DTW) distance as the similarity metric.

        @details CCORE option can be used to use the pyclustering core - C/C++ shared library for processing that
        significantly increases performance.

        """

    def __init__(
        self,
        rows,
        cols,
        conn_type=type_conn.grid_eight,
        parameters=None,
        dtw_params=None,
    ):
        """!
        @brief Constructor of DTW self-organized map.

        @param[in] rows (uint): Number of neurons in the column (number of rows).
        @param[in] cols (uint): Number of neurons in the row (number of columns).
        @param[in] conn_type (type_conn): Type of connection between oscillators in the network (grid four, grid eight,
        honeycomb, function neighbour).
        @param[in] parameters (DtwSomParameters): Parameters for training the DTW SOM.
        @param[in] dtw_params (DtwParameters): Parameters specific for the DTW distance function.

        """
        super().__init__(rows, cols, conn_type, parameters, ccore=False)
        self.__ccore_som_pointer = None
        if parameters is None:
            init_radius = self._params.init_radius
            self._params = DtwSomParameters()
            self._params.init_radius = init_radius
        if dtw_params is not None:
            self._dtw_params = dtw_params
        else:
            self._dtw_params = DtwParameters()
        self.current_dtw_dic_list = None

    def __initialize_distances(self, size, location):
        """!
        @brief Initialize distance matrix in SOM grid.

        @param[in] size (uint): Amount of neurons in the network.
        @param[in] location (list): List of coordinates of each neuron in the network.

        @return (list) Distance matrix between neurons in the network.

        """
        sqrt_distances = [[[] for i in range(size)] for j in range(size)]
        for i in range(size):
            for j in range(i, size, 1):
                dist = distance.euclidean(location[i], location[j])
                sqrt_distances[i][j] = dist
                sqrt_distances[j][i] = dist

        return sqrt_distances

    # TODO: Fix "martelada" in Anchor init type
    def _create_initial_weights(self, init_type, anchors=None):
        """!
        @brief Creates initial weights for neurons in line with the specified initialization.

        @param[in] init_type (type_init): Type of initialization of initial neuron weights (random_sample or anchors).

        """
        if init_type == DtwTypeInit.random_sample:
            self._weights = random.sample(self._data, self._size)
        elif init_type == DtwTypeInit.anchors:
            if anchors is None:
                raise AttributeError("To chose the anchor initialization, you must provide a list of anchors")
            input_sample = random.sample(self._data, self._size)
            for i in range(self._rows):
                for j in range(self._cols):
                    neuron_index = i * self._cols + j
                    if i == j:
                        self._weights[neuron_index] = anchors[i]
                    else:
                        self._weights[neuron_index] = input_sample[i]

    def __initialize_bmu_distance_list(self):
        data_size = len(self._data)
        self._bmu_distance_list = [
            {"bmu": None, "dtw_dist": None} for i in range(data_size)
        ]

    def _set_dtw_path_and_distance_dic_list(self, x):
        """!
        @brief Calculates and sets the list of dictionaries with the DTW distance and best path between each neuron and
        the input pattern x. Thus the length of the list is equal to the size of the SOM network.

        @param[in] x (list): Input pattern from the input data set. It should be a list of arrays/floats, where each
        array/float is an observation of the one-dimensional/multidimensional time-series window given as input pattern.
        """
        dtw_dic_list = []

        for neuron_index in range(self._size):
            neuron_dist, neuron_paths = dtw_ndim.warping_paths(
                self._weights[neuron_index],
                x,
                window=self._dtw_params.window,
                max_dist=self._dtw_params.max_dist,
                max_step=self._dtw_params.max_step,
                max_length_diff=self._dtw_params.max_length_diff,
            )
            best_path = dtw.best_path(neuron_paths)
            matching_dic = {i: [] for i in range(len(self._weights[neuron_index]))}
            for i, j in best_path:
                matching_dic[i].append(j)
            neuron_dic = {
                "index": neuron_index,
                "dist": neuron_dist,
                "matching_dic": matching_dic,
            }

            dtw_dic_list.append(neuron_dic)
        self.current_dtw_dic_list = dtw_dic_list

    def _competition(self, x):
        """!
        @brief Calculates neuron winner (distance, neuron index).

        @param[in] x (list): Input pattern from the input data set. It should be a list of arrays/floats, where each
        array/float is an observation of the one-dimensional/multidimensional time-series window given as input pattern.

        @return (uint) Returns index of neuron that is winner.
        """
        self._set_dtw_path_and_distance_dic_list(x)
        win_neuron_dic = min(
            self.current_dtw_dic_list, key=lambda neuron_dic: neuron_dic["dist"]
        )
        index = win_neuron_dic["index"]
        return index

    def _adaptation(self, winner_index, x):
        """!
        @brief Change weight of neurons in line with won neuron.

        @param[in] winner_index (uint): Index of neuron-winner.
        @param[in] x (list): Input pattern from the input data set. It should be a list of arrays/floats, where each
        array/float is an observation of the one-dimensional/multidimensional time-series window given as input pattern.
        """
        if self._conn_type == type_conn.func_neighbor:
            # Update all neurons within the _local_radius
            for neuron_index in range(self._size):
                self._update_weights(
                    winner_index, neuron_index, x, self.current_dtw_dic_list
                )

        else:
            # Update winner neuron
            self._update_weights(
                winner_index, winner_index, x, self.current_dtw_dic_list
            )

            # Update neighboring neurons if they are within the _local_radius
            for neighbor_index in self._neighbors[winner_index]:
                self._update_weights(
                    winner_index, neighbor_index, x, self.current_dtw_dic_list
                )

    def train(self, data, epochs, autostop=False, anchors=None):
        """!
        @brief Trains self-organized feature map (SOM).

        @param[in] data (list): Input data - list of points where each point is represented by list of features, for
        example coordinates.
        @param[in] epochs (uint): Number of epochs for training.
        @param[in] autostop (bool): Automatic termination of learining process when adaptation is not occurred.

        @return (uint) Number of learining iterations.

        """

        self._data = data

        self.__initialize_bmu_distance_list()

        if self.__ccore_som_pointer is not None:
            return wrapper.som_train(self.__ccore_som_pointer, data, epochs, autostop)

        self._sqrt_distances = self.__initialize_distances(self._size, self._location)

        for i in range(self._size):
            self._award[i] = 0
            self._capture_objects[i].clear()

        # weights
        self._create_initial_weights(self._params.init_type, anchors)

        previous_weights = None

        for epoch in range(1, epochs + 1):
            # Depression term of coupling
            self._local_radius = (
                self._params.init_radius * math.exp(-(epoch / epochs))
            ) ** 2
            self._learn_rate = self._params.init_learn_rate * math.exp(
                -(epoch / epochs)
            )

            # Clear statistics
            if autostop:
                for i in range(self._size):
                    self._award[i] = 0
                    self._capture_objects[i].clear()

            for i in range(len(self._data)):
                # Step 1: Competition:
                index = self._competition(self._data[i])

                # Step 2: Adaptation:
                self._adaptation(index, self._data[i])

                # Update statistics
                if (autostop == True) or (epoch == epochs):
                    self._award[index] += 1
                    self._capture_objects[index].append(i)

                # Update BMU dic
                self._bmu_distance_list[i]["bmu"] = index
                self._bmu_distance_list[i]["dtw_dist"] = self.current_dtw_dic_list[
                    index
                ]["dist"]

            # Compute the average quantization error and print (average quantization error and topographic error)
            avg_quantization_error = np.mean(
                [bmu_dic["dtw_dist"] for bmu_dic in self._bmu_distance_list]
            )
            print(
                "Epoch {} achieved an average quantization error of {}".format(
                    epoch, np.round(avg_quantization_error, 4)
                )
            )

            # Check requirement of stopping
            if autostop:
                if previous_weights is not None:
                    maximal_adaptation = self._get_maximal_adaptation(previous_weights)
                    if maximal_adaptation < self._params.adaptation_threshold:
                        return epoch

                previous_weights = [item[:] for item in self._weights]

        return epochs

    def _update_weights(self, winner_index, neighbor_index, x, dtw_dic_list):
        sqrt_distance = self._sqrt_distances[winner_index][neighbor_index]

        if sqrt_distance < self._local_radius:
            # if winner_index == neighbor_index, then distance = 0 and influence = 1
            influence = math.exp(-(sqrt_distance / (2.0 * self._local_radius)))

            # update weights based on dtw matchings
            neighbor_matching_dic = dtw_dic_list[neighbor_index]["matching_dic"]
            for i in range(len(self._weights[neighbor_index])):
                matching_values = [x[j] for j in neighbor_matching_dic[i]]
                mean_matching_value = np.mean(matching_values)
                self._weights[neighbor_index][i] = self._weights[neighbor_index][
                    i
                ] + self._learn_rate * influence * (
                    mean_matching_value - self._weights[neighbor_index][i]
                )

    def _get_maximal_adaptation(self, previous_weights):
        """!
        @brief Calculates maximum changes of weight in line with comparison between previous weights and current
        weights.

        @param[in] previous_weights (list): Weights from the previous step of learning process.

        @return (double) Value that represents maximum changes of weight after adaptation process.

        """

        maximal_adaptation = 0.0

        for neuron_index in range(self._size):
            dimension = len(previous_weights[neuron_index])
            for dim in range(dimension):
                current_adaptation = (
                    previous_weights[neuron_index][dim]
                    - self._weights[neuron_index][dim]
                )

                if current_adaptation < 0:
                    current_adaptation = -current_adaptation

                if maximal_adaptation < current_adaptation:
                    maximal_adaptation = current_adaptation

        return maximal_adaptation

    def get_distance_matrix(self):
        """!
        @brief Calculates distance matrix (U-matrix).
        @details The U-Matrix visualizes based on the distance in input space between a weight vector and its neighbors
        on map.

        @return (list) Distance matrix (U-matrix).

        @see show_distance_matrix()
        @see get_density_matrix()

        """
        distance_matrix = [[0.0] * self._cols for i in range(self._rows)]

        for i in range(self._rows):
            for j in range(self._cols):
                neuron_index = i * self._cols + j

                if self._conn_type == type_conn.func_neighbor:
                    self._create_connections(type_conn.grid_eight)

                for neighbor_index in self._neighbors[neuron_index]:
                    distance_matrix[i][j] += dtw_ndim.distance(
                        self._weights[neuron_index],
                        self._weights[neighbor_index],
                        window=self._dtw_params.window,
                        max_dist=self._dtw_params.max_dist,
                        max_step=self._dtw_params.max_step,
                        max_length_diff=self._dtw_params.max_length_diff,
                    )

                distance_matrix[i][j] /= len(self._neighbors[neuron_index])

        return distance_matrix

    def show_distance_matrix(self):
        """!
        @brief Shows gray visualization of U-matrix (distance matrix).

        @see get_distance_matrix()

        """
        distance_matrix = self.get_distance_matrix()

        plt.imshow(
            distance_matrix, cmap=plt.get_cmap("inferno_r"), interpolation="kaiser"
        )
        plt.title("U-Matrix")
        plt.colorbar()
        plt.show()

    # TODO: Implement get_density_matrix function
    def get_density_matrix(self, surface_divider=20.0):
        """!
        @brief Calculates density matrix (P-Matrix).

        @param[in] surface_divider (double): Divider in each dimension that affect radius for density measurement.

        @return (list) Density matrix (P-Matrix).

        @see get_distance_matrix()

        """
        raise NotImplementedError(
            "The function `get_density_matrix` is not yet implemented in the `DtwSom` class"
        )