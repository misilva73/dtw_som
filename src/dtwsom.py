import math
import copy
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

from pyclustering.nnet.som import type_conn, som
from enum import IntEnum


class DtwTypeInit(IntEnum):
    """!
    @brief Enumeration of initialization types for DTW-SOM.

    """
    # Neurons are initialized as a random sample of the input data
    random_sample = 0
    # Anchors?
    anchors = 1


class DtwSomParameters:
    """!
    @brief Represents DTW-SOM parameters.

    """

    def __init__(self):
        """!
        @brief Constructor container of DTW-SOM parameters.

        """
        # Type of initialization of initial neuron weights (random, random in center of the input data, random
        # distributed in data, distributed in line with uniform grid).
        self.init_type = DtwTypeInit.random_sample
        # Initial radius (if not specified then will be calculated by SOM).
        self.init_radius = None
        # Rate of learning.
        self.init_learn_rate = 0.1
        # Condition when learning process should be stopped. It's used when autostop mode is used.
        self.adaptation_threshold = 0.001


class DtwParameters:
    """!
    @brief Represents main parameters for DTW distance.
    """

    def __init__(self, window=None, max_step=None, max_length_diff=None):
        """!
        @brief Constructor container of DTW parameters.

        """
        self.window = window
        self.max_dist = None
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
        @brief Initialize distance matrix in DTW-SOM grid.

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

    def _create_initial_weights(self, init_type, anchors=None):
        """!
        @brief Creates initial weights for neurons in line with the specified initialization.

        @param[in] init_type (type_init): Type of initialization of initial neuron weights (random_sample or anchors).

        """
        if init_type == DtwTypeInit.random_sample:
            self._weights = random.sample(self._data, self._size)

        elif init_type == DtwTypeInit.anchors:
            if anchors is None:
                raise AttributeError(
                    "To chose the anchor initialization, you must provide a non-empty list of anchors"
                )
            n_anchors = len(anchors)
            max_square = min(self._rows, self._cols)
            n_diagonals = 2 * max_square - 1
            # No anchors : raise error
            if n_anchors == 0:
                raise AttributeError(
                    "To chose the anchor initialization, you must provide a non-empty list of anchors"
                )
            # Anchors < units : fill the diagonals, then fill the rest with anchors plus a sample from input data
            if n_anchors < self._size:
                remaining_data = [
                    obs
                    for obs in self._data
                    if not self.__np_is_contained_in(obs, anchors)
                ]
                data_sample = random.sample(remaining_data, self._size - n_anchors)
                if n_anchors < n_diagonals:
                    diagonals_list = anchors + data_sample[: n_diagonals - n_anchors]
                    others_list = data_sample[n_diagonals - n_anchors:]
                else:
                    diagonals_list = anchors[:n_diagonals]
                    others_list = anchors[n_diagonals:] + data_sample
            # Anchors >= units : fill the diagonals, then fill the rest with anchors, raise warning of unused anchors
            else:
                diagonals_list = anchors[:n_diagonals]
                others_list = anchors[n_diagonals:]
                if n_anchors > self._size:
                    warnings.warn(
                        "Provided list contains more anchors than units. On the first are used to "
                        "initialize the network"
                    )
            random.shuffle(diagonals_list)
            random.shuffle(others_list)
            self.__fill_weights_with_anchors(max_square, diagonals_list, others_list)
        else:
            raise AttributeError("The provided initialization type is not supported")
        self._weights = copy.deepcopy(self._weights)

    @staticmethod
    def __np_is_contained_in(obs, obs_lis):
        for other_obs in obs_lis:
            if np.array_equal(obs, other_obs):
                return True
        return False

    def __fill_weights_with_anchors(self, max_square, diagonals_list, others_list):
        for i in range(self._rows):
            for j in range(self._cols):
                neuron_index = i * self._cols + j
                if i == j and (2 * j != max_square - 1):
                    self._weights[neuron_index] = diagonals_list[i]
                elif i == max_square - 1 - j:
                    self._weights[neuron_index] = diagonals_list[i + max_square - 1]
                else:
                    self._weights[neuron_index] = others_list.pop(0)

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
        @brief Calculates the neuron winner (i.e., the BMU of x) and returns its index.

        @param[in] x (list): Input pattern from the input data set. It should be a list of arrays/floats, where each
        array/float is an observation of the one-dimensional/multidimensional time-series window given as input pattern.

        @return (uint) Returns index of neuron that is winner.
        """
        self._set_dtw_path_and_distance_dic_list(x)
        win_neuron_dic = min(
            self.current_dtw_dic_list, key=lambda neuron_dic: neuron_dic["dist"]
        )
        # Make sure that we always have a BMU by increasing self._dtw_params.max_dist
        initial_max_bmu_dist = self._dtw_params.max_dist
        while not win_neuron_dic["dist"] < np.inf:
            self._dtw_params.max_dist = self._dtw_params.max_dist*1.05
            self._set_dtw_path_and_distance_dic_list(x)
            win_neuron_dic = min(
                self.current_dtw_dic_list, key=lambda neuron_dic: neuron_dic["dist"]
            )
        self._dtw_params.max_dist = initial_max_bmu_dist
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
        @brief Trains DTW self-organized feature map (DTW-SOM).

        @param[in] data (list): Input data - list of points where each point is represented by list of features, for
        example coordinates.
        @param[in] epochs (uint): Number of epochs for training.
        @param[in] autostop (bool): Automatic termination of learning process when adaptation is not occurred.
        @param[in] anchors (list): List of input patterns that should be considered as anchors. Only used for the
        anchor initialization.

        @return (uint) Number of learning iterations.

        """

        self._data = data

        self.__initialize_bmu_distance_list()

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
                if (autostop is True) or (epoch == epochs):
                    self._award[index] += 1
                    self._capture_objects[index].append(i)

                # Update BMU dic
                self._bmu_distance_list[i]["bmu"] = index
                self._bmu_distance_list[i]["dtw_dist"] = self.current_dtw_dic_list[
                    index
                ]["dist"]

            # Update the dtw max_dist
            max_bmu_dist = max(
                self._bmu_distance_list, key=lambda bmu_dic: bmu_dic["dtw_dist"]
            )["dtw_dist"]
            self._dtw_params.max_dist = max_bmu_dist * 1.1

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
        is_multi_dim = isinstance(x[0], np.ndarray)
        if is_multi_dim:
            n_dim = len(x[0])

        sqrt_distance = self._sqrt_distances[winner_index][neighbor_index]

        if sqrt_distance < self._local_radius:
            # if winner_index == neighbor_index, then distance = 0 and influence = 1
            influence = math.exp(-(sqrt_distance / (2.0 * self._local_radius)))

            # update weights based on dtw matchings
            neighbor_matching_dic = dtw_dic_list[neighbor_index]["matching_dic"]
            for i in range(len(self._weights[neighbor_index])):
                matching_values = [x[j] for j in neighbor_matching_dic[i]]
                if len(matching_values) > 0:
                    if is_multi_dim:
                        mean_matching_value = np.zeros(n_dim)
                        for k in range(n_dim):
                            k_values = [val[k] for val in matching_values]
                            mean_matching_value[k] = np.mean(k_values)
                    else:
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
            np.array(distance_matrix).T, cmap=plt.get_cmap("inferno_r"), interpolation="kaiser"
        )
        plt.title("U-Matrix")
        plt.colorbar()
        plt.show()

    def save_distance_matrix(self, output_file):
        distance_matrix = self.get_distance_matrix()

        plt.imshow(
            np.array(distance_matrix).T, cmap=plt.get_cmap("inferno_r"), interpolation="kaiser"
        )
        plt.title("U-Matrix")
        plt.colorbar()
        plt.savefig(output_file)

    def save_winner_matrix(self, output_file):
        (fig, ax) = plt.subplots()
        winner_matrix = [[0] * self._cols for i in range(self._rows)]
        for i in range(self._rows):
            for j in range(self._cols):
                neuron_index = i * self._cols + j

                winner_matrix[i][j] = self._award[neuron_index]
                ax.text(i, j, str(winner_matrix[i][j]), va='center', ha='center')
        ax.imshow(np.array(winner_matrix).T, cmap=plt.get_cmap('cool'), interpolation='none')
        ax.grid(True)
        plt.title("Winner Matrix")
        plt.savefig(output_file)

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
