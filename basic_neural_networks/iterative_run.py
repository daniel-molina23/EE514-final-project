import numpy as np
from scipy.optimize import minimize
import os
import math

def get_optimal_half_matrix(num_layers, addHalf=False):
    n_qubits = 6
    size_of_vec = 2**n_qubits

    from utils import DFS
    inputStates, expectedStates = DFS().getInitialTargetStates()

    from utils import MatrixUtils
    matrixUtils = MatrixUtils()
    # Define the correct operations you want the matrix to perform on basis vectors
    def target_operations(parameters, inputStates):
        # Reshape the parameters into the matrix form
        normalParams, halfParams = parameters[:num_layers*5], parameters[num_layers*5:] if addHalf else None

        parameters = np.reshape(normalParams, (num_layers, 5))
        matrix = matrixUtils.get_total_matrix(size_of_vec, weights=parameters, halfWeights=halfParams)

        # Perform matrix multiplication with basis vectors
        results = []
        for i in range(len(inputStates)):
            results.append(np.matmul(matrix, inputStates[i]))

        # Define the target operations you want (modify this based on your specific task)
        target_result = np.array(expectedStates)

        # Calculate the loss as the difference between the obtained result and the target result
        loss = matrixUtils.f_cnot_loss(target_result, results)
        return loss

    # Generate random basis vectors and target result
    basis_vectors = np.array(inputStates)

    # Flatten the matrix parameters for optimization
    initial_parameters = np.ndarray.flatten(matrixUtils.get_random_weights(num_layers))

    initial_parameters = np.concatenate((initial_parameters, matrixUtils.get_random_half_layer_weights())) if addHalf else initial_parameters

    # Use scipy's minimize function to optimize the parameters
    result = minimize(target_operations, initial_parameters, args=(basis_vectors,), method='L-BFGS-B')

    firstHalf, secondHalf = result.x[:num_layers*5], result.x[num_layers*5:] if addHalf else None
    # Reshape the optimized parameters back into the matrix form
    optimized_matrix = matrixUtils.get_total_matrix(size_of_vec=2**6, weights=firstHalf.reshape((num_layers, 5)), halfWeights=secondHalf)

    # print("Optimized Matrix:")
    # print(optimized_matrix)
    total_layers = num_layers + 0.5 if addHalf else num_layers

    predStates = [np.matmul(optimized_matrix, mat) for mat in inputStates]
    fcnot_loss = matrixUtils.f_cnot_loss(expectedStates, predStates)
    print(f"f_cnot_loss for {total_layers} layers = {fcnot_loss}")
    print(f"square_loss for {total_layers} layers = {matrixUtils.square_loss(expectedStates, predStates)}")
    return optimized_matrix, result.x, fcnot_loss

def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def __main__():
    # layers_arr = []
    # f_loss_arr = []
    # params_arr = []
    # magnitudes = []
    # for layers in range(3, 12, 1):
    #     for addHalf in [False, True]:
    #         l = layers + 0.5 if addHalf else layers
    #         currentPath = os.getcwd()
    #         directory = os.path.join(currentPath, f'layers_{l}')
    #         make_dir_if_not_exists(directory)
    #         layers_arr.append([])
    #         f_loss_arr.append([])
    #         params_arr.append([])
    #         magnitudes.append([])
    #         for i in range(1,21,1):
    #             layers_arr[-1].append(l)
    #             m, p, f_loss = get_optimal_half_matrix(num_layers=layers, addHalf=addHalf)
    #             f_loss_arr[-1].append(f_loss)
    #             params_arr[-1].append(p)
    #             path = os.path.join(directory, f'model_{i}.txt')
    #             np.save(path, m)
    #             magnitude = math.floor(math.log10(f_loss))
    #             magnitudes[-1].append(magnitude)
    #             print(f"Trained model with layers: {l}, itertation: {i}, fcnot_loss: {f_loss}")
    print(get_optimal_half_matrix(num_layers=2, addHalf=True))


if __name__ == "__main__":
    __main__()