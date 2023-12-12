from pennylane import numpy as np
import pennylane as qml
import numpy as npy
class Ansatz():
    def __init__(self, num_layers=30, seed=np.random.randint(10, 10000)):
        np.random.seed(seed)
        self.num_layers = num_layers
        self.n_qubits = 6
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        self.weights = self.get_random_weights()
        # functions
        self.amp_sqrd = lambda c : npy.real(c*npy.conjugate(c))
        self.Udot = lambda s1, U, s2 : npy.dot(npy.conjugate(npy.transpose(s1)),npy.matmul(U,s2))

    def get_random_weights(self):
        return 2 * np.pi * np.random.random(size=(self.num_layers, 5), requires_grad=True) - np.pi # range from -pi to pi

    def U_ex(self, p):
        from scipy.linalg import expm
        X = [[0,1],[1,0]]
        Y = npy.array([[0,-1j],[1j,0]], dtype=npy.complex128)
        Z = [[1,0],[0,-1]]

        H_ex = (1/4)*(npy.kron(X,X) + npy.kron(Y,Y) + npy.kron(Z,Z))
        # print(f'H_ex.type = {type(H_ex)}')
        U_exchange = expm(-1j*p*H_ex) # p is now -pi to pi
        return np.array(U_exchange, requires_grad=False)

    def single_layer(self, layer_weights):
        """Trainable circuit block."""
        qml.QubitUnitary(self.U_ex(layer_weights[0]), wires=[0,1])
        qml.QubitUnitary(self.U_ex(layer_weights[1]), wires=[2,3])
        qml.QubitUnitary(self.U_ex(layer_weights[2]), wires=[4,5])
        #
        qml.QubitUnitary(self.U_ex(layer_weights[3]), wires=[1,2])
        qml.QubitUnitary(self.U_ex(layer_weights[4]), wires=[3,4])

    def Unitary_CNOT(self, weights, inputStateVector):
        qml.AmplitudeEmbedding(inputStateVector, wires=range(self.n_qubits))
        for layer_weights in weights:
            self.single_layer(qml.math.toarray(layer_weights))
    

    def quantum_model(self, weights, inputStateVector):
        self.Unitary_CNOT(weights, inputStateVector)
        return qml.state()
    
    def draw_circuit(self, inputStates):
        print(qml.draw(self.quantum_model)(self.weights, inputStates))

    def get_predictions(self, weights, inputStates):
        qnode_ = qml.QNode(self.quantum_model, self.dev, interface='autograd')
        outputStates = []
        for i in range(len(inputStates)):
            newState = qnode_(weights, inputStates[i])
            outputStates.append(newState)
        return np.array(outputStates, requires_grad=False)

    def get_density_mat_predictions(self, weights, inputStates):
        outputMatrices = []
        for i in range(len(inputStates)):
            newMatrix = qml.matrix(self.Unitary_CNOT)(weights, inputStates[i])
            outputMatrices.append(newMatrix)
        return np.array(outputMatrices, requires_grad=False)
    
    def square_loss(self, expectedStates, predictedStates):
        loss = 0
        for i in range(len(expectedStates)):
            # c = npy.dot(npy.conjugate(expectedStates[i]), predictedStates[i])
            # c_2 = self.amp_sqrd(c)
            fidelity = qml.math.fidelity_statevector(expectedStates[i], predictedStates[i])
            loss += (1 - fidelity) ** 2
        loss /= len(expectedStates)
        return 0.5*loss
    
    def square_dm_loss(self, expectedStates, predictedStates):
        loss = 0
        for i in range(len(expectedStates)):
            st = npy.outer(expectedStates[i],npy.conjugate(expectedStates[i]))
            fidelity = qml.math.fidelity(st, predictedStates[i])
            loss += (1 - fidelity) ** 2
        loss /= len(expectedStates)
        return 0.5*loss
    
    def loss_function(self, expectedStates, predictedStates):
        values = []
        for i in range(len(expectedStates)):
            c = npy.dot(npy.conjugate(expectedStates[i]), predictedStates[i])
            c_2 = self.amp_sqrd(c)
            values.append(c_2)
        return npy.sqrt(1 - (1/4)*sum(values))
    
    def cost(self, weights, inputStates, expectedStates):
        predictedStates = self.get_predictions(weights, inputStates)
        return self.square_loss(expectedStates, predictedStates)
        # predictedDensityMatrices = self.get_density_mat_predictions(weights, inputStates)
        # return self.square_dm_loss(expectedStates, predictedDensityMatrices)
    
    def train(self, inputStates, expectedStates, max_steps=80, alpha=0.1):
        opt = qml.AdamOptimizer(stepsize=alpha, beta1=0.9, beta2=0.99, eps=1e-08)
        # opt = qml.AdagradOptimizer(stepsize=alpha, eps=1e-08)
        # opt = qml.GradientDescentOptimizer(alpha)

        self.weights = self.get_random_weights()

        cst = [self.cost(self.weights, inputStates, expectedStates)]  # initial cost

        for step in range(max_steps):

            # Update the weights by one optimizer step
            self.weights, _, _ = opt.step(self.cost, self.weights, inputStates, expectedStates)

            # Save current cost
            c = self.cost(self.weights, inputStates, expectedStates)
            cst.append(c)
            # if (step + 1) % 10 == 0:
            print("Cost at step {0:3}: {1}".format(step + 1, c))
    