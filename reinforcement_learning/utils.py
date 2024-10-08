import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


######################################################################################



import pennylane as qml


class RL_Util():
    def __init__(self):
        self.inputStates, self.expectedStates = DFS().getInitialTargetStates()
        self.matrixUtils = MatrixUtils()
        pass
    # Define the correct operations you want the matrix to perform on basis vectors
    def cost_func(self, parameters, num_layers):
        # Reshape the parameters into the matrix form
        parameters = np.reshape(parameters, (num_layers, 5))
        matrix = self.matrixUtils.get_total_matrix(size_of_vec=2**6, weights=parameters)

        # Perform matrix multiplication with basis vectors
        results = []
        for i in range(len(self.inputStates)):
            results.append(np.matmul(matrix, self.inputStates[i]))

        # Define the target operations you want (modify this based on your specific task)
        target_result = np.array(self.expectedStates)

        # Calculate the loss as the difference between the obtained result and the target result
        # loss = square_loss(target_result, results)
        loss = self.matrixUtils.f_cnot_loss(target_result, results)
        return loss

class MatrixUtils():
    def __init__(self):
        pass
    def Udot(s1,U,s2):
        return np.dot(np.conjugate(np.transpose(s1)),np.matmul(U,s2))

    def nestedKron(self, *args): # use "*args" to access an array of inputs
        assert len(args) >= 2
        temp = args[0]
        for arg in args[1:]:
            temp = np.kron(temp, arg)
        return temp

    def get_random_weights(self, num_layers):
        return 2 * np.random.random(size=(num_layers, 5)) - 1

    def U_ex(self, p):
        from scipy.linalg import expm
        X = [[0,1],[1,0]]
        Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
        Z = [[1,0],[0,-1]]

        H_ex = (1/4)*(np.kron(X,X) + np.kron(Y,Y) + np.kron(Z,Z))
        # print(f'H_ex.type = {type(H_ex)}')
        U_exchange = expm(-1j*p*H_ex) # p is from -pi to +pi
        return np.array(U_exchange)

    def get_predictions(self, inputStates, weights):
        matOp = self.get_total_matrix(weights)
        results = []
        for i in range(len(inputStates)):
            results.append(np.matmul(matOp, inputStates[i]))
        return np.array(results)


    def single_layer_U(self,layer_weights):
        """Trainable circuit block."""
        I = np.eye(2)
        firstPart = self.nestedKron(self.U_ex(layer_weights[0]), self.U_ex(layer_weights[1]), self.U_ex(layer_weights[2]))
        secondPart = self.nestedKron(I, self.U_ex(layer_weights[3]), self.U_ex(layer_weights[4]), I)
        return np.matmul(secondPart, firstPart)

    def get_total_matrix(self,size_of_vec, weights):
        totalMatrix = np.eye(size_of_vec)
        for layer_weights in weights:
            mat = self.single_layer_U(layer_weights)
            totalMatrix = np.matmul(totalMatrix, mat)
        return totalMatrix

    def f_cnot_loss(self,y_true, y_pred):
        loss = 0
        for i in range(len(y_true)):
            fidelity = qml.math.fidelity_statevector(y_true[i], y_pred[i])
            loss += fidelity
        return np.sqrt(1 - (1/4)*abs(loss))
    
    def square_loss(self, expectedStates, predictedStates):
        loss = 0
        for i in range(len(expectedStates)):
            # c = npy.dot(npy.conjugate(expectedStates[i]), predictedStates[i])
            # c_2 = self.amp_sqrd(c)
            fidelity = qml.math.fidelity_statevector(expectedStates[i], predictedStates[i])
            loss += (1 - fidelity) ** 2
        loss /= len(expectedStates)
        return 0.5*loss
    
    def cost_fn(self, expectedStates, weights):
        preds = self.get_predictions(weights)
        loss = self.f_cnot_loss(expectedStates, preds)
        return loss



class MatrixOperations(MatrixUtils):
    def __init__(self, expectedStates, inputStates, num_layers, weights = np.array([])):
        super().__init__()
        self.expectedStates = expectedStates
        self.inputStates = inputStates
        self.num_layers = num_layers
        self.weights = self.get_random_weights() if weights.size <= 1 else weights
        self.n_qubits = 6
        self.size_of_vec = 2**self.n_qubits
    
    def get_random_weights(self):
        return super().get_random_weights(self.num_layers)

    def get_predictions(self, weights):
        return super().get_predictions(self.inputStates, weights)

    def get_total_matrix(self,weights):
        return super().get_total_matrix(self.size_of_vec, weights)
    
    def cost_fn(self, weights):
        return super().cost_fn(self.expectedStates, weights)
    
    # Define the correct operations you want the matrix to perform on basis vectors
    def target_operations(self,parameters):
        # Reshape the parameters into the matrix form
        parameters = np.reshape(parameters, (self.num_layers, 5))
        matrix = super(self).get_total_matrix(parameters)

        # Perform matrix multiplication with basis vectors
        results = super(self).get_predictions(parameters)

        # Define the target operations you want (modify this based on your specific task)
        target_result = np.array(self.expectedStates)

        # Calculate the loss as the difference between the obtained result and the target result
        # loss = square_loss(target_result, results)
        loss = super(self).f_cnot_loss(target_result, results)
        return loss

class DFS():
    def __init__(self):
        pass
    def basisVector(self, binaryStr):
        def nestedKronecker(args): # use "*args" to access an array of inputs
            assert len(args) >= 2
            temp = args[0]
            for arg in args[1:]:
                temp = np.kron(temp, arg)
            return temp

        basis = {0: [1,0], 1: [0,1], '0': [1,0], '1': [0,1]}
        return nestedKronecker([basis[x] for x in binaryStr])
    
    def getAllStates(self):
        def nestedKronecker(args): # use "*args" to access an array of inputs
            assert len(args) >= 2
            temp = args[0]
            for arg in args[1:]:
                temp = np.kron(temp, arg)
            return temp

        basis = {0: [1,0], 1: [0,1], '0': [1,0], '1': [0,1]}

        basisVector = lambda binstr : nestedKronecker([basis[x] for x in binstr])

        # common states
        zero, one = basis['0'], basis['1']
        tplus = basisVector('11')
        tminus = basisVector('00')
        tzero = (1/np.sqrt(2))*(basisVector('01') + basisVector('10'))
        singlet = np.sqrt(1/2)*(basisVector('01') - basisVector('10'))


        # ------------------------ FOR STATE 1 ------------------------

        state1 = np.kron(np.kron(singlet, singlet), singlet)

        # ------------------------ FOR STATE 2 ------------------------

        largePyramid = np.sqrt(1/3)*(np.kron(tplus,tminus)+np.kron(tminus,tplus)-np.kron(tzero,tzero))
        state2 = np.kron(singlet,largePyramid)

        # ------------------------ FOR STATE 3 ------------------------

        state3 = np.kron(largePyramid,singlet)

        # ------------------------ FOR STATE 4 ------------------------

        # for psi0 and psi1 we are combining j1=1 and j2=1/2 (this is combinind the first peak and trough)
        # J = 1/2, M = -1/2
        psi0 = np.sqrt(1/3)*np.kron(tzero, zero) - np.sqrt(2/3)*np.kron(tminus, one)
        # J = 1/2, M = +1/2
        psi1 = np.sqrt(2/3)*np.kron(tplus, zero) - np.sqrt(1/3)*np.kron(tzero, one)


        # for phiminus, phizero, phiplus, we are are combining j1=1/2 and j2=1/2
        # J = 1, M = -1
        phiminus = np.kron(psi0,zero)
        # J = 1, M = 0
        phizero = np.sqrt(1/2)*(np.kron(psi1,zero) + np.kron(psi0,one))
        # J = 1, M = +1
        phiplus = np.kron(psi1,one)

        # J=0,M=0 and j1=1,j2=1
        state4 = np.sqrt(1/3)*(np.kron(phiplus, tminus) - np.kron(phizero, tzero) + np.kron(phiminus, tplus))

        # ------------------------ FOR STATE 5 ------------------------

        eta_minus3 = np.kron(tminus, basis['0'])
        eta_minus1 = np.sqrt(2/3)*np.kron(tzero,zero) + np.sqrt(1/3)*np.kron(tminus,one)
        eta_plus1 = np.sqrt(1/3)*np.kron(tplus,zero) + np.sqrt(2/3)*np.kron(tzero, one)
        eta_plus3 = np.kron(tplus,one)

        gamma_minus = np.sqrt(1/4)*np.kron(eta_minus1, zero) - np.sqrt(3/4)*np.kron(eta_minus3, one)
        gamma_zero = np.sqrt(1/2)*np.kron(eta_plus1, zero) - np.sqrt(1/2)*np.kron(eta_minus1,one)
        gamma_plus = np.sqrt(3/4)*np.kron(eta_plus3, zero) - np.sqrt(1/4)*np.kron(eta_plus1, one)

        state5 = np.sqrt(1/3)*(np.kron(gamma_plus,tminus) - np.kron(gamma_zero, tzero) - np.kron(gamma_minus, tplus))

        states = np.array([state1, state2, state3, state4, state5])

        return states

    def getInitialTargetStates(self):
        s1,s2,s3,s4,_ = self.getAllStates()
        inputStates = np.array([s1,s2,s3,s4])
        expectedStates = np.array([s1,s2,s4,s3])
        return inputStates, expectedStates