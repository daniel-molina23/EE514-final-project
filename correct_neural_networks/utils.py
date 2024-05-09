import numpy as np

class MatrixUtils():
    def __init__(self, pi_range=False, extended_states=False):
        self.pi_range = pi_range
        self.extended_states = extended_states
        pass

    def fidelity_statevector(self, state1, state2):
        return np.abs(np.dot(np.conjugate(state1), state2))**2
    
    def Udot(s1,U,s2):
        return np.dot(np.conjugate(np.transpose(s1)),np.matmul(U,s2))

    def nestedKron(self, *args): # use "*args" to access an array of inputs
        assert len(args) >= 2
        temp = args[0]
        for arg in args[1:]:
            temp = np.kron(temp, arg)
        return temp

    def get_random_weights(self, num_layers):
        multiplier = np.pi if self.pi_range else 1
        return 2 * multiplier * np.random.random(size=(num_layers, 5)) - (1*multiplier)

    def get_random_half_layer_weights(self):
        multiplier = np.pi if self.pi_range else 1
        return 2 * multiplier * np.random.random(size=(3,)) - (1 * multiplier)

    def U_ex(self, p, scale=1):
        from scipy.linalg import expm
        X = [[0,1],[1,0]]
        Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
        Z = [[1,0],[0,-1]]

        H_ex = (1/4)*scale*(np.kron(X,X) + np.kron(Y,Y) + np.kron(Z,Z))
        # print(f'H_ex.type = {type(H_ex)}')
        U_exchange = expm(-1j*p*H_ex) if self.pi_range else expm(-1j*np.pi*p*H_ex) # p is [-pi, pi] or [-1, 1]

        return np.array(U_exchange)

    def get_predictions(self, inputStates, weights, halfWeights=None):
        matOp = self.get_total_matrix(2**6,weights, halfWeights=halfWeights)
        results = []
        for i in range(len(inputStates)):
            results.append(np.matmul(matOp, inputStates[i]))
        return np.array(results)

    def single_layer_U(self,layer_weights):
        """Trainable circuit block."""
        I = np.eye(2)
        firstPart = self.first_half_layer_weights(layer_weights)
        secondPart = self.nestedKron(I, self.U_ex(layer_weights[3]), self.U_ex(layer_weights[4]), I)
        return np.matmul(secondPart, firstPart)

    def first_half_layer_weights(self,layer_weights):
        return self.nestedKron(self.U_ex(layer_weights[0]), self.U_ex(layer_weights[1]), self.U_ex(layer_weights[2]))

    def get_total_matrix(self, size_of_vec, weights, halfWeights=None):
        # WOWWWWWW I SAVED THIS......
        totalMatrix = np.eye(size_of_vec)
        for layer_weights in weights:
            mat = self.single_layer_U(layer_weights)
            totalMatrix = np.matmul(mat, totalMatrix)
        if halfWeights is not None:
            mat = self.first_half_layer_weights(halfWeights)
            totalMatrix = np.matmul(mat, totalMatrix)
        return totalMatrix

    def f_cnot_loss(self, y_true, y_pred):
        first_loss = 0
        for i in range(0,4):
            fidelity = self.fidelity_statevector(y_true[i], y_pred[i])
            first_loss += fidelity
        second_loss = 0
        second_end_point = len(y_true)-4 if self.extended_states else len(y_true)
        for i in range(4,second_end_point):
            fidelity = self.fidelity_statevector(y_true[i], y_pred[i])
            second_loss += fidelity
        
        other_loss = 0
        if self.extended_states:
            # it is included in both partitions so that we don't weigh heavily on the last two states
            for i in range(-4,0):
                fidelity = self.fidelity_statevector(y_true[i], y_pred[i])
                other_loss += fidelity
        
        third_portion = abs(other_loss)/4 if self.extended_states else 0

        total = 3 if self.extended_states else 2
        return np.sqrt(total - (1/4)*abs(first_loss) - (1/4)*abs(second_loss) - third_portion)

    def square_loss(self, expectedStates, predictedStates):
        loss = 0
        for i in range(len(expectedStates)):
            fidelity = self.fidelity_statevector(expectedStates[i], predictedStates[i])
            loss += (1 - fidelity) ** 2
        loss /= len(expectedStates)
        return 0.5*loss

    def cost_fn(self, expectedStates, weights, halfWeights=None):
        preds = self.get_predictions(weights, halfWeights=halfWeights)
        loss = self.f_cnot_loss(expectedStates, preds)
        return loss



class MatrixOperations(MatrixUtils):
    def __init__(self, expectedStates, inputStates, num_layers, weights = np.array([]), pi_range=False):
        super(self).__init__()
        self.expectedStates = expectedStates
        self.inputStates = inputStates
        self.num_layers = num_layers
        self.weights = self.get_random_weights() if weights.size <= 1 else weights
        self.n_qubits = 6
        self.size_of_vec = 2**self.n_qubits
        self.pi_range = pi_range
    
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

    def nestedKronecker(self, args): # use "*args" to access an array of inputs
        assert len(args) >= 2
        temp = args[0]
        for arg in args[1:]:
            temp = np.kron(temp, arg)
        return temp
    
    def basisVector(self, binaryStr):
        basis = {0: [1,0], 1: [0,1], '0': [1,0], '1': [0,1]}
        return self.nestedKronecker([basis[x] for x in binaryStr])
    
    def getModifiedAllStates(self):
        basis = {0: [1,0], 1: [0,1], '0': [1,0], '1': [0,1]}

        basisVector = lambda binstr : self.nestedKronecker([basis[x] for x in binstr])

        # common states
        zero, one = basis['0'], basis['1']
        tplus = basisVector('11')
        tminus = basisVector('00')
        tzero = (1/np.sqrt(2))*(basisVector('01') + basisVector('10'))
        singlet = np.sqrt(1/2)*(basisVector('01') - basisVector('10'))

        # the four states which comprise the unleaked states of the DFS of N=3
        s1 = self.nestedKronecker([singlet,one]) # M = + 1/2
        s2 = self.nestedKronecker([singlet,zero]) # M = - 1/2
        s3 = 1/np.sqrt(3)*(np.sqrt(2)*self.nestedKronecker([tplus,zero]) - self.nestedKronecker([tzero,one])) # M = +1/2
        s4 = 1/np.sqrt(3)*(np.sqrt(2)*self.nestedKronecker([tzero,zero]) - self.nestedKronecker([tminus,one])) # M = -1/2

        state1 = 1/np.sqrt(2)*(self.nestedKronecker([s1[::-1],s2]) - self.nestedKronecker([s2[::-1],s1]))
        state2 = 1/np.sqrt(2)*(self.nestedKronecker([s1[::-1],s4]) - self.nestedKronecker([s2[::-1],s3]))
        state3 = 1/np.sqrt(2)*(self.nestedKronecker([s3[::-1],s2]) - self.nestedKronecker([s4[::-1],s1]))
        state4 = 1/np.sqrt(2)*(self.nestedKronecker([s3[::-1],s4]) - self.nestedKronecker([s4[::-1],s3]))

        state5 = self.nestedKronecker([s2[::-1],s2])
        state6 = self.nestedKronecker([s2[::-1],s4])
        state7 = self.nestedKronecker([s4[::-1],s2])
        state8 = self.nestedKronecker([s4[::-1],s4])

        extraState9 = self.nestedKronecker([s4[::-1],s1])
        extraState10 = self.nestedKronecker([s4[::-1],s3])
        extraState11 = self.nestedKronecker([s3[::-1],s2])
        extraState12 = self.nestedKronecker([s3[::-1],s4])
        
        states = np.array([state1,state2,state3,state4,state5,state6,state7,state8, extraState9, extraState10, extraState11, extraState12])

        return states
    
    def getAllStates(self):
        basis = {0: [1,0], 1: [0,1], '0': [1,0], '1': [0,1]}

        basisVector = lambda binstr : self.nestedKronecker([basis[x] for x in binstr])

        # common states
        zero, one = basis['0'], basis['1']
        tplus = basisVector('11')
        tminus = basisVector('00')
        tzero = (1/np.sqrt(2))*(basisVector('01') + basisVector('10'))
        singlet = np.sqrt(1/2)*(basisVector('01') - basisVector('10'))

        # the four states which comprise the unleaked states of the DFS of N=3
        s1 = self.nestedKronecker([singlet,one]) # M = + 1/2
        s2 = self.nestedKronecker([singlet,zero]) # M = - 1/2
        s3 = 1/np.sqrt(3)*(np.sqrt(2)*self.nestedKronecker([tplus,zero]) - self.nestedKronecker([tzero,one])) # M = +1/2
        s4 = 1/np.sqrt(3)*(np.sqrt(2)*self.nestedKronecker([tzero,zero]) - self.nestedKronecker([tminus,one])) # M = -1/2

        state1 = 1/np.sqrt(2)*(self.nestedKronecker([s1,s2]) - self.nestedKronecker([s2,s1]))
        state2 = 1/np.sqrt(2)*(self.nestedKronecker([s1,s4]) - self.nestedKronecker([s2,s3]))
        state3 = 1/np.sqrt(2)*(self.nestedKronecker([s3,s2]) - self.nestedKronecker([s4,s1]))
        state4 = 1/np.sqrt(2)*(self.nestedKronecker([s3,s4]) - self.nestedKronecker([s4,s3]))
        
        state5 = self.nestedKronecker([s2,s2])
        state6 = self.nestedKronecker([s2,s4])
        state7 = self.nestedKronecker([s4,s2])
        state8 = self.nestedKronecker([s4,s4])

        extraState9 = self.nestedKronecker([s4,s1])
        extraState10 = self.nestedKronecker([s4,s3])
        extraState11 = self.nestedKronecker([s3,s2])
        extraState12 = self.nestedKronecker([s3,s4])
        
        states = np.array([state1,state2,state3,state4,state5,state6,state7,state8, extraState9, extraState10, extraState11, extraState12])

        return states
    
    def helperInitialTargetStates(self, extraStates=False, modified=False):
        s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12 = self.getModifiedAllStates() if modified else self.getAllStates()
        inputStates = np.array([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12])
        expectedStates = np.array([s1,s2,s4,s3,s5,s6,s8,s7,s10,s9,s12,s11])
        return (inputStates, expectedStates) if extraStates else (inputStates[:8], expectedStates[:8])

    def getModifiedInitialTargetStates(self, extraStates=False):
        return self.helperInitialTargetStates(extraStates=extraStates, modified=True)

    def getInitialTargetStates(self, extraStates=False):
        return self.helperInitialTargetStates(extraStates=extraStates, modified=False)