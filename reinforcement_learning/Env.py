import numpy as np
import gym
import math

#Creating the custom environment
#Custom environment needs to inherit from the abstract class gym.Env
class QuantumEnvironment(gym.Env):
    #add the metadata attribute to your class
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, num_layers, num_of_wanted_identity_gates ,matrix_operator_obj):
        # define the environment's action_space and observation space
        self.num_layers = num_layers
        self.num_of_wanted_identity_gates = num_of_wanted_identity_gates
        '''Box-The argument low specifies the lower bound of each dimension and high specifies the upper bounds
        '''
        self.matrix_operator_obj = matrix_operator_obj
        #  weights from -1 to 1
        # self.observation_space=gym.spaces.Box(low=-1, high=1, shape=(num_layers,5,)) # this is the current state of the board
        
        # action_space are move increase values, move decrease values or stay where you are
        # creates action for increase, decrease or stay on each weight value by 0.0000001
        # subtract by one before using this value
        self.action_space=gym.spaces.Box(low=-1, high=1, shape=(num_layers*5,))
        
        self.reward_range = [0,3000]
        # current state
        self.state=self.getNewState()
        self.observation_space = self.state
        
        # rewards 
        self.reward=0

    def getNewState(self):
        return 2*np.random.random(size=(self.num_layers*5,)) - 1
    
    def makeWeightsDigestible(self, weights):
        return np.array(weights).reshape((self.num_layers,5))
    
    def computeRewardAndUpdateState(self, state, zeroThreshold):
        cond = lambda loss : True if loss<0.1 else False
        getBoost = lambda loss : int(math.ceil(-math.log2(loss**3))*15) if cond(loss) else 0
        getExtraZeroGateBoost = lambda loss, zero_counts : int(math.ceil(-math.log2((1/zero_counts)**10)))*10 if zero_counts>=1 and cond(loss) else zero_counts*5
        # update weights
        updatedState, zero_counts = self.fixBoundsAndCountZeros(np.array(state), zeroThreshold) # make -1 to 1 strictly
        # compute loss
        loss = self.matrix_operator_obj.cost_fn(self.makeWeightsDigestible(updatedState))
        totalReward = (1-loss)*500 + getBoost(loss) + getExtraZeroGateBoost(loss, zero_counts)

        return updatedState, totalReward
    
    def fixBoundsAndCountZeros(self, state, zeroThreshold):
        constraint = lambda val : val > -zeroThreshold and val < zeroThreshold
        getSign = lambda x : -1 if x < 0 else 1
        zeroCount = 0
        for i, block_val in enumerate(state):
            if abs(state[i]) > 1:
                state[i] = 1.0*getSign(state[i])
            if constraint(block_val):
                zeroCount += 1
                state[i] = 0.0
        return state, zeroCount

    
    def step(self, action):
        '''defines the logic of your environment when the agent takes an action
        Accepts an action, computes the state of the environment after applying that action
        '''
        done=False
        info={}
        zeroThreshold = 0.005

        #setting the state of the environment based on agent's action
        # rewarding the agent for the action
        self.state += action
        self.state, self.reward = self.computeRewardAndUpdateState(self.state, zeroThreshold)
        self.observation_space = self.state
        
        # define the completion of the episode
        if self.reward>=(1200):
            # self.reward+=100
            done= True
        self.render()
        return self.state, self.reward, done, info
    def render(self):
        # Visualize your environment
        print(f"\n Reward Received:{self.reward} ")
        print("==================================================")
    def reset(self):
        #reset your environment
        self.state=self.getNewState()
        self.reward=0
        return self.state
    def close(self):
        # close the environment
        self.state=0
        self.reward=0
###############################
