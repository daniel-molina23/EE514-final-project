import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from numpy import pi

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=2**7, fc2_dims=2**9,
            name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims # this is the action dimension
        self.fc2_dims = fc2_dims # fully connected dims?

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc2_dims, activation='relu')
        self.fc4 = Dense(self.fc1_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        action_value = self.fc3(action_value)
        action_value = self.fc4(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=2**6, fc2_dims=2**8, n_actions=2**6, name='actor',
            chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.fc3 = Dense(self.fc2_dims, activation='relu')
        self.fc4 = Dense(self.fc2_dims//2, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)

        mu = self.mu(prob) * pi

        return mu

