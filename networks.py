import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class actorCriticNetwork(keras.Model):
    def __init__(self, action_space, fc1_dims=1024, fc2_dims=512, name='actorCritic', chkpt_dir='logs/actor_critic'):
        super(actorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.action_space = action_space
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.d1 = Dense(self.fc1_dims, activation='relu')
        self.d2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.policy = Dense(self.action_space, activation='softmax')

    def call(self, state):
        value = self.d1(state)
        value = self.d2(value)

        v = self.v(value)
        policy = self.policy(value)

        return v, policy
