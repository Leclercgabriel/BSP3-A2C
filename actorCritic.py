import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
#from networks import actorCriticNetwork
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class ActorCritic:
    def __init__(self, alpha=0.01, gamma=0.95, action_space=4, observation_space=8):
        super(ActorCritic, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.action = None
        self.actionSpace = [i for i in range (0, self.action_space)]

        self.actor_checkpoint_file = os.path.join("logs/actor")
        self.critic_checkpoint_file = os.path.join("logs/critic")

        #self.aD1_dims = aD1_dims
        #self.aD2_dims = aD2_dims
        self.aOutput_dims = action_space

        #self.cD1_dims = cD1_dims
        #self.cD2_dims = acD2_dims
        #self.cOutput_dims = 1

        self.stateInput = Input(shape=(observation_space))

        self.aD1 = Dense(86, activation=tf.keras.layers.LeakyReLU())(self.stateInput)
        self.aD2 = Dense(32, activation=tf.keras.layers.LeakyReLU())(self.aD1)
        self.aOutput = Dense(self.aOutput_dims, activation="softmax")(self.aD2)

        self.cD1 = Dense(86, activation=tf.keras.layers.LeakyReLU())(self.stateInput)
        self.cD2 = Dense(32, activation=tf.keras.layers.LeakyReLU())(self.cD1)
        self.cOutput = Dense(1, activation=None)(self.cD2)

        self.Actor = Model(inputs=self.stateInput, outputs=self.aOutput)
        self.Critic = Model(inputs=self.stateInput, outputs=self.cOutput)

        self.Actor.summary()
        self.Critic.summary()
        
        self.Actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.Critic.compile(optimizer=Adam(learning_rate=self.alpha))

    def chooseAction(self, observations):
        state = tf.convert_to_tensor([observations])
        proba = self.Actor(state)

        actionProba = tfp.distributions.Categorical(probs=proba)
        action = actionProba.sample()
        self.action = action

        return action.numpy()[0]

    def aSave(self):
        print("...saving actor...")
        self.Actor.save_weights(self.actor_checkpoint_file)

    def aLoad(self):
        print("...loading actor...")
        self.Actor.load_weights(self.actor_checkpoint_file)
    
    def cSave(self):
        print("...saving critic...")
        self.Critic.save_weights(self.critic_checkpoint_file)

    def cLoad(self):
        print("...loading critic...")
        self.Critic.load_weights(self.critic_checkpoint_file)

    def save(self):
        print("...saving...")
        self.aSave()
        self.cSave()

    def load(self):
        print("...loading...")
        self.aLoad()
        self.cLoad()

    def learn(self, state, reward, state1, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state1 = tf.convert_to_tensor([state1], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            
            actionProba = self.Actor(state)

            V_value = self.Critic(state)
            V_value1 = self.Critic(state1)
            V_value = tf.squeeze(V_value)
            V_value1 = tf.squeeze(V_value1)

            actionProba = tfp.distributions.Categorical(probs=actionProba)
            logProba = actionProba.log_prob(self.action)

            advantage = reward + self.gamma*V_value1*(1-int(done)) - V_value

            aLoss = -logProba * advantage
            cLoss = advantage ** 2

        aGradient = tape.gradient(aLoss, self.Actor.trainable_variables)
        cGradient = tape.gradient(cLoss, self.Critic.trainable_variables)
        
        self.Actor.optimizer.apply_gradients(zip(aGradient, self.Actor.trainable_variables))
        self.Critic.optimizer.apply_gradients(zip(cGradient, self.Critic.trainable_variables))
        
        del tape


