import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import actorCriticNetwork
import numpy as np

class Agent:
    def __init__(self, alpha=0.01, gamma=0.95, action_space=4, observation_space=8):
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.action = None
        self.actionSpace = [i for i in range (0, self.action_space)]

        self.actorCritic = actorCriticNetwork(action_space=action_space)

        self.actorCritic.compile(optimizer=Adam(learning_rate=alpha))

    def chooseAction(self, observations):
        state = tf.convert_to_tensor([observations])
        _, proba = self.actorCritic(state)

        actionProba = tfp.distributions.Categorical(probs=proba)
        action = actionProba.sample()
        self.action = action

        return action.numpy()[0]

    def save(self):
        print("...saving...")
        self.actorCritic.save_weights(self.actorCritic.checkpoint_file)

    def load(self):
        print("...loading...")
        self.actorCritic.load_weights(self.actorCritic.checkpoint_file)
    
    def learn(self, state, reward, state1, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state1 = tf.convert_to_tensor([state1], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            V_value, proba = self.actorCritic(state)
            V_value1, _ = self.actorCritic(state1)
            V_value = tf.squeeze(V_value)
            V_value1 = tf.squeeze(V_value1)

            actionProba = tfp.distributions.Categorical(probs=proba)
            log_proba = actionProba.log_prob(self.action)

            delta = reward + self.gamma*V_value1*(1-int(done)) - V_value
            actorLoss = -log_proba*delta
            criticLoss = delta**2
            totalLoss = actorLoss + criticLoss

        gradient = tape.gradient(totalLoss, self.actorCritic.trainable_variables)
        self.actorCritic.optimizer.apply_gradients(zip(gradient, self.actorCritic.trainable_variables))
