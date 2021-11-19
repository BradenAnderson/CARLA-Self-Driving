import numpy as np

import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.optimizers import Adam


# ==================================================================================
# Actor
# ==================================================================================
class SACActor(tf.keras.Model):
    def __init__(self, num_continuous_actions, action_max_value=1, dense1_units=256, dense2_units=256,
                 log_std_max=2, log_std_min=-20):
        super().__init__()

        # Max action value (the max value our enviornment will let us send back to it). 
        self.action_max_value = action_max_value
        
        # For clipping the value of the standard deviation output. We don't want to let this 
        # take on any extreme values. 
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        
        # Number of continuous actions the actor can take at each timestep (i.e. steering wheel and throttle).
        self.num_continuous_actions = num_continuous_actions
        
        # Dense layers
        self.dense1_units = dense1_units
        self.dense2_units = dense2_units
        
        self.dense1 = layers.Dense(units=self.dense1_units, activation = "relu")
        self.dense2 = layers.Dense(units=self.dense2_units, activation = "relu")
        self.mu_output_layer = layers.Dense(units=self.num_continuous_actions, activation="linear")
        self.log_std_output_layer = layers.Dense(units=self.num_continuous_actions, activation="linear")

    def call(self, state): 
        
        # Pass the state through the actors dense layers
        x = self.dense1(state)
        x = self.dense2(x)
        mu = self.mu_output_layer(x)             # Pass dense layer outputs to mean output layer
        log_std = self.log_std_output_layer(x)   # Pass dense layer outputs to log(std) output layer
        
        log_std = tf.clip_by_value(log_std, self.log_std_max, self.log_std_min)  # clip log_std to fall within -20, 2.
        standard_deviation = tf.exp(log_std)     # Calculate the standard deviation by exponentiating the log(std) output.
        
        # mu --> mean actions from policy given states
        # pi --> sampled actions from policy given states (sampled according to mu, sigma from actor network output)
        # logprob_pi --> log probability, according to the policy, of the action sampled by pi.
        
        # Sample actions according to mu and sigma from the actor network.
        pi = mu + tf.random.normal(tf.shape(mu)) * standard_deviation
        
        # Gaussian likelihood for entropy calculations
        logprob_pi = self._gaussian_likelihood(pi, mu, log_std)
        
        # Use tanh to bound the gaussian continuous actions to a finite range (-1 to +1)
        mu, action, logprob_pi = self._bound_actions(mu, pi, logprob_pi)
        
        # This doesn't really do anything for the CARLA probalem because all action commands max at 1, 
        # however its important to get the entire action range back after tanh squashing in enviornments
        # that can accept larger valued inputs.
        mu *= self.action_max_value
        action *= self.action_max_value
        
        return mu, action, logprob_pi
    
    # Get the log likihood for the entropy calculation
    def _gaussian_likelihood(self, x, mu, log_std):
        
        # Prevent possible division by zero.
        EPS = 1e-8
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
    
    # We use an unbounded Gaussian as the action distribution. However in practice, the actions need to be
    # bounded to a finite interval. To that end, we apply an invertible squashing function (tanh) to the Gaussian samples.
    # Appendix C, soft actor critic
    def _bound_actions(self, mu, pi, logp_pi):
        
        # Numerically stable version of eq 21, appendix C, SAC. 
        logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)
        
        # Squash those unbounded actions!
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        return mu, pi, logp_pi

# ==================================================================================
# Critic
# ==================================================================================
class SACCritic(tf.keras.Model):
    def __init__(self, dense1_units=256, dense2_units=256):
        super().__init__()
        
        # Dense layers
        self.dense1_units = dense1_units
        self.dense2_units = dense2_units
        self.dense1 = layers.Dense(units=self.dense1_units, activation = "relu")
        self.dense2 = layers.Dense(units=self.dense2_units, activation = "relu")
        
        # Q-Value output layer
        self.Q_output = layers.Dense(units=1) 
        
    def call(self, state, action):
        
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        q_value = self.Q_output(x)
        
        # Squeeze out the batch dimension and return. 
        return tf.squeeze(q_value, -1)


