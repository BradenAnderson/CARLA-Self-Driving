import numpy as np
import sys
from copy import deepcopy
from SoftActorCritic import SACActor, SACCritic
from ReplayBuffer import ReplayBuffer
from Backbone import BackboneNetwork
from ModifiedTensorboard import ModifiedTensorBoard
import time

import tensorflow as tf
from tensorflow.keras.models import save_model
from tensorflow.keras import layers 
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam

# ==================================================================================
# SAC Agent
# ==================================================================================
class SoftActorCriticAgent:
    def __init__(self, actor_dense1_units = 256, actor_dense2_units = 256, critic1_dense1_units = 256, critic1_dense2_units = 256,
                 critic2_dense1_units = 256, critic2_dense2_units = 256, num_continuous_actions=2, tau = 0.005, gamma=0.99,
                 convnet_input_shape=(600, 800, 3), convnet_type=None, actor_lr=3e-4, critic_lr=3e-4,
                 max_buffer_size=1_000, batch_size=1, alpha=0.2, verbose=False, load_weights_paths=None, critics_train_conv=False):
        
        
        self.verbose = verbose # For print statements, used to help verify implementation
        
        self.tau = tau        # Smoothing coefficient for polyak averaging of the target networks.        
        self.gamma = gamma    # Discounting factor for future rewards
        self.alpha = alpha    # Temperature for entropy scaling
        
        self.convnet_input_shape = convnet_input_shape
        self.batch_size = batch_size
        self.let_critics_train_conv = critics_train_conv
        
        # Convolutional backbone, takes in images from the cars camera
        self.convnet_backbone = BackboneNetwork(input_shape=convnet_input_shape, network_type=convnet_type, batch_size=batch_size)
        
        # Instantiate two critic networks and an actor network
        self.critic1_network = SACCritic(dense1_units = critic1_dense1_units, dense2_units=critic1_dense2_units)
        self.critic2_network = SACCritic(dense1_units = critic2_dense1_units, dense2_units=critic2_dense2_units)
        self.actor_network = SACActor(dense1_units=actor_dense1_units, dense2_units=actor_dense2_units, num_continuous_actions=num_continuous_actions)
        
        # Target critic networks, these never get updated with gradient desecent
        # they are just an exponential moving average of the associated critic networks weights
        self.critic1_target_network = deepcopy(self.critic1_network)
        self.critic2_target_network = deepcopy(self.critic2_network)
        
        # Compile the two critics, the actor and the convnet backbone.
        self.critic1_network.compile(optimizer=Adam(learning_rate=critic_lr))
        self.critic2_network.compile(optimizer=Adam(learning_rate=critic_lr))
        self.actor_network.compile(optimizer=Adam(learning_rate=actor_lr))
        self.convnet_backbone.compile(optimizer=Adam(learning_rate=actor_lr))
        
        # As stated before, these networks don't get updated with gradient descent
        # Passing the optimizer and learning rate here is just to satisfy the keras convention. 
        self.critic1_target_network.compile(optimizer=Adam(learning_rate=critic_lr))
        self.critic2_target_network.compile(optimizer=Adam(learning_rate=critic_lr))
        
        self.replay_buffer = ReplayBuffer(max_size=max_buffer_size, input_shape=convnet_input_shape, n_actions=num_continuous_actions)

        self.initialization_time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.load_weight_paths = load_weights_paths
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/SAC/SAC_logs_{self.initialization_time_stamp}")
        self.policy_loss_tracker = []
        self.critic1_loss_tracker = []
        self.critic2_loss_tracker = []
        self.total_critic_loss_tracker = []

        if self.load_weight_paths is not None:
            try:
                self._save_model_filepaths_from_param()
            except:
                print("\n=====================================================================================")
                print(f"Exception occured:\n {sys.exc_info()[0]}\n")
                print("ERROR! Exception occured when storing the model load paths.")
                print("load_weight_paths parameter must either be 'None' or type dict")
                print("If passing a dict, it must have the following keys: ")
                print("critic1, critic2, actor, critic1_target, critic2_target, backbone\n")
                print("WARNING: Agent was constructed using new filepaths due to improper load_weight_paths input.")
                print("=====================================================================================\n")
                self._create_model_filepaths()
        else:
            self._create_model_filepaths()

    def get_loss_trackers(self):
        tracker = {'policy_loss':self.policy_loss_tracker,
                   'critic1_loss':self.critic1_loss_tracker,
                   'critic2_loss':self.critic2_loss_tracker,
                   'total_critic_loss':self.total_critic_loss_tracker}
        return tracker

    def reset_loss_trackers(self):
        self.policy_loss_tracker = []
        self.critic1_loss_tracker = []
        self.critic2_loss_tracker = []
        self.total_critic_loss_tracker = []

    def _create_model_filepaths(self):
        self.convnet_backbone_filepath = f"./models/backbone/backbone_{self.initialization_time_stamp}.ckpt"
        self.critic1_filepath = f"./models/critic1/critic1_{self.initialization_time_stamp}.ckpt"
        self.critic2_filepath = f"./models/critic2/critic2_{self.initialization_time_stamp}.ckpt"
        self.actor_filepath = f"./models/actor/actor_{self.initialization_time_stamp}.ckpt"
        self.critic1_target_filepath = f"./models/critic1_target/critic1_target_{self.initialization_time_stamp}.ckpt"
        self.critic2_target_filepath = f"./models/critic2_target/critic2_target_{self.initialization_time_stamp}.ckpt"

    def _save_model_filepaths_from_param(self):
        self.convnet_backbone_filepath = self.load_weight_paths['backbone']
        self.critic1_filepath = self.load_weight_paths['critic1']
        self.critic2_filepath = self.load_weight_paths['critic2']
        self.actor_filepath = self.load_weight_paths['actor']
        self.critic1_target_filepath = self.load_weight_paths['critic1_target']
        self.critic2_target_filepath = self.load_weight_paths['critic2_target']
    
    def _initialize_weights(self):
        random_data = np.random.sample((1, *self.convnet_input_shape))

        preprocessed_states = self.convnet_backbone(random_data)

        rand_mu, rand_act, rand_logprob = self.actor_network(preprocessed_states)

        rand_q1 = self.critic1_network(preprocessed_states, rand_act)
        rand_q2 = self.critic2_network(preprocessed_states, rand_act)
        rand_q1t = self.critic1_target_network(preprocessed_states, rand_act)
        rand_q2t = self.critic2_target_network(preprocessed_states, rand_act)

    def save_all_models(self):
        self.convnet_backbone.save_weights(self.convnet_backbone_filepath, save_format="tf")
        self.critic1_network.save_weights(self.critic1_filepath, save_format="tf")
        self.critic2_network.save_weights(self.critic2_filepath, save_format="tf")
        self.actor_network.save_weights(self.actor_filepath, save_format="tf")
        self.critic1_target_network.save_weights(self.critic1_target_filepath, save_format="tf")
        self.critic2_target_network.save_weights(self.critic2_target_filepath, save_format="tf")

    def load_all_models(self):

        self._initialize_weights()

        self.convnet_backbone.load_weights(self.convnet_backbone_filepath)
        self.critic1_network.load_weights(self.critic1_target_filepath)
        self.critic2_network.load_weights(self.critic1_target_filepath)
        self.actor_network.load_weights(self.actor_filepath)
        self.critic1_target_network.load_weights(self.critic1_target_filepath)
        self.critic2_target_network.load_weights(self.critic2_target_filepath) 

    def update_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state, action, reward, new_state, done)
        
    def update_target_networks(self, soft=True):
            
        # if we are doing a soft update, i.e. polyak averaging. 
        if soft:
                
            # Create updated weights for critic_1s target network with polyak averaging.
            updated_target_critic1_weights = []
            for online_weight, target_weight in zip(self.critic1_network.get_weights(), self.critic1_target_network.get_weights()):
                    
                updated_weight = online_weight*self.tau + (1 - self.tau)*target_weight
                updated_target_critic1_weights.append(updated_weight)

            # Update target critic_1s weights
            self.critic1_target_network.set_weights(updated_target_critic1_weights)

            # Create updated weights for critic_2s target network with polyak averaging.
            updated_target_critic2_weights = []
            for online_weight, target_weight in zip(self.critic2_network.get_weights(), self.critic2_target_network.get_weights()): 

                updated_weight = online_weight*self.tau + (1 - self.tau)*target_weight
                updated_target_critic2_weights.append(updated_weight)

            # Update target critic_2s weights
            self.critic2_target_network.set_weights(updated_target_critic2_weights)

    def compute_qloss_targets(self, states, rewards, dones):

        # actions pulled fresh from the current policy
        mus, actions, logprobs = self.actor_network(states) 

        # Q-values from the two offline critics
        qval_target_network1 = self.critic1_target_network(states, actions)
        qval_target_network2 = self.critic2_target_network(states, actions)
        q_min = tf.minimum(qval_target_network1, qval_target_network2)

        # Here we create the estimated "target" to use when updating the q_networks. 
        targets = rewards + self.gamma*(1 - dones)*(q_min - self.alpha*logprobs)

        return targets

    def compute_policy_loss(self, states):

        # Actions sampled fresh from current policy
        mu, actions, logprobs = self.actor_network(states)

        # Q_values from each critic, using states from buffer and actions from current policy
        critic1_qvals = self.critic1_network(states, actions)
        critic2_qvals = self.critic2_network(states, actions)

        # Using the minimum of the q_values from the two critics to help prevent overestimation bias. 
        q_mins = tf.minimum(critic1_qvals, critic2_qvals)

        entropy = self.alpha*logprobs

        # Negative here for gradient Assent! 
        policy_loss = -tf.reduce_mean(q_mins - entropy)

        return policy_loss

    def update_policy_and_qvalue_networks(self):

        # If we haven't put one batch worth of experiences in the replay buffer, we can't do this yet
        # so return.
        if self.replay_buffer.memory_counter < self.batch_size:
            return

        if self.let_critics_train_conv:
            all_critic_weights = self.critic1_network.trainable_variables + self.critic2_network.trainable_variables + self.convnet_backbone.trainable_variables
        else:    
            all_critic_weights = self.critic1_network.trainable_variables + self.critic2_network.trainable_variables

        all_policy_weights = self.actor_network.trainable_variables + self.convnet_backbone.trainable_variables

        # Randomly sample a batch of experiences from memory.
        states, actions, rewards, new_states, dones = self.replay_buffer.sample_buffer(batch_size=self.batch_size, as_tensor=True)
        
        if self.verbose: print("Updating QNetworks....")
        
        # 1. Update Q Function Parameters, forward prop
        with tf.GradientTape() as tape:
            
            preprocessed_states = self.convnet_backbone(states)

            # Compute target values for q_loss function
            targets = self.compute_qloss_targets(preprocessed_states, rewards, dones)

            # Calculate critic 1 and critic 2 qvalues with both states and actions from the buffer
            critic1_qval = self.critic1_network(preprocessed_states, actions)
            critic2_qval = self.critic2_network(preprocessed_states, actions)

            # Calculate the loss values for critic 1 and critic 2
            # as the mean squared error between Q_Online_Network_SA_From_Buffer and targest (where s from buffer but a sampled fresh from current policy).
            critic1_loss = tf.reduce_mean((critic1_qval - targets)**2)
            critic2_loss = tf.reduce_mean((critic2_qval - targets)**2) 

            total_qloss = critic1_loss + critic2_loss
            
            self.critic1_loss_tracker.append(critic1_loss)
            self.critic2_loss_tracker.append(critic2_loss)
            self.total_critic_loss_tracker.append(total_qloss)

        # Backprop and Gradient Descent for Critic Networks. 
        critic_gradients = tape.gradient(total_qloss, all_critic_weights)
        self.critic1_network.optimizer.apply_gradients(zip(critic_gradients, all_critic_weights))
        #critic1_gradients = tape.gradient(total_qloss, self.critic1_network.trainable_variables)
        #critic2_gradients = tape.gradient(total_qloss, self.critic2_network.trainable_variables)
        #self.critic1_network.optimizer.apply_gradients(zip(critic1_gradients, self.critic1_network.trainable_variables))
        #self.critic2_network.optimizer.apply_gradients(zip(critic2_gradients, self.critic2_network.trainable_variables))
        
        if self.verbose: print("QNetwork update complete! Starting policy and convnet updates...")
        
        # 2. Update Policy Weights
        with tf.GradientTape() as tape:

            preprocessed_states = self.convnet_backbone(states)
            policy_loss = self.compute_policy_loss(preprocessed_states)

            self.policy_loss_tracker.append(policy_loss)

        all_policy_gradients = tape.gradient(policy_loss, all_policy_weights)
        self.actor_network.optimizer.apply_gradients(zip(all_policy_gradients, all_policy_weights))
        #policy_gradients = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        #convnet_gradients = tape.gradient(policy_loss, self.convnet_backbone.trainable_variables)

        #if self.verbose: print("Policy updated! Starting target network updates with polyak averaging...")
        #self.actor_network.optimizer.apply_gradients(zip(policy_gradients, self.actor_network.trainable_variables))

        #if self.verbose: print("Convnet updated! Starting policy network update...")
        #self.convnet_backbone.optimizer.apply_gradients(zip(convnet_gradients, self.convnet_backbone.trainable_variables))

        # 3. Adjust Temperature - If using automatic temperature tuning per Soft Actor-Critic Algorithms and Applications
        # Update here

        # 4. Update Target Network weights - Update the offline networks using polyak averaging
        self.update_target_networks(soft=True)
        
        if self.verbose:
            print("Target network update complete!")
            print(f"Finished network updates, tensorboard step: {self.tensorboard.step}\n")