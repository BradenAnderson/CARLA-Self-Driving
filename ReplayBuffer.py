import numpy as np
import tensorflow as tf 

# ==================================================================================
# Replay Buffer
# ==================================================================================
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        
        self.mem_size = max_size
        self.memory_counter = 0 # Track the first available memory position
        
        # Initialize empty numpy arrays for the agents replay buffer (state, action, reward, next_state, done)
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)     # State the agent was in
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32) # New states the agent sees as a result of its actions
        self.action_memory = np.zeros((self.mem_size, n_actions))       # n_actions means the number of components to the actions (these are continuous actions)
        self.reward_memory = np.zeros(self.mem_size)                    # reward recieved at that time step
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)   # Array to track the "done" flags recieved.
                                                                        # Need this because the future "value" of the terminal state is always zero. (because no future rewards follow).
        
    def store_transition(self, state, action, reward, state_, done):
        
        # Need the position of the first available memory
        # While memory_counter < mem_size, this will give us the next available position not yet written to.
        # When memory_counter > mem_size, this will start overwriting the buffer, starting at the oldest item.
        index = self.memory_counter % self.mem_size
        
        # Update memory arrays
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        # Increment memory counter
        self.memory_counter += 1
    
    # ==================================================================================================
    # This is just doing uniform sampling from the buffer (take the good experiences with the bad)
    # Could improve later by implementing prioritized experience replay!
    # ==================================================================================================
    def sample_buffer(self, batch_size, as_tensor=False):
        
        # We want to know the first available memory, given by the minimum of memory size and memory counter.
        # Avoid sampling unfilled memory locations (that are still just all zeros).
        max_mem = min(self.memory_counter, self.mem_size)
        
        # Grab a batch of memories.
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.terminal_memory[batch]
        
        if as_tensor:
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        return states, actions, rewards, states_, dones