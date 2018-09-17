import tensorflow as tf
import numpy as np
import gym

from ou_noise import OUNoise
from critic_network import CriticNetwork 
from actor_network import ActorNetwork
from normalizer import Normalizer

class DDPG:
	def __init__(self, env, replay_buffer, sample_batch, train_iter, gamma, tau,
		batch_size, n_train, n_episode):
		# Gym environment
		self.env = env
		
		env_flattened = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'achieved_goal', 'desired_goal'])
		
		# Get space sizes
		self.state_dim = env_flattened.observation_space.shape[0]
		#self.state_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		
		# Get replay buffer and function get a batch from it
		self.replay_buffer = replay_buffer
		self.sample_batch = sample_batch
		
		self.sess = tf.InteractiveSession()
		
		# Hyper parameters
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.n_train = n_train
		self.n_episode = n_episode
			
		# Initialize networks
		self.critic = CriticNetwork(self.sess, self.state_dim, self.action_dim)
		self.actor = ActorNetwork(self.sess, self.state_dim, self.action_dim)

	 
		self.exploration_noise = OUNoise(self.action_dim)
		
		
		
	def train(self):
		batch = self.sample_batch(self.batch_size)
		
		state_batch = np.asarray([data[0] for data in batch])
		action_batch = np.asarray([data[1] for data in batch])
		reward_batch = np.asarray([data[2] for data in batch])
		next_state_batch = np.asarray([data[3] for data in batch])
		done_batch = np.asarray([data[4] for data in batch])
		
				
		next_action_batch = self.actor.target_actions(next_state_batch)
		q_value_batch = self.critic.target_q(next_state_batch,
			next_action_batch)
		y_batch = []  
		for i in range(len(batch)): 
			if done_batch[i]:
				y_batch.append(reward_batch[i])
			else :
				y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i])
		y_batch = np.resize(y_batch,[self.batch_size,1])
		# Update critic by minimizing the loss L
		self.critic.train(y_batch,state_batch,action_batch)

		# Update the actor policy using the sampled gradient:
		action_batch_for_gradients = self.actor.actions(state_batch)
		q_gradient_batch = self.critic.gradients(state_batch,action_batch_for_gradients)

		self.actor.train(q_gradient_batch,state_batch)

		# Update the target networks
		self.actor.update_target()
		self.critic.update_target()
		
	def noise_action(self,state):
		action = self.actor.action(state)
		return action + self.exploration_noise.noise()
		
	def action(self,state):
		return self.actor.action(state)
		
	def reset_noise(self):
		self.exploration_noise.reset()
		
	def save_policy(self, save_path):
		self.actor.save_network(save_path)
