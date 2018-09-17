import gym
import ddpg
import replay_buffer
import numpy as np
import random
import os
import datetime
import pickle

# Global variables
train_iter = 1000000
gamma = 0.98
tau = 0.001
batch_size = 256
buffer_size = 1000000
replay_start = batch_size
n_train = 40
n_episode = 256
n_cycles = 50
n_epoch = 30
n_test = 100

env_name = 'FetchReach-v1'

her = False
render = False



if __name__ == '__main__':
	env = gym.make(env_name)
	env.seed(0)
	random.seed(0)
	np.random.seed(0)
	
	# Make a directory to store the learned policies
	dirname = datetime.datetime.now().isoformat()
	os.mkdir(dirname)
	
	replay_buffer = replay_buffer.ReplayBuffer(buffer_size)
	sample_batch = replay_buffer.get_batch
	
	
	ddpg = ddpg.DDPG(env, replay_buffer, sample_batch, train_iter, gamma, tau,
		batch_size, n_train, n_episode)
		
		
	for epoch in range(n_epoch):
		print("Start training epoch", epoch)
		for cycle in range(n_cycles):
			for episode in range(n_episode):
				state = env.reset()
				state = np.concatenate((state['observation'], state['achieved_goal'], state['desired_goal']))
				tot_reward = 0
				ddpg.reset_noise()
				for step in range(env.spec.timestep_limit):
					if random.random() < 0.2:
						action = env.action_space.sample()
					else: action = ddpg.noise_action(state)
					obs, reward, done, info = env.step(action)
					next_state = obs
					next_state = np.concatenate((obs['observation'], obs['achieved_goal'], obs['desired_goal']))
					replay_buffer.add(state, action, reward, next_state, done)
					
					if her:
						substitute_goal = obs['achieved_goal'].copy()
						substitute_reward = env.compute_reward(obs['achieved_goal'],
							substitute_goal, info)
						substitute_state = np.concatenate((obs['observation'], obs['achieved_goal'], substitute_goal))
						substitute_next_state = np.concatenate((obs['observation'], obs['achieved_goal'], substitute_goal))
						substitute_done = True
						replay_buffer.add(substitute_state,action,substitute_reward, substitute_next_state, substitute_done)

					state = next_state
					if done:
						ddpg.reset_noise()
						break
			for _ in range(n_train):
				ddpg.train()
				


		# Test
			tot_reward = 0
		for test in range(n_test):
			state = env.reset()
			state = np.concatenate((state['observation'], state['achieved_goal'], state['desired_goal']))
			for i in range(env.spec.timestep_limit):
				if render: env.render()
				action = ddpg.action(state)
				state,reward,done,info = env.step(action)
				state = np.concatenate((state['observation'], state['achieved_goal'], state['desired_goal']))
				tot_reward += reward
				if done:
					break
			avg_reward = tot_reward/n_test
		print('Epoch : ', epoch,' has reward : ', avg_reward)
		
		# Save the policy
		path = '{}/model-{}.ckpt'.format(dirname,epoch)
		ddpg.save_policy(path)
