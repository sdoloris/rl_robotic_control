import gym
import tensorflow as tf
import numpy as np
from actor_network import ActorNetwork

dirname = 'FetchReach-best'
env_name = 'FetchReach-v1'
model_name = 'model-30'

n_test = 10
render = True

if __name__ == '__main__':
	env = gym.make(env_name)
	env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'achieved_goal', 'desired_goal'])
	
	
	state_dim = env.observation_space.shape[0]

	tf.reset_default_graph()
	successes = 0
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	new_saver = tf.train.import_meta_graph('{}/{}.ckpt.meta'.format(dirname,model_name))
	new_saver.restore(sess, '{}/{}.ckpt'.format(dirname,model_name))
	graph = tf.get_default_graph()
	state_input = graph.get_tensor_by_name("state:0")
	action_output = graph.get_tensor_by_name("action:0")
	
	# set state and get action
	# Evaluate the policy on 100 episodes
	successes = 0
	for test in range(n_test):
		state = env.reset()
		reward_count = 0
		for i in range(env.spec.timestep_limit):
			if render: env.render()
			state = np.array(state).reshape(1,state_dim)
			action = sess.run(action_output,feed_dict={state_input:state})[0]
			state,reward,done,info = env.step(action)		
			reward_count += reward
		if reward_count > -50:
			successes += 1
	success_rate = successes/n_test
	avg_reward = reward_count/n_test
	epochs = np.append(epochs, k+1)
	rates = np.append(rates, success_rate)
	print("Model has success rate {}".format(success_rate))
		
