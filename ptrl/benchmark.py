import algos.core as core
import algos.dqn as dqn
import gymnasium as gym
	
def create_env_fn(render_mode=None):
	return gym.make("LunarLander-v3",render_mode=render_mode)		# https://gymnasium.farama.org/environments/box2d/lunar_lander/

temp_env = create_env_fn()
env_name = temp_env.spec.name
del temp_env

num_episodes = 600
num_test_runs_per_experiment = 20

experiment = core.Experiment( "benchmark_dqn_"+env_name+str(num_episodes),
	[
		[0,1,2,3,4,5,6,7,8,9]
	] )
experiment.plot()

settings = { "hidden_layer_sizes" : [128,64,32] }

while experiment.iterate():
	this_experiment = experiment.experiment
	settings["experiment"] = this_experiment[0]
	print(f"Experiment {settings['experiment']}")
	algo = dqn.Algo( create_env_fn, settings=settings)
	#algo.visualize()
	algo.loop_episodes(num_episodes, visualize_every=0, show_graph=True)
	print(f"Running {num_test_runs_per_experiment} test episodes for benchmarking")
	results = []
	for test in range(num_test_runs_per_experiment):
		last_step_reward, steps, episode_reward = algo.do_episode_test(seed=test)
		results.append(episode_reward)
	print(f"Episode rewards = {results}")
	experiment.experiment_completed(results)
	#experiment.plot()

experiment.plot(block=True)
		

