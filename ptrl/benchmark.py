import gymnasium as gym
import algos.core as core
import algos.environments as env

# select the learning algorithm to use
import algos.dqn as algo_type
#import algos.dqn_frame_mem as algo_type
#import algos.handwritten as algo_type

create_env_fn = env.create_env_fn_LunarLanderWithWind

env_name = env.get_env_name_from_create_fn(create_env_fn)
num_episodes = 1000
num_test_runs_per_experiment = 20

def run_experiment(settings):
	experiment = core.Experiment( "benchmark_"+algo_type.algo_name+"_"+env_name+str(num_episodes),
		[
			[0,1,2,3,4,5,6,7,8,9]
		] )
	if len(experiment.completed_experiments) > 0:
		experiment.plot()
	while experiment.iterate():
		this_experiment = experiment.experiment
		settings["experiment"] = this_experiment[0]
		print(f"Experiment {settings['experiment']}")
		algo = algo_type.Algo( create_env_fn, settings=settings)

		# if algo.steps_done == 0:
			# print("preloading with succesful actor from LunarLander no wind")
			# temp_saver = core.Saver("LunarLander_no_wind_128_64_32")
			# temp_saver.load_data_into("actor", algo.actor.mlp, is_net=True)
		
		#algo.visualize(num_episodes=5)
		algo.loop_episodes(num_episodes, visualize_every=0, show_graph=True)
		
		print(f"Running {num_test_runs_per_experiment} test episodes for benchmarking")
		results = []
		for test in range(num_test_runs_per_experiment):
			last_step_reward, steps, episode_reward = algo.do_episode_test(seed=test)
			results.append(episode_reward)
		
		print(f"Episode rewards = {results}")
		experiment.experiment_completed(results)
		experiment.plot()
	#experiment.plot(block=True)

# run_experiment( { "hidden_layer_sizes" : [64,32,16] } )
# run_experiment( { "hidden_layer_sizes" : [32,16,8] } )
# run_experiment( { "hidden_layer_sizes" : [16,8,4] } )
# run_experiment( { "hidden_layer_sizes" : [64,32] } )
# run_experiment( { "hidden_layer_sizes" : [256,256,256] } )

run_experiment( { "hidden_layer_sizes" : [256,128,64,32] } )
		

