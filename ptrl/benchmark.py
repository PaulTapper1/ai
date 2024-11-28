import gymnasium as gym
import algos.core as core
import algos.environments as env

show_graphs = True
#show_graphs = False

############################################
# select the learning algorithm to use
import algos.dqn as algo_type
#import algos.dqn_frame_mem as algo_type
#import algos.handwritten as algo_type

############################################
# select any meta algorithm
import algos.meta_keep_best as meta_algo_type

############################################
# select environment to train and test on
#create_env_fn = env.create_env_fn_LunarLanderWithWind
create_env_fn = env.create_env_fn_LunarLanderModWithWind

env_name = env.get_env_name_from_create_fn(create_env_fn)
num_episodes = 2000
num_test_episodes_per_experiment = 20

def run_experiment(settings):
	experiment_savename = "experiment_"+core.generic_get_save_name(meta_algo_type.meta_algo_name+"_"+algo_type.algo_name, env_name, settings)+"_ep"+str(num_episodes)
	experiment = core.Experiment( experiment_savename,
		[
			[0,1,2,3,4,5,6,7,8,9]
		] )
	if show_graphs and len(experiment.completed_experiments) > 0:
		experiment.plot()
	while experiment.iterate():
		this_experiment = experiment.experiment
		settings["experiment"] = this_experiment[0]
		print(f"Experiment {settings['experiment']}")
		algo = algo_type.Algo( create_env_fn, settings=settings)
		meta_algo = meta_algo_type.MetaAlgo( create_env_fn, settings=settings, algo=algo)

		# if algo.steps_done == 0:
			# print("preloading with succesful actor from LunarLander no wind")
			# temp_saver = core.Saver("LunarLander_no_wind_128_64_32")
			# temp_saver.load_data_into("actor", algo.actor.mlp, is_net=True)
		
		#algo.visualize(num_episodes=10)
		#if meta_algo.algo.steps_done > 0:
		#	meta_algo.visualize(num_episodes=10)
		
		meta_algo.loop_episodes(num_episodes=num_episodes, visualize_every=0, show_graph=show_graphs)

		results = meta_algo.test_actor(num_test_episodes=num_test_episodes_per_experiment, seed_offset=int(1e6))
		print(f"Episode rewards = {results}")
		experiment.experiment_completed(results)
		if show_graphs:
			experiment.plot()
	print(f"{experiment_savename} finished")
	experiment.plot(block=True, save_image=True)

def visualizer_actor_from_run(savename):
	actor = core.MLPActorDiscreteActions(create_env_fn, hidden_layer_sizes=[256,128,64,32])
	saver = core.Saver(savename)
	saver.load_data_into("best_actor", actor.mlp, True)
	actor.test(create_env_fn=create_env_fn, num_test_episodes=10, visualize=True) #, seed_offset=1e6)


# run_experiment( { "hidden_layer_sizes" : [64,32,16] } )
# run_experiment( { "hidden_layer_sizes" : [32,16,8] } )
# run_experiment( { "hidden_layer_sizes" : [16,8,4] } )
# run_experiment( { "hidden_layer_sizes" : [64,32] } )
# run_experiment( { "hidden_layer_sizes" : [256,256,256] } )
run_experiment( { "hidden_layer_sizes" : [256,128,64,32] } )
		
#visualizer_actor_from_run("meta_keep_best_dqn_LunarLanderModWithWind_256_128_64_32_ex3")
