import gymnasium as gym
import algos.core as core
import algos.environments as env

show_graphs = True
#show_graphs = False

############################################
# select the learning algorithm to use
#import algos.dqn as algo_type
#import algos.dqn_frame_mem as algo_type
import algos.sac as algo_type
#import algos.handwritten as algo_type

############################################
# select any meta algorithm (or comment them all out to not use any)
import algos.meta_keep_best as meta_algo_type
#import algos.meta_explore_rewards as meta_algo_type

############################################
# select any algorithm modifiers
#import algos.mod_explore_rewards

############################################
# select environment to train and test on
#gym.pprint_registry()
#create_env_fn = env.create_env_fn_CartPole
#create_env_fn = env.create_env_fn_LunarLander
#create_env_fn = env.create_env_fn_LunarLanderWithWind
#create_env_fn = env.create_env_fn_LunarLanderModWithWind
#create_env_fn = env.create_env_fn_LunarLanderContinuous
#create_env_fn = env.create_env_fn_LunarLanderContinuousWithWind
#create_env_fn = env.create_env_fn_LunarLanderContinuousToDiscreteWithWind
#create_env_fn = env.create_env_fn_MountainCarContinuous
#create_env_fn = env.create_env_fn_MountainCarContinuousMod
create_env_fn = env.create_env_fn_BipedalWalker

env_name = env.get_env_name_from_create_fn(create_env_fn)
num_episodes = 5000
num_test_episodes_per_experiment = 20

def run_experiment(settings):
	using_meta_algo = ("meta_algo_type" in globals())
	if using_meta_algo:
		experiment_savename = core.generic_get_save_name(meta_algo_type.meta_algo_name+"_"+algo_type.algo_name, env_name, settings)+"_ep"+str(num_episodes)
	else:
		experiment_savename = core.generic_get_save_name(algo_type.algo_name, env_name, settings)+"_ep"+str(num_episodes)
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
		if using_meta_algo:
			algo = meta_algo_type.MetaAlgo( create_env_fn, settings=settings, algo=algo)
		#mod_explore_rewards = algos.mod_explore_rewards.Modifier(algo)
		
		#algo.visualize(num_episodes=5)
		if algo.steps_done > 0:
			algo.visualize(num_episodes=5)
		
		algo.loop_episodes(num_episodes=num_episodes, visualize_every=0, show_graph=show_graphs)
		results = algo.test_actor(num_test_episodes=num_test_episodes_per_experiment, seed_offset=int(1e6))
		
		print(f"Episode rewards = {results}")
		experiment.experiment_completed(results)
		if show_graphs:
			experiment.plot()
	print(f"{experiment_savename} finished")
	experiment.plot(block=True, save_image=True)

def visualizer_actor_from_run(savename, settings):
	algo = algo_type.Algo( create_env_fn, settings=settings)
	algo.saver.filename = savename
	algo.load()
	algo.visualize(num_episodes=10)
	# #actor = core.MLPActorDiscreteActions(create_env_fn, hidden_layer_sizes=[256,128,64,32])
	# actor = core.MLPActorCritic(create_env_fn, hidden_layer_sizes=[256,256])
	# saver = core.Saver(savename)
	# actor.load_data_into( "actor", saver)
	# actor.visualize(create_env_fn=create_env_fn, num_episodes=10) #, seed_offset=1e6)

settings = { "hidden_layer_sizes" : [256,256] }
# settings = { "hidden_layer_sizes" : [128,64,32] }
# settings = { "hidden_layer_sizes" : [64,32,16] }
# settings = { "hidden_layer_sizes" : [32,16,8] }
# settings = { "hidden_layer_sizes" : [16,8,4] }
# settings = { "hidden_layer_sizes" : [128,64] }
# settings = { "hidden_layer_sizes" : [256,256,256] }
# settings = { "hidden_layer_sizes" : [256,128,64,32] }

#run_experiment(settings=settings)
		
visualizer_actor_from_run(savename="meta_keep_best_sac_BipedalWalker_256_256_ex3", settings=settings)
