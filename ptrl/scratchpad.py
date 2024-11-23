
# # load back in an actor and test it in an environment
# import gymnasium as gym
# import algos.core as core
# import algos.dqn as algo_type
# def create_env_fn(render_mode=None):
	# return gym.make("LunarLander-v3",render_mode=render_mode)	
	# #return gym.make("LunarLander-v3",render_mode=render_mode, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5)	
# settings = { "hidden_layer_sizes" : [128,64,32] }
# actor = core.MLPActorDiscreteActions(create_env_fn, hidden_layer_sizes=settings["hidden_layer_sizes"])
# saver = core.Saver("LunarLander_no_wind_128_64_32")
# saver.load_data_into("actor", actor.mlp, is_net=True)
# actor.visualize(create_env_fn, num_episodes = 5)


import pathlib
print("Current script name = "+pathlib.Path(__file__).stem)