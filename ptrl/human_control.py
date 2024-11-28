import gymnasium as gym
import algos.core as core
import algos.environments as env

import algos.human_control as algo_type

############################################
# select environment to train and test on
create_env_fn = env.create_env_fn_LunarLander
#create_env_fn = env.create_env_fn_LunarLanderModWithWind

algo = algo_type.Algo( create_env_fn )
algo.visualize(num_episodes=10)
