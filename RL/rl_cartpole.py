import rl_utils as rl
import rl_dqn

# # to see a list of all available gyms use...
# import gymnasium as gym
# gym.pprint_registry()

gym_name = "CartPole-v1"  # https://gymnasium.farama.org/environments/classic_control/cart_pole/

def creat_env_fn(render_mode=None):
  return rl_dqn.PT_GYM(gym_name, render_mode=render_mode)

settings_iterator = rl.SettingsIterator( [
    [128],         # linear layer0
    [128],         # linear layer1
    [128],         # linear layer1
    #["A"]
    ["A","B","C","D","E","F"]
    #["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
  ] )

num_episodes = 1000
#num_episodes = 200
while settings_iterator.iterate():
  ptdqn = rl_dqn.PT_DQN(creat_env_fn, settings_iterator.settings)
  ptdqn.visualize_model(10)
  ptdqn.loop_episodes(num_episodes)

ptdqn.plot_progress(True)
  
  #ptdqn.visualize_model_hardwired_cartpole()
