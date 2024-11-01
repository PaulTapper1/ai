import rl_utils as rl
import rl_dqn

# # to see a list of all available gyms use...
# import gymnasium as gym
# gym.pprint_registry()

gym_name = "CartPole-v1"  # https://gymnasium.farama.org/environments/classic_control/cart_pole/

settings_iterator = rl.SettingsIterator( [
    [128],         # linear layer0
    [128],         # linear layer1
    #["A"]
    ["A","B","C","D","E","F"]
    #["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
  ] )

#num_episodes = 600
num_episodes = 100
while settings_iterator.iterate():
  ptdqn = rl_dqn.PT_DQN(gym_name, settings_iterator.settings)
  ptdqn.loop_episodes(num_episodes)
  #ptdqn.visualize_model()

ptdqn.plot_progress(True)
  
  #ptdqn.visualize_model_hardwired_cartpole()
