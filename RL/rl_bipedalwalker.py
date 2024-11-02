#pip install swig
#pip install "gymnasium[box2d]"
# I had to (from https://stackoverflow.com/questions/54252800/python-cant-install-box2d-swig-exe-failed-with-error-code-1)...
# - download swig here : http://www.swig.org/download.html
# - Extract the archive somewhere, add the path of the extracted directory to your PATH environment variable.
# - Restart your cmd console window (close it and reopen it), do your pip install Box2D


import rl_utils as rl
import rl_dqn

# # to see a list of all available gyms use...
# import gymnasium as gym
# gym.pprint_registry()

# https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
gym_name = "BipedalWalker-v3"  
#gym_name = "BipedalWalkerHardcore-v3"  


def creat_env_fn(render_mode=None):
  return rl_dqn.PT_GYM(gym_name, render_mode=render_mode)

settings_iterator = rl.SettingsIterator( [
    [128],         # linear layer0
    [128],         # linear layer1
    [128],         # linear layer2
    #["A"]
    ["A","B","C","D","E","F"]
    #["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
  ] )

num_episodes = 1000
while settings_iterator.iterate():
  ptdqn = rl_dqn.PT_DQN(creat_env_fn, settings_iterator.settings)
  #ptdqn.visualize_model(5)
  ptdqn.loop_episodes(num_episodes,visualize_every=10)

ptdqn.plot_progress(True)
