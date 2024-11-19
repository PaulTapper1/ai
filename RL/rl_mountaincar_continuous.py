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

gym_name = "MountainCarContinuous-v0" # https://github.com/openai/gym/wiki/MountainCarContinuous-v0

# customise the environment
class PT_GYM_Custom(rl_dqn.PT_GYM):
  def __init__(self, render_mode):
    super().__init__(gym_name, render_mode)
    self.num_steps = 0
    self.speed_reward = 0.1
  def reset(self):
    self.num_steps = 0
    return self.env.reset()
  def step(self, action):
    observation, reward, terminated, truncated, _ =  self.env.step(action)
    self.num_steps += 1
    if truncated:
      print("Episode truncated")
    else:
      if terminated:
        if reward > 50: # if landed
          reward += 50 - self.num_steps * self.speed_reward  # reduce reward by long it took to land, to encourage fast landing
          print(f"Landed successfully (step reward = {reward})")
        else:
          print(f"Crashed (step reward = {reward})")
    return observation, reward, terminated, truncated, _
def creat_env_fn_LunarLander(render_mode=None):
  return PT_GYM_Custom(render_mode)

settings_iterator = rl.SettingsIterator( [
    [128],         # linear layer0
    [64],         # linear layer1
    [32],         # linear layer2
    ["A"]
    #["A","B","C","D","E","F"]
    #["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
  ] )

num_episodes = 10000
while settings_iterator.iterate():
  ptdqn = rl_dqn.PT_DQN(creat_env_fn_LunarLander, settings_iterator.settings)
  ptdqn.visualize_model(1)
  ptdqn.loop_episodes(num_episodes,visualize_every=0)

ptdqn.plot_progress(True)
