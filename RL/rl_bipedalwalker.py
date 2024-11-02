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
# https://github.com/openai/gym/blob/master/gym/envs/box2d/bipedal_walker.py
gym_name = "BipedalWalker-v3"  
#gym_name = "BipedalWalkerHardcore-v3"  

# customise the environment
class PT_GYM_Custom(rl_dqn.PT_GYM):
  def __init__(self, render_mode):
    super().__init__(gym_name, render_mode)
    self.reward_scale_x = 1
    self.reward_scale_y = 0 #0.1
  def reset(self):
    self.pos_x = 0
    self.pos_y = 0
    self.steps = 0
    return self.env.reset()
  def step(self, action):
    observation, reward, terminated, truncated, _ =  self.env.step(action)
    
    self.steps += 1
    vel_x = observation[2]
    vel_y = observation[3]
    self.pos_x += vel_x
    self.pos_y += vel_y
    
    reward += self.pos_x * self.reward_scale_x + self.pos_y * self.reward_scale_y
    #print(f"step({self.steps}): (x,y) = ({self.pos_x:0.2f},{self.pos_y:0.2f}), reward = {reward:0.3f}")
    
    if self.pos_x < self.steps*0.01-3:
      truncated = True
      reward -= 50
      #print(f"PT_GYM_Custom.step({self.steps}): truncated. (x,y) = ({self.pos_x:0.2f},{self.pos_y:0.2f}), reward = {reward:0.3f}")
    
    # if truncated:
      # print("Episode truncated")
    # else:
      # if terminated:
        # if reward > 50: # if landed
          # reward += 50 - self.num_steps * self.speed_reward  # reduce reward by long it took to land, to encourage fast landing
          # print(f"Landed successfully (step reward = {reward})")
        # else:
          # print(f"Crashed (step reward = {reward})")
    return observation, reward, terminated, truncated, _
def create_env_fn_custom(render_mode=None):
  return PT_GYM_Custom(render_mode)

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
  ptdqn = rl_dqn.PT_DQN(create_env_fn_custom, settings_iterator.settings)
  #ptdqn.visualize_model(5)
  ptdqn.loop_episodes(num_episodes, visualize_every=0)

ptdqn.plot_progress(True)
