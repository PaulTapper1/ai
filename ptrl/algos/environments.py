import gymnasium as gym
import algos.env_LunarLanderMod as env_LunarLanderMod
import algos.env_MountainCarContinuousMod as env_MountainCarContinuousMod

# to see a list of all available gyms use...
#	import gymnasium as gym
#	gym.pprint_registry()
# for more info see https://gymnasium.farama.org/environments

def get_env_name_from_create_fn(create_env_fn):
	temp_env = create_env_fn()
	env_name = temp_env.spec.name
	del temp_env
	return env_name

class EnvDummySpec():
	def __init__(self, name):
		self.name = name
		self.id = name

def create_env_fn_CartPole(render_mode=None):
	return gym.make("CartPole-v1",render_mode=render_mode)		
def create_env_fn_LunarLander(render_mode=None):
	return gym.make("LunarLander-v3",render_mode=render_mode)	
def create_env_fn_MountainCar(render_mode=None):
	return gym.make("MountainCar-v0",render_mode=render_mode)	
def create_env_fn_LunarLanderWithWind(render_mode=None):
	env = gym.make("LunarLander-v3",render_mode=render_mode, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5)	
	env.spec.name = "LunarLanderWithWind"
	return env
def create_env_fn_LunarLanderModWithWind(render_mode=None):
	env = env_LunarLanderMod.LunarLanderMod(render_mode=render_mode, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5)	
	env.spec = EnvDummySpec("LunarLanderModWithWind")
	return env
def create_env_fn_LunarLanderContinuous(render_mode=None):
	return gym.make("LunarLanderContinuous-v3",render_mode=render_mode)	
def create_env_fn_LunarLanderContinuousWithWind(render_mode=None):
	env = env_LunarLanderMod.LunarLanderMod(render_mode=render_mode, continuous=True, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5)	
	env.spec = EnvDummySpec("LunarLanderContinuousWithWind")
	return env
def create_env_fn_LunarLanderContinuousToDiscreteWithWind(render_mode=None):
	env = env_LunarLanderMod.LunarLanderMod(render_mode=render_mode, continuous=True, gravity=-10.0, enable_wind=True, wind_power=15.0, turbulence_power=1.5, continuous_to_discrete=True)	
	env.spec = EnvDummySpec("LunarLanderContinuousToDiscreteWithWind")
	return env
def create_env_fn_MountainCarContinuous(render_mode=None):
	return gym.make("MountainCarContinuous-v0",render_mode=render_mode)	
def create_env_fn_MountainCarContinuousMod(render_mode=None):
	env = env_MountainCarContinuousMod.Continuous_MountainCarEnvMod(render_mode=render_mode)
	env.spec = EnvDummySpec("MountainCarContinuousMod")
	return env
def create_env_fn_BipedalWalker(render_mode=None):
	return gym.make("BipedalWalker-v3",render_mode=render_mode)


