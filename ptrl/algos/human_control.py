import algos.core as core
import pathlib
algo_name = pathlib.Path(__file__).stem
import pygame

class Algo(core.AlgoBase):
	def __init__(self, create_env_fn, settings=[]):
		super().__init__(name=algo_name, create_env_fn=create_env_fn, settings=settings)
		match self.env_name:
			# case "CartPole":
				# self.actor = ActorHumanControl_CartPole()
			case "LunarLander" | "LunarLanderWithWind" | "LunarLanderModWithWind":
				self.actor = ActorHumanControl_LunarLander()
			case _:
				core.print_error(f"Unable to find correct human control actor for env_name {self.env_name}")
		self.load_if_save_exists()

class ActorHumanControl_LunarLander(core.ActorCore):
	def select_action(self, observation):
		# # https://gymnasium.farama.org/environments/box2d/lunar_lander/
		# # action space
		# # 0: do nothing
		# # 1: fire left orientation engine
		# # 2: fire main engine
		# # 3: fire right orientation engine

		pressed_keys = pygame.key.get_pressed()
		modifiers_bitmask = pygame.key.get_mods()

		action = 0
		if pressed_keys[pygame.K_a]:
			action = 1
		elif pressed_keys[pygame.K_d]:
			action = 3
		elif pressed_keys[pygame.K_SPACE]:
			action = 2

		return action
