import algos.core as core
import pathlib
import numpy as np
algo_name = pathlib.Path(__file__).stem

class Algo(core.AlgoBase):
	def __init__(self, create_env_fn, settings=[]):
		super().__init__(name=algo_name, create_env_fn=create_env_fn, settings=settings)
		match self.env_name:
			case "CartPole":
				self.actor = ActorHandwritten_CartPole()
			case "LunarLander" | "LunarLanderWithWind" | "LunarLanderModWithWind":
				self.actor = ActorHandwritten_LunarLander()
			case _:
				core.print_error(f"Unable to find correct handwritten actor for env_name {self.env_name}")
		self.load_if_save_exists()

class ActorHandwritten_CartPole(core.ActorCore):
	def select_action(self, observation):
		# custom hand-written control code
		# extract observation into human readable form
		cart_position		= observation[0]
		cart_velocity		= observation[1]
		pole_angle			= observation[2]
		pole_angle_velocity = observation[3]

		# # simple minded move left if tilting left, move right if tilting right
		# action = np.array([ 0 if pole_angle<0 else 1 ]) 

		# account for angle and angle velocity (worked pretty well)
		angle_comb = pole_angle + pole_angle_velocity
		action =  0 if angle_comb<0 else 1
		return action

class ActorHandwritten_LunarLander(core.ActorCore):
	def select_action(self, observation):
		# # https://gymnasium.farama.org/environments/box2d/lunar_lander/
		# # custom hand-written control code
		# # extract observation into human readable form
		# pos_x			= observation[0]
		# pos_y			= observation[1]
		# vel_x			= observation[2]
		# vel_y			= observation[3]
		# ang				= observation[4]
		# ang_vel			= observation[5]
		# leg_contact_l	= observation[6]
		# leg_contact_r	= observation[7]
		# wind			= observation[8]
		# torque			= observation[9]

		# # action space
		# # 0: do nothing
		# # 1: fire left orientation engine
		# # 2: fire main engine
		# # 3: fire right orientation engine

		# # control variables
		# boost_up = 2
		# target_width_x = 0.05
		# panic_angle = 0.5
		# turn_angle = 0.3
		# landing_max_speed = 0.1
		# landing_approach_y = 0.5

		# future_ang = ang + ang_vel + torque*0.1
		# future_pos_x = pos_x + vel_x + wind*0.1

		# action = 0
		# if leg_contact_l>0 or leg_contact_r>0:
			# action = 0
			# if vel_y < -landing_max_speed:
				# action = 2
		# else:
			# if  future_ang > panic_angle:
				# action = 3
			# elif future_ang < -panic_angle:
				# action = 1
			# elif pos_y + vel_y*boost_up < 0:
				# action = 2
			# elif pos_y < landing_approach_y and vel_y < 0 and (future_pos_x < -target_width_x*2 or future_pos_x > target_width_x*2):
				# action = 2
			# else:
				# target_ang = 0
				# if future_pos_x < -target_width_x:
					# target_ang = -turn_angle
				# elif future_pos_x > target_width_x:
					# target_ang = turn_angle
				# if ang + ang_vel > target_ang:
					# action = 3
				# else:
					# action = 1
		# return action

		# code from LunarLander.heuristic
		s = observation
		angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
		if angle_targ > 0.4:
			angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
		if angle_targ < -0.4:
			angle_targ = -0.4
		hover_targ = 0.55 * np.abs(
			s[0]
		)  # target y should be proportional to horizontal offset

		angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
		hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

		if s[6] or s[7]:  # legs have contact
			angle_todo = 0
			hover_todo = (
				-(s[3]) * 0.5
			)  # override to reduce fall speed, that's all we need after contact

		if False: #env.continuous:
			a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
			a = np.clip(a, -1, +1)
		else:
			a = 0
			if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
				a = 2
			elif angle_todo < -0.05:
				a = 3
			elif angle_todo > +0.05:
				a = 1
		return a
