import algos.core as core
import time

def view_graph_for_training_run(savename,**kwargs):
	saver = core.Saver(savename)
	logger = core.Logger(savename)
	saver.load_data_into( "logger", 			logger )
	#logger.plot(data_to_plot=["episode_reward", "most_recent_av_test_score", "last_step_reward","episode_durations","memory_size","epsilon"])
	logger.plot(data_to_plot=["episode_reward",["most_recent_actor_score","best_actor_score"],"episodes_before_revert_to_best_cursor","last_step_reward","episode_durations","epsilon"],**kwargs)
	

#core.Experiment( "benchmark_dqn_LunarLander300", [[0]]).plot()
#core.Experiment( "benchmark_dqn_LunarLander500", [[0]]).plot()
#core.Experiment( "benchmark_dqn_frame_mem_LunarLanderWithWind1000", [[0]]).plot(block=True)

# view_graph_for_training_run("meta_keep_best_dqn_LunarLanderModWithWind_256_128_64_32_ex0")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex1")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex2")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex3")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex4")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex5")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex6")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex7")
# view_graph_for_training_run("dqn_LunarLanderWithWind_256_128_64_32_ex8")
view_graph_for_training_run("sac_LunarLanderContinuousWithWind_256_256_ex3", block=True)

# print("Pausing")
# while True:
	# time.sleep(1)