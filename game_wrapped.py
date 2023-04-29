import doodle_jump_game
import gym
import numpy as np


class DoodleEnv(gym.Env):
    def __init__(self, render=False, time_step=6, fps=60):
        self.max_score = 20000
        self.game_state = doodle_jump_game.GameState(render, fps)
        self.action_space = gym.spaces.Discrete(3)
        self.time_step = time_step  # RNN单次接收几帧数据输入
        self.peak_score = 0

    def step(self, action):
        # 每次step()将与游戏交互time_step帧，某帧done=True则整体为True
        obsv, r, done_ = self.game_state.frame_step(action)
        for i in range(self.time_step - 1):
            observation, reward, done = self.game_state.frame_step(action)
            obsv = np.vstack((obsv, observation))
            if done:
                done_ = done
                r -= 6
                self.peak_score = self.game_state.peak_score  # 每局游戏结束记录最高分
            else:
                r += reward
        if self.game_state.score >= 3000: r += 6
        if self.game_state.score >= 6000: r += 6
        if self.game_state.score >= self.max_score:
            # print('______win:%s_____' % self.game_state.score)
            self.reset()
            r = 12
            done_ = True
        return obsv, r/6, done_, {}

    def reset(self):
        self.game_state.initialize()
        obsv, r_, d_ = self.game_state.frame_step(0)
        for i in range(self.time_step - 1):
            observation, r_, d_ = self.game_state.frame_step(0)
            obsv = np.vstack((obsv, observation))
        return obsv  # np.asarray(observation)

    # def render(self, mode='human'):
    #     pass


if __name__ == '__main__':  # 并没有通过check_env审查，状态空间会报错
    pass
    # import stable_baselines3
    # from stable_baselines3.common.env_checker import check_env
    # env = DoodleEnv(render=False)
    # env = env.unwrapped
    # check_env(env)
