# -*- coding: utf-8 -*-
import os
import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 学習環境の準備
env = make_vec_env('Hopper-v3', n_envs=8)
# モデルの準備
model = PPO("MlpPolicy", env, verbose=1)
# 学習の実行
model.learn(total_timesteps=10000)

model.save("/tmp/ppo_hopper")

del env, model
env = gym.make('Hopper-v3')
model = PPO.load("/tmp/ppo_hopper")
# model = PPO("MlpPolicy", env, verbose=1)

# 推論の実行
obs = env.reset()
start_time = time.time()
for _ in range(10000):
    # 推論の実行
    action, _states = model.predict(obs, deterministic=True)
    # 1step実行
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    # 学習環境の描画
    env.render()

    end_time = time.time()
    if float(end_time-start_time) < env.dt:
        time.sleep(env.dt-float(end_time-start_time))
    start_time = end_time

