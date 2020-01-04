import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

rewards = []
win_percentages = []
for i in range(1000):
    done = False
    print("Episode {}".format(i+1))
    env.reset()
    score = 0
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        score += reward
    rewards.append(score)

    if i % 10 == 0:
        average = np.mean(rewards[-10:])
        win_percentages.append(average)
env.close()

plt.plot(win_percentages)
plt.show()