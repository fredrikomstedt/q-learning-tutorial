import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

rewards = []
win_percentages = []

policy = {
    0: 1, 
    1: 2,
    2: 1,
    3: 0,
    4: 1,
    6: 1,
    8: 2,
    9: 1,
    10: 1,
    13: 2,
    14: 2
}

for i in range(1000):
    done = False
    print("Episode {}".format(i+1))
    obs = env.reset()
    score = 0
    while not done:
        obs, reward, done, info = env.step(policy[obs])
        score += reward
    rewards.append(score)

    if i % 10 == 0:
        average = np.mean(rewards[-10:])
        win_percentages.append(average)
env.close()

plt.plot(win_percentages)
plt.show()