from ddpg_torch import Agent
import gym
import numpy as np
from utils import plot_learning_curve

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
                batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []
for i in range(1000):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action = agent.choose_action(obs)
        new_state, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
            '100 game average %.2f' % np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()

filename = 'lunar-lander.png'
# TODO: plot function