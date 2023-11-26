import gymnasium as gym
import numpy as np
from actorCritic import ActorCritic
from utils import plot_learning_curve
from datetime import datetime
import atexit
import sys

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    agent = ActorCritic(alpha=1e-4, gamma=0.90, action_space=env.action_space.n)
    n_games = 1500

    filePath = "tmp/img/LunarLander"+"-"+str(datetime.now())+"-g"+str(agent.gamma)+"-a"+str(agent.alpha)+".png"
    figureFile = filePath
    highScore = env.reward_range[0]
    scoreHistory = []
    loadCheckpoint = False

    if loadCheckpoint:
        agent.load()
    for i in range (n_games):
        print(i)
        observation = env.reset()[0]
        done = False
        score = 0
        try:
            while not done:
                action = agent.chooseAction(observation)
                observation1, reward, done, _, info = env.step(action)
                score += reward
                if not loadCheckpoint:
                    agent.learn(observation, reward, observation1, done)
                observation = observation1
        except (KeyboardInterrupt, AssertionError, ValueError):
            print(action)
            x = [j+1 for j in range (i)]
            plot_learning_curve(x, scoreHistory, figureFile)
            sys.exit()

        scoreHistory.append(score)
        averageScore = np.mean(scoreHistory[-100:])

        if i % 100 == 0 and i >= 100:
            x = [j+1 for j in range (i)]
            plot_learning_curve(x, scoreHistory, figureFile)

        if averageScore > highScore:
            highScore = averageScore
            if not loadCheckpoint:
                agent.save()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % averageScore)

    x = [i+1 for i in range (n_games)]
    plot_learning_curve(x, scoreHistory, figureFile)
