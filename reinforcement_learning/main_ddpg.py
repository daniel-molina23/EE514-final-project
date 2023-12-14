import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve, MatrixOperations, DFS
from Env import QuantumEnvironment

if __name__ == '__main__':
    size = lambda boxSpace : boxSpace.shape[0]
    num_layers = 7

    expectedStates, inputStates = DFS().getInitialTargetStates()
    mat_op_obj = MatrixOperations(expectedStates, inputStates,num_layers)
    env = QuantumEnvironment(num_layers=num_layers, num_of_wanted_identity_gates=3, matrix_operator_obj=mat_op_obj)
    agent = Agent(input_dims=size(env.observation_space), env=env,
            n_actions=size(env.action_space))
    n_games = 250

    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

