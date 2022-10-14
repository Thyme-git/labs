import gym
import numpy as np

env = gym.make('CartPole-v1')

num_episodes = 3000


q_table = np.random.uniform(low=-1, high=1, size=(256, env.action_space.n))


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):

    cart_pos, cart_v, pole_angle, pole_v = observation[0], observation[1], observation[2], observation[3]

    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]

    return sum([x * (4 ** i) for i, x in enumerate(digitized)])


def get_action(state, action, observation, reward, episode, train):
    next_state = digitize_state(observation)
    epsilon = 0.5 * (0.99 ** episode)
    if  epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    if train:
        alpha = 0.2
        gamma = 0.99
        q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])
    return next_action, next_state


for episode in range(num_episodes):
    observation = env.reset()[0]
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(2000):
        observation, reward, done, _ , _= env.step(action)
        action, state = get_action(state, action, observation, reward, episode, True)
        episode_reward += reward
        
        if done:
            reward = -100

        action, state = get_action(state, action, observation, reward, episode, True)
        
        if done:
            print('%d Episode finished after %d steps' % (episode, t + 1))
            break

np.savetxt("./q_table.txt", q_table)
env.close()

env = gym.make('CartPole-v1', render_mode = 'human')
observation, info = env.reset(seed=42)
for _ in range(2000):
    action, state = get_action(state, action, observation, reward, episode, False)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
