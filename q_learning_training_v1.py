from q_learning_v1 import CalicoEnv1, QLearningAgent, compute_score

env = CalicoEnv1(scoring_function=compute_score)
agent = QLearningAgent(action_size=(25*2))

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        if env.mode == "placing":
            action_space = 25 * 2
        else:
            action_space = 2 * 3
        agent.action_size = action_space

        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

    print(f"Episode {episode}: total reward = {total_reward}")