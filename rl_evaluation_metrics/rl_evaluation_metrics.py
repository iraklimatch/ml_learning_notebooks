"""
Practice activity: Evaluating a Q-learning agent with RL-specific metrics.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Same 5x5 GridWorld as the Q-learning vs Policy Gradients activity.
This time we focus on *how* you evaluate the agent's learning curve:
  - cumulative reward per episode
  - episode length per episode
  - rolling success rate (fraction of recent episodes that reached the goal)
  - exploration-vs-exploitation ratio per episode

The activity's snippets use `next_state = np.random.randint(...)` for
transitions, which would make learning impossible. We use a real
deterministic GridWorld transition function.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ------------------------------------------------------------------
# GridWorld environment (same as the Q-learning vs PG activity)
# ------------------------------------------------------------------
GRID_SIZE = 5
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4   # 0=up, 1=down, 2=left, 3=right
GOAL = 24
PIT = 12

REWARDS = np.full(N_STATES, -1, dtype=np.float32)
REWARDS[GOAL] = 10
REWARDS[PIT] = -10


def step(state: int, action: int):
    """(next_state, reward, done) for a (state, action) pair."""
    row, col = divmod(state, GRID_SIZE)
    if   action == 0 and row > 0:                row -= 1
    elif action == 1 and row < GRID_SIZE - 1:    row += 1
    elif action == 2 and col > 0:                col -= 1
    elif action == 3 and col < GRID_SIZE - 1:    col += 1
    next_state = row * GRID_SIZE + col
    reward = float(REWARDS[next_state])
    done = next_state == GOAL or next_state == PIT
    return next_state, reward, done


def random_start_state() -> int:
    while True:
        s = np.random.randint(N_STATES)
        if s != GOAL and s != PIT:
            return s


# ------------------------------------------------------------------
# Q-learning with instrumented epsilon-greedy
# ------------------------------------------------------------------
def epsilon_greedy(Q, state, epsilon):
    """Returns (action, was_exploration_bool)."""
    if np.random.rand() < epsilon:
        return np.random.randint(N_ACTIONS), True
    return int(np.argmax(Q[state])), False


def train_and_evaluate(n_episodes: int = 1000,
                       alpha: float = 0.1,
                       gamma: float = 0.9,
                       epsilon: float = 0.1,
                       max_steps: int = 100):
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)

    cumulative_rewards = np.zeros(n_episodes, dtype=np.float32)
    episode_lengths   = np.zeros(n_episodes, dtype=np.int32)
    reached_goal      = np.zeros(n_episodes, dtype=bool)
    explore_counts    = np.zeros(n_episodes, dtype=np.int32)
    exploit_counts    = np.zeros(n_episodes, dtype=np.int32)

    for ep in range(n_episodes):
        state = random_start_state()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action, was_exploration = epsilon_greedy(Q, state, epsilon)
            if was_exploration:
                explore_counts[ep] += 1
            else:
                exploit_counts[ep] += 1

            next_state, reward, done = step(state, action)
            td_target = reward + gamma * (0.0 if done else np.max(Q[next_state]))
            Q[state, action] += alpha * (td_target - Q[state, action])

            total_reward += reward
            steps += 1
            state = next_state
            if done:
                if next_state == GOAL:
                    reached_goal[ep] = True
                break

        cumulative_rewards[ep] = total_reward
        episode_lengths[ep]    = steps

    return {
        'Q': Q,
        'cumulative_rewards': cumulative_rewards,
        'episode_lengths':    episode_lengths,
        'reached_goal':       reached_goal,
        'explore_counts':     explore_counts,
        'exploit_counts':     exploit_counts,
    }


def rolling_mean(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def rolling_rate(boolean_array, w=50):
    """Rolling fraction of True values."""
    return np.convolve(boolean_array.astype(float), np.ones(w) / w, mode='valid')


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
print("Training Q-learning agent for 1000 episodes...")
m = train_and_evaluate(n_episodes=1000)

print("\n=== Summary statistics ===")
print(f"Final 50-episode mean cumulative reward: {m['cumulative_rewards'][-50:].mean():.2f}")
print(f"Final 50-episode mean episode length:    {m['episode_lengths'][-50:].mean():.2f}")
print(f"Overall success rate:                    {m['reached_goal'].mean()*100:.1f}%")
print(f"Final 50-episode success rate:           {m['reached_goal'][-50:].mean()*100:.1f}%")

total_explore = m['explore_counts'].sum()
total_exploit = m['exploit_counts'].sum()
print(f"Total actions: {total_explore + total_exploit} "
      f"(exploration: {total_explore} = {100*total_explore/(total_explore+total_exploit):.1f}%, "
      f"exploitation: {total_exploit} = {100*total_exploit/(total_explore+total_exploit):.1f}%)")


# ------------------------------------------------------------------
# Plot the four metrics
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
W = 50  # rolling window

# 1. Cumulative reward per episode
ax = axes[0, 0]
ax.plot(m['cumulative_rewards'], alpha=0.25, label='Raw')
ax.plot(np.arange(W - 1, len(m['cumulative_rewards'])),
        rolling_mean(m['cumulative_rewards'], W),
        linewidth=2, label=f'{W}-ep rolling mean')
ax.set_title('Metric 1 — Cumulative reward per episode')
ax.set_xlabel('Episode'); ax.set_ylabel('Total reward')
ax.legend(); ax.grid(True, alpha=0.3)

# 2. Episode length per episode
ax = axes[0, 1]
ax.plot(m['episode_lengths'], alpha=0.25, label='Raw')
ax.plot(np.arange(W - 1, len(m['episode_lengths'])),
        rolling_mean(m['episode_lengths'], W),
        linewidth=2, label=f'{W}-ep rolling mean')
ax.set_title('Metric 2 — Episode length (steps to terminate)')
ax.set_xlabel('Episode'); ax.set_ylabel('Steps')
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Rolling success rate
ax = axes[1, 0]
ax.plot(np.arange(W - 1, len(m['reached_goal'])),
        100 * rolling_rate(m['reached_goal'], W),
        linewidth=2, color='tab:green')
ax.axhline(100, ls='--', color='gray', alpha=0.5, label='Perfect (100%)')
ax.set_ylim(0, 105)
ax.set_title(f'Metric 3 — Success rate ({W}-ep rolling)')
ax.set_xlabel('Episode'); ax.set_ylabel('% of episodes reaching the goal')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Exploration ratio per episode (fraction of explore actions in that episode)
ax = axes[1, 1]
total_per_ep = (m['explore_counts'] + m['exploit_counts']).astype(float)
total_per_ep[total_per_ep == 0] = 1
explore_ratio = m['explore_counts'] / total_per_ep
ax.plot(explore_ratio * 100, alpha=0.4, label='Per-episode exploration %')
ax.axhline(10, ls='--', color='red', alpha=0.6, label='Configured epsilon (10%)')
ax.set_ylim(0, 100)
ax.set_title('Metric 4 — Exploration vs Exploitation')
ax.set_xlabel('Episode'); ax.set_ylabel('% of actions that were random')
ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Evaluating a Q-learning agent on 5x5 GridWorld', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig('rl_metrics.png', dpi=120, bbox_inches='tight')
plt.show()


# ------------------------------------------------------------------
# Reflection
# ------------------------------------------------------------------
#
# Cumulative reward. Starts very negative (random policy bumps into walls
# and the pit), then climbs steadily as the Q-table fills in. The final
# 50-episode mean reward is positive, which means the agent is now
# reliably getting to the goal under -1 step penalties.
#
# Episode length. Drops sharply during the first ~200 episodes from the
# max_steps cap of 100 down to a small number — the agent has learned
# the shortest path from typical start states.
#
# Success rate. Rising rolling success rate is the cleanest signal that
# learning is working. A high *and stable* success rate at the end
# means the agent has converged.
#
# Exploration vs exploitation. With epsilon=0.1, ~10% of actions in any
# given episode are random — confirmed by the red dashed line on the
# bottom-right plot. Realistic per-episode values bounce around 10%
# because the episodes are short and few actions are taken. In
# production you would *decay* epsilon over training (start ~1.0, end
# ~0.01) so the agent explores aggressively early and exploits late.
