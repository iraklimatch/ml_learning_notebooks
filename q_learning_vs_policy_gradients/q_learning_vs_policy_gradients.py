"""
Practice activity: Q-learning vs Policy Gradients on a 5x5 GridWorld.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Note on the activity's snippets: the prescribed code uses
`next_state = np.random.randint(...)` for transitions, which means the
agent's actions have no effect and nothing can be learned. We replace
that with a proper deterministic GridWorld transition function so the
comparison between Q-learning and policy gradients is meaningful.

Environment:
  - 5x5 grid, 25 states (positions 0..24, row-major).
  - 4 actions: 0=up, 1=down, 2=left, 3=right.
  - Goal at position 24 (bottom-right): +10 reward, episode ends.
  - Pit at position 12 (center): -10 reward, episode ends.
  - Every other step: -1 reward, episode continues.
  - Hitting a wall keeps the agent in place but still costs -1.
"""

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # quiet TF info logs

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)


# ------------------------------------------------------------------
# GridWorld environment
# ------------------------------------------------------------------
GRID_SIZE = 5
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
GOAL = 24
PIT = 12

# Rewards lookup for the agent landing on a given state
REWARDS = np.full(N_STATES, -1.0, dtype=np.float32)
REWARDS[GOAL] = 10.0
REWARDS[PIT] = -10.0


def step(state: int, action: int):
    """Return (next_state, reward, done) for a (state, action) pair."""
    row, col = divmod(state, GRID_SIZE)

    if action == 0 and row > 0:                # up
        row -= 1
    elif action == 1 and row < GRID_SIZE - 1:  # down
        row += 1
    elif action == 2 and col > 0:              # left
        col -= 1
    elif action == 3 and col < GRID_SIZE - 1:  # right
        col += 1
    # else: action runs into a wall; position unchanged

    next_state = row * GRID_SIZE + col
    reward = float(REWARDS[next_state])
    done = next_state == GOAL or next_state == PIT
    return next_state, reward, done


def random_start_state() -> int:
    """Random start, never start on the goal or in the pit."""
    while True:
        s = np.random.randint(N_STATES)
        if s != GOAL and s != PIT:
            return s


# ==================================================================
# Q-LEARNING
# ==================================================================
def train_q_learning(n_episodes: int = 1000,
                     alpha: float = 0.1,
                     gamma: float = 0.9,
                     epsilon: float = 0.1,
                     max_steps: int = 100):
    """Tabular Q-learning with epsilon-greedy exploration."""
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    episode_returns = []

    for _ in range(n_episodes):
        state = random_start_state()
        total_reward = 0.0
        for _ in range(max_steps):
            if np.random.uniform() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = step(state, action)
            td_target = reward + gamma * (0.0 if done else np.max(Q[next_state]))
            Q[state, action] += alpha * (td_target - Q[state, action])

            total_reward += reward
            state = next_state
            if done:
                break

        episode_returns.append(total_reward)

    return Q, np.array(episode_returns)


# ==================================================================
# POLICY GRADIENTS (REINFORCE) with TensorFlow
# ==================================================================
def build_policy_network() -> tf.keras.Model:
    """A tiny softmax policy: one-hot state -> 24 hidden units -> 4 action probs."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_STATES,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(N_ACTIONS, activation='softmax'),
    ])


def compute_returns(rewards, gamma=0.99):
    """Discounted returns: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ..."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train_policy_gradient(n_episodes: int = 1000,
                          gamma: float = 0.99,
                          lr: float = 0.01,
                          max_steps: int = 100):
    model = build_policy_network()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    episode_returns = []

    for _ in range(n_episodes):
        states, actions, rewards = [], [], []
        state = random_start_state()

        # --- Roll out one episode under the current policy
        for _ in range(max_steps):
            state_oh = tf.one_hot(state, N_STATES)[None, :]
            probs = model(state_oh, training=False).numpy()[0]
            action = int(np.random.choice(N_ACTIONS, p=probs))

            next_state, reward, done = step(state, action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        episode_returns.append(float(np.sum(rewards)))

        # --- REINFORCE update on the collected episode
        returns = compute_returns(rewards, gamma)
        # Baseline subtraction for variance reduction
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        state_inputs = tf.one_hot(states, N_STATES)
        action_masks = tf.one_hot(actions, N_ACTIONS)
        returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            action_probs = model(state_inputs, training=True)
            # Negative log-likelihood weighted by return (REINFORCE loss)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs + 1e-8), axis=1)
            loss = -tf.reduce_mean(log_probs * returns_tf)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model, np.array(episode_returns)


# ==================================================================
# RUN AND COMPARE
# ==================================================================
N_EPISODES = 1000

print("Training Q-learning...")
Q, q_returns = train_q_learning(n_episodes=N_EPISODES)
print(f"  Final 50-episode mean return: {q_returns[-50:].mean():.2f}")

print("\nTraining policy gradient (REINFORCE)...")
pg_model, pg_returns = train_policy_gradient(n_episodes=N_EPISODES)
print(f"  Final 50-episode mean return: {pg_returns[-50:].mean():.2f}")


# Smooth with a rolling window so the noisy episode-by-episode returns
# don't drown out the trend.
def rolling_mean(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(q_returns, alpha=0.25, label='Q-learning (raw)')
axes[0].plot(np.arange(49, len(q_returns)),
             rolling_mean(q_returns), label='Q-learning (50-ep rolling mean)', linewidth=2)
axes[0].plot(pg_returns, alpha=0.25, label='Policy gradient (raw)')
axes[0].plot(np.arange(49, len(pg_returns)),
             rolling_mean(pg_returns), label='Policy gradient (50-ep rolling mean)', linewidth=2)
axes[0].set_title('Episode return over time')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Greedy policy heatmap learned by Q-learning
greedy_actions = np.argmax(Q, axis=1).reshape(GRID_SIZE, GRID_SIZE)
arrows = np.array(['↑', '↓', '←', '→'])[greedy_actions]
arrows_flat = arrows.flatten()
for i, (s, ar) in enumerate(zip(range(N_STATES), arrows_flat)):
    if s == GOAL:
        arrows_flat[i] = 'G'
    elif s == PIT:
        arrows_flat[i] = 'P'
arrows = arrows_flat.reshape(GRID_SIZE, GRID_SIZE)

axes[1].imshow(np.max(Q, axis=1).reshape(GRID_SIZE, GRID_SIZE), cmap='viridis')
axes[1].set_title('Q-learning value map (max Q per state, arrow = greedy action)')
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        axes[1].text(c, r, arrows[r, c], ha='center', va='center',
                     color='white', fontsize=16, fontweight='bold')
axes[1].set_xticks(range(GRID_SIZE))
axes[1].set_yticks(range(GRID_SIZE))

plt.suptitle('Q-learning vs Policy Gradients on 5x5 GridWorld', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('q_vs_pg.png', dpi=120, bbox_inches='tight')
plt.show()


# Summary stats
print("\n=== Summary ===")
print(f"{'Method':<28s} {'Final mean return':>18s} {'Best return':>14s}")
print(f"{'Q-learning (1000 eps)':<28s} {q_returns[-50:].mean():>18.2f} {q_returns.max():>14.2f}")
print(f"{'Policy gradient (1000 eps)':<28s} {pg_returns[-50:].mean():>18.2f} {pg_returns.max():>14.2f}")


# Reflection
#
# - Convergence speed. Q-learning typically converges much faster on
#   small tabular tasks like this: the table has only 25*4 = 100 entries
#   and TD updates propagate value backward efficiently. REINFORCE needs
#   to estimate gradients from noisy Monte-Carlo returns, so it shows
#   higher variance and slower convergence.
#
# - Reward maximization. Both reach positive returns near the end of
#   training. Q-learning's final policy lands much closer to optimal on
#   this fully-discrete problem.
#
# - Exploration. Q-learning uses an explicit epsilon-greedy mechanism;
#   the policy gradient explores by sampling actions from its current
#   softmax distribution. As the policy sharpens, exploration drops on
#   its own — there's no temperature schedule here, so PG can lock onto
#   suboptimal behavior if early gradients point the wrong way.
#
# - Suitability. Q-learning is the right tool for small *discrete* state
#   and action spaces. Policy gradients shine when the action space is
#   continuous (robot joint torques) or when the state space is too big
#   for a table (raw pixels). For this 25-state world, Q-learning is
#   simpler and faster; the PG implementation here is mainly pedagogical.
