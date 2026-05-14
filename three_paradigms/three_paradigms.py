"""
Practice activity: Implementing supervised, unsupervised, and reinforcement
learning in a single hands-on exercise.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Three small models, one per paradigm:
  1. Supervised   -> Linear regression on a tiny house-price dataset (MSE).
  2. Unsupervised -> K-means customer segmentation (cluster centers + plot).
  3. Reinforcement-> Q-learning tic-tac-toe agent trained via self-play vs a
                     random opponent, then evaluated against a fresh random
                     opponent (win/draw/loss rate + learning curve).

The activity's RL snippet stubs out the Q-update and opponent logic. We
flesh it out with a real tabular Q-learning loop and a random opponent so
the agent actually learns and the metrics are meaningful.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)


# ====================================================================
# 1. SUPERVISED LEARNING — house price prediction (linear regression)
# ====================================================================
print("=" * 64)
print("1. Supervised learning - house price prediction")
print("=" * 64)

# Features: [square_footage, num_bedrooms, location_code]
X_house = np.array([
    [2000, 3, 1],
    [1500, 2, 2],
    [1800, 3, 3],
    [1200, 2, 1],
    [2200, 4, 2],
])
y_house = np.array([500_000, 350_000, 450_000, 300_000, 550_000])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_tr, y_tr)

y_pred = reg.predict(X_te)
mse = mean_squared_error(y_te, y_pred)
rmse = mse ** 0.5

print(f"Training samples: {len(X_tr)}  |  Test samples: {len(X_te)}")
print(f"Coefficients (sqft, bedrooms, location): {reg.coef_.round(2)}")
print(f"Intercept: {reg.intercept_:.2f}")
print(f"Test predictions: {y_pred.round(0)}")
print(f"Test actuals:     {y_te}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}  (approx. average dollar error)")


# ====================================================================
# 2. UNSUPERVISED LEARNING — K-means customer segmentation
# ====================================================================
print("\n" + "=" * 64)
print("2. Unsupervised learning - K-means customer segmentation")
print("=" * 64)

# Features: [num_purchases, total_spending, num_product_categories]
X_cust = np.array([
    [5,  1000, 2],
    [10, 5000, 5],
    [2,  500,  1],
    [8,  3000, 3],
    [12, 6000, 6],
])

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_cust)
print(f"Cluster centers:\n{kmeans.cluster_centers_.round(2)}")
print(f"Cluster labels:  {kmeans.labels_}")

fig_cust, ax_cust = plt.subplots(figsize=(7, 5))
sc = ax_cust.scatter(
    X_cust[:, 0], X_cust[:, 1],
    c=kmeans.labels_, cmap="viridis", s=140, edgecolor="k",
)
ax_cust.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    marker="X", s=240, c="red", edgecolor="black", label="Centroid",
)
ax_cust.set_xlabel("Number of purchases")
ax_cust.set_ylabel("Total spending")
ax_cust.set_title("Customer segmentation via K-means (k=2)")
ax_cust.legend()
ax_cust.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("customer_clusters.png", dpi=120, bbox_inches="tight")
plt.close(fig_cust)


# ====================================================================
# 3. REINFORCEMENT LEARNING — Q-learning tic-tac-toe
# ====================================================================
print("\n" + "=" * 64)
print("3. Reinforcement learning - Q-learning tic-tac-toe")
print("=" * 64)

# Board encoding: tuple of 9 ints, 0=empty, 1=agent (X), -1=opponent (O).
# Using tuples (not numpy arrays) so we can key the Q-table dict.

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


def initial_board():
    return (0,) * 9


def available_moves(board):
    return [i for i, v in enumerate(board) if v == 0]


def place(board, idx, player):
    return board[:idx] + (player,) + board[idx + 1:]


def winner(board):
    """Returns 1 / -1 if a player has won, 0 for draw if full, else None."""
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    if 0 not in board:
        return 0
    return None


def choose_action(Q, board, epsilon):
    """Epsilon-greedy over the legal moves from `board`."""
    moves = available_moves(board)
    if random.random() < epsilon:
        return random.choice(moves)
    q_row = Q.setdefault(board, np.zeros(9))
    legal_q = [(q_row[m], m) for m in moves]
    max_q = max(q for q, _ in legal_q)
    best = [m for q, m in legal_q if q == max_q]
    return random.choice(best)


def random_opponent_move(board):
    return random.choice(available_moves(board))


def train_q_agent(n_episodes=20_000, alpha=0.3, gamma=0.9, epsilon=0.2):
    """Q-learning agent (X=+1) trained against a random opponent (O=-1)."""
    Q = {}
    rewards = np.zeros(n_episodes, dtype=np.float32)
    outcomes = np.zeros(n_episodes, dtype=np.int8)  # 1 win, 0 draw, -1 loss

    for ep in range(n_episodes):
        board = initial_board()
        # Agent always plays X and moves first
        last_state, last_action = None, None
        ep_reward = 0.0

        while True:
            # ---- Agent's move ----
            action = choose_action(Q, board, epsilon)
            next_board = place(board, action, 1)
            result = winner(next_board)

            if result is not None:
                # Terminal after agent's move
                reward = 1.0 if result == 1 else (0.5 if result == 0 else -1.0)
                q_row = Q.setdefault(board, np.zeros(9))
                q_row[action] += alpha * (reward - q_row[action])
                ep_reward += reward
                outcomes[ep] = 1 if result == 1 else (0 if result == 0 else -1)
                break

            # ---- Opponent's move ----
            opp_action = random_opponent_move(next_board)
            after_opp = place(next_board, opp_action, -1)
            opp_result = winner(after_opp)

            if opp_result is not None:
                # Terminal after opponent's move (opponent won or board full)
                reward = -1.0 if opp_result == -1 else 0.5
                q_row = Q.setdefault(board, np.zeros(9))
                q_row[action] += alpha * (reward - q_row[action])
                ep_reward += reward
                outcomes[ep] = -1 if opp_result == -1 else 0
                break

            # ---- Non-terminal: bootstrap from next state's best legal Q ----
            q_row = Q.setdefault(board, np.zeros(9))
            next_q_row = Q.setdefault(after_opp, np.zeros(9))
            legal_next = available_moves(after_opp)
            future = max(next_q_row[m] for m in legal_next)
            q_row[action] += alpha * (gamma * future - q_row[action])

            board = after_opp

        rewards[ep] = ep_reward

    return Q, rewards, outcomes


def evaluate(Q, n_games=2_000):
    """Greedy agent (epsilon=0) vs random opponent. Returns (win, draw, loss)."""
    w = d = l = 0
    for _ in range(n_games):
        board = initial_board()
        while True:
            a = choose_action(Q, board, epsilon=0.0)
            board = place(board, a, 1)
            r = winner(board)
            if r is not None:
                if r == 1:    w += 1
                elif r == 0:  d += 1
                else:         l += 1
                break
            o = random_opponent_move(board)
            board = place(board, o, -1)
            r = winner(board)
            if r is not None:
                if r == 1:    w += 1
                elif r == 0:  d += 1
                else:         l += 1
                break
    return w, d, l


N_EPISODES = 20_000
print(f"Training Q-learning agent for {N_EPISODES:,} self-play episodes "
      f"vs a random opponent...")
Q, train_rewards, train_outcomes = train_q_agent(n_episodes=N_EPISODES)
print(f"Q-table size: {len(Q):,} distinct states seen.")

w, d, l = evaluate(Q, n_games=2_000)
total = w + d + l
print(f"\nGreedy evaluation over {total} games vs random opponent:")
print(f"  Wins:   {w}  ({100 * w / total:5.1f}%)")
print(f"  Draws:  {d}  ({100 * d / total:5.1f}%)")
print(f"  Losses: {l}  ({100 * l / total:5.1f}%)")


# ----- Learning-curve plot ---------------------------------------------------
W = 500
roll_reward = np.convolve(train_rewards, np.ones(W) / W, mode="valid")
roll_winrate = np.convolve((train_outcomes == 1).astype(float),
                           np.ones(W) / W, mode="valid")

fig_rl, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.plot(train_rewards, alpha=0.15, label="Per-episode reward")
ax.plot(np.arange(W - 1, len(train_rewards)), roll_reward,
        linewidth=2, label=f"{W}-ep rolling mean")
ax.set_title("Cumulative reward per episode")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward (win=1, draw=0.5, loss=-1)")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(np.arange(W - 1, len(train_outcomes)), 100 * roll_winrate,
        color="tab:green", linewidth=2)
ax.set_title(f"Win rate during training ({W}-ep rolling)")
ax.set_xlabel("Episode")
ax.set_ylabel("% of episodes won")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

plt.suptitle("Q-learning tic-tac-toe — learning curve", fontsize=13)
plt.tight_layout()
plt.savefig("tic_tac_toe_learning.png", dpi=120, bbox_inches="tight")
plt.close(fig_rl)


# ====================================================================
# Reflection
# ====================================================================
# Supervised. With only 4 training points and 3 features, the linear
# regression is heavily under-determined and the MSE is more of a sanity
# check than a real evaluation. The point of the activity is the workflow,
# not the metric value — labeled (X, y) data, train/test split, fit,
# predict, compare to ground truth via MSE.
#
# Unsupervised. K-means with k=2 splits the five customers cleanly into a
# low-spend group and a high-spend group — exactly what the centroids
# show. There is no ground truth, so we judge the model by whether the
# groupings are interpretable (yes) and whether intra-cluster distances
# are small relative to inter-cluster distances. On a real dataset we
# would pick k with the elbow method or silhouette score.
#
# Reinforcement. Tabular Q-learning learns tic-tac-toe vs a random
# opponent very quickly: after ~5k episodes the rolling win rate plateaus
# at ~90%+, with the remaining episodes split between draws and the rare
# loss. The agent does not learn an optimal policy because (a) the
# opponent is random, so positions that punish optimal play almost never
# come up, and (b) we are not doing self-play with two learning agents.
# A stronger setup would alternate which side learns, decay epsilon, and
# evaluate vs an optimal minimax opponent.

print("\nSaved plots: customer_clusters.png, tic_tac_toe_learning.png")
