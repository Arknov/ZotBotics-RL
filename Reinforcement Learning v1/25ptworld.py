import numpy as np
import matplotlib.pyplot as plt

n_states = 25
n_actions = 4
goal_state = 24

learning_rate = .8
discount_factor = .95
exploration_prob = .2
epochs = 100000

Q_table = np.zeros((n_states, n_actions), dtype=float)

# im on a line with 25 possible positions -> 0 to 24
# start at pos 0, i wanna end at 24
# actions; 0= move left, 1= move right 2=move up, 3= move down
#rewards: +1 if you reach position 24, 0 otherwise

#algorithm: Q(s,a) = Q(s,a) + learning_rate * (reward + discount * max(Q(next_state)) - Q(s,a))

def get_next_state(state, action):
    row, col = divmod(state, 5)  

    if action == 0 and col > 0:        # left
        col -= 1
    elif action == 1 and col < 4:      # right
        col += 1
    elif action == 2 and row > 0:      # up
        row -= 1
    elif action == 3 and row < 4:      # down
        row += 1
    return row * 5 + col

for epoch in range(epochs):
    current_state = 0
    while True:
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(Q_table[current_state])
        next_state = get_next_state(current_state, action)

        reward = 1 if next_state == goal_state else 0

        #mathing Q(s,a) = Q(s,a) + learning_rate * (reward + discount * max(Q(next_state)) - Q(s,a))
        Q_table[current_state, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state,action])

        if next_state == goal_state:
            break
        current_state = next_state


#qtable
print(Q_table)
print(Q_table.shape)
q_values_max = np.max(Q_table, axis=1).reshape((5, 5))  # 1 row, 5 columns

plt.figure(figsize=(6, 6))
plt.imshow(q_values_max, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Q-value')
plt.title('Max Q-values per state (2D world)')
plt.xticks(range(5), ['0','1','2','3','4'])
plt.yticks(range(5), ['0','1','2','3','4'])
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{q_values_max[i,j]:.2f}', ha='center', va='center', color='black')
policy = np.argmax(Q_table, axis=1)

arrow_map = {0:'←', 1:'→', 2:'↑', 3:'↓', -1:'G'}
policy[goal_state] = -1  # mark goal
policy_grid = np.array([arrow_map[a] for a in policy]).reshape((5,5))
print(policy_grid)

plt.show()

