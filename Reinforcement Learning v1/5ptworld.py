import numpy as np
import matplotlib.pyplot as plt

n_states = 5
n_actions = 2
goal_state = 4

learning_rate = .8
discount_factor = .95
exploration_prob = .2
epochs = 100000

Q_table = np.zeros((n_states, n_actions), dtype=float)

# im on a line with 5 possible positions -> 0, 1, 2, 3, 4
# start at pos 0, i wanna end at 4
# actions; 0= move left, 1= move right
#rewards: +1 if you reach position 4, 0 otherwise

#algorithm: Q(s,a) = Q(s,a) + learning_rate * (reward + discount * max(Q(next_state)) - Q(s,a))

def get_next_state(state, action):
    if action == 0:
        return max(0, state - 1)
    elif action == 1:
        return min(n_states - 1, state + 1)

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
q_values_max = np.max(Q_table, axis=1).reshape((1, 5))  # 1 row, 5 columns

plt.figure(figsize=(6, 2))
plt.imshow(q_values_max, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Q-value')
plt.title('Max Q-values per state (1D world)')
plt.xticks(range(5), ['0','1','2','3','4'])
plt.yticks([0], ['row 0'])
plt.show()
policy = np.argmax(Q_table, axis=1)
print(policy)
