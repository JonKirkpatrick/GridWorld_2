import numpy as np
import matplotlib.pyplot as plt

class GridWorldEnv:
    def __init__(self):
        self.grid_size = 7
        self.action_space = ['up', 'down', 'left', 'right']  # 4 possible actions
        self.terminal_states = {(6, 0): -1, (0, 6): 1}
        self.start_state = (3, 3)
        self.reset()
    
    def reset(self):
        self.state = self.start_state
        self.total_reward = 0
        self.steps = 0
        self.done = False
        return self.state
    
    def step(self, action):
        if self.done:
            return self.state, self.total_reward, self.done
        
        action_directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        next_state = (self.state[0] + action_directions[action][0], 
                      self.state[1] + action_directions[action][1])
        
        if not (0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size):
            next_state = self.state
            reward = 0
        elif next_state in self.terminal_states:
            self.done = True
            reward = self.terminal_states[next_state]
        else:
            reward = 0
        
        self.state = next_state
        self.total_reward += reward
        self.steps += 1
        return self.state, reward, self.done

def get_decayed_alpha(alpha_0, beta, episode):
    return alpha_0 / (1 + beta * episode)

def feature_vector(state):
    features = np.array([1,state[0]+6-state[1],state[1]+6-state[0]])
    return features

def gradient_monte_carlo(env, episodes, alpha_0, gamma, beta):
    # Initialize weights
    w = np.zeros(3)
    
    for episode in range(episodes):
        alpha = get_decayed_alpha(alpha_0, beta, episode)
        state = env.reset()
        trajectory = []
        while not env.done:
            action = np.random.choice(env.action_space)
            next_state, reward, done = env.step(action)
            trajectory.append(state)
            state = next_state
        
        T = len(trajectory)
        for t, state in enumerate(reversed(trajectory)):
            # Compute G_t as gamma^(T-t-1) * R_T
            G_t = gamma**(T-t-1)*env.total_reward
            
            phi_s = feature_vector(state)
            v_s = np.dot(w, phi_s)
            # Update weights
            w += alpha * (G_t - v_s) * phi_s
    
    return w

def compute_value_function(w, grid_size):
    V = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            phi_s = feature_vector((i, j))
            V[i, j] = np.dot(w, phi_s)
    return V

def compute_exact_value_function(grid_size, terminal_states):
    V = np.zeros((grid_size, grid_size))
    delta = 1e-5
    while True:
        delta_max = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in terminal_states:
                    continue
                
                v = V[i, j]
                value_sum = 0
                action_directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
                for action in action_directions.values():
                    next_state = (i + action[0], j + action[1])
                    if not (0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size):
                        next_state = (i, j)
                    
                    if next_state in terminal_states:
                        value_sum += 0.25 * terminal_states[next_state]
                    else:
                        value_sum += 0.25 * V[next_state[0], next_state[1]]
                
                V[i, j] = value_sum
                delta_max = max(delta_max, abs(v - V[i, j]))
        
        if delta_max < delta:
            break
    return V

def plot_value_function(ax, V, title):
    im = ax.imshow(V, cmap='cool', interpolation='none')
    for (i, j), val in np.ndenumerate(V):
        ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

# Create environment
env = GridWorldEnv()

# Parameters
episodes = 50000
alpha = 0.000002
gamma = 1.0
beta = 0

# Run Gradient Monte Carlo with function approximation
w_mc = gradient_monte_carlo(env, episodes, alpha, gamma, beta)
V_mc = compute_value_function(w_mc, env.grid_size)
V_exact = compute_exact_value_function(env.grid_size, env.terminal_states)

# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

plot_value_function(axs[0], V_mc, "Gradient Monte Carlo")
plot_value_function(axs[1], V_exact, "Exact Value Function")
#plt.imshow(V_mc, cmap='cool', interpolation='none')
#plt.colorbar()
#plt.title("Gradient Monte Carlo Value Function")
plt.show()



