import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors

class GridWorldEnv:
    def __init__(self):
        self.grid_size = 11
        self.action_space = ['up', 'down', 'left', 'right']  # 4 possible actions
        self.red_states = [(3,4),(3,6),(2,4),(2,6),(1,4),(1,6),(5,4),(5,6),(4,4),(4,6),(6,4),(6,6),(7,4),(7,6),(8,4),(8,6)]
        self.blue_state = (10, 5)
        self.black_states = [(0, 5)]  # (5,5) with reward of 25, (0,0) with reward of 0
        
        self.reset()
    
    def reset(self):
        self.state = self.blue_state
        self.total_reward = 0
        self.steps = 0
        self.done = False
        self.history = [self.state]
        return self.state
    
    def step(self, action):
        if self.done:
            return self.state, self.total_reward, self.done
        
        # Define action directions
        action_directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        next_state = (self.state[0] + action_directions[action][0], 
                      self.state[1] + action_directions[action][1])
        
        # Check if next_state is out of bounds
        if not (0 <= next_state[0] < self.grid_size and 0 <= next_state[1] < self.grid_size):
            next_state = self.state
            reward = -1
        elif next_state in self.red_states:
            next_state = self.blue_state
            reward = -20
        elif next_state in self.black_states:
            self.done = True
            reward = 0
        else:
            reward = -1
        
        self.state = next_state
        self.total_reward += reward
        self.steps += 1
        self.history.append(self.state)
        return self.state, reward, self.done

def epsilon_greedy_policy(Q, state, epsilon, action_space):
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return action_space[np.argmax([Q[state][action] for action in action_space])]

def run_sarsa(env, episodes, alpha, gamma, epsilon):
    Q = {state: {action: 0 for action in env.action_space} for state in [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]}
    visit_counter = {state: 0 for state in [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]}
    
    Q_snapshots = []
    visit_snapshots = []
    
    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon, env.action_space)
        
        while not env.done:
            visit_counter[state] += 1
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.action_space)
            
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            
            state = next_state
            action = next_action
        
        Q_snapshots.append({state: q.copy() for state, q in Q.items()})
        visit_snapshots.append(visit_counter.copy())

    return Q_snapshots, visit_snapshots

def run_q_learning(env, episodes, alpha, gamma, epsilon):
    Q = {state: {action: 0 for action in env.action_space} for state in [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]}
    visit_counter = {state: 0 for state in [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]}
    
    Q_snapshots = []
    visit_snapshots = []
    
    for _ in range(episodes):
        state = env.reset()
        
        while not env.done:
            visit_counter[state] += 1
            action = epsilon_greedy_policy(Q, state, epsilon, env.action_space)
            next_state, reward, done = env.step(action)
            
            best_next_action = max(Q[next_state], key=Q[next_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
            
            state = next_state
        
        Q_snapshots.append({state: q.copy() for state, q in Q.items()})
        visit_snapshots.append(visit_counter.copy())

    return Q_snapshots, visit_snapshots

def create_heatmap(Q, visit_counter, env):
    heatmap = np.zeros((env.grid_size, env.grid_size, 4))  # RGBA

    max_visit = max(visit_counter.values()) if visit_counter.values() else 1

    for (i, j), q_values in Q.items():
        max_q = max(q_values.values())
        min_q = min(q_values.values())
        normalized_value = (max_q - min_q) / (1 if max_q == min_q else max_q - min_q)
        
        color = plt.cm.cool(normalized_value)  # Map to cool colormap
        alpha = visit_counter[(i, j)] / max_visit if max_visit != 0 else 0
        
        heatmap[i, j, :3] = color[:3]
        heatmap[i, j, 3] = alpha
    
    return heatmap

def render_grid(ax, heatmap, env):
    ax.clear()
    grid = np.zeros((env.grid_size, env.grid_size, 4))
    
    # Draw initial grid with outlines
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            grid[i, j] = [1, 1, 1, 1]  # White with full opacity
    
    for red_state in env.red_states:
        grid[red_state] = [1, 0, 0, 1]  # Red outline
    
    grid[env.blue_state] = [0, 0, 1, 1]  # Blue outline
    
    for black_state in env.black_states:
        if black_state == (5, 5):
            grid[black_state] = [0, 1, 0, 1]  # Green outline for reward 25
        else:
            grid[black_state] = [0, 0, 0, 1]  # Black outline for reward 0
    
    ax.imshow(grid, interpolation='none')
    ax.imshow(heatmap, interpolation='none', alpha=0.8)  # Overlay heatmap
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Q-Value Heatmap")

def render_static_grid(env):
    fig, ax = plt.subplots()
    grid = np.zeros((env.grid_size, env.grid_size, 4))
    
    # Draw initial grid with outlines
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            grid[i, j] = [1, 1, 1, 1]  # White with full opacity
    
    for red_state in env.red_states:
        grid[red_state] = [1, 0, 0, 1]  # Red outline
    
    grid[env.blue_state] = [0, 0, 1, 1]  # Blue outline
    
    for black_state in env.black_states:
        if black_state == (5, 5):
            grid[black_state] = [0, 1, 0, 1]  # Green outline for reward 25
        else:
            grid[black_state] = [0, 0, 0, 1]  # Black outline for reward 0
    
    ax.imshow(grid, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Grid World Environment")
    plt.show()

def animate_heatmap(Q_snapshots, visit_snapshots, env):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.3)

    heatmap = create_heatmap(Q_snapshots[0], visit_snapshots[0], env)
    render_grid(ax, heatmap, env)

    ax_slider = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Episode', 0, len(Q_snapshots) - 1, valinit=0, valstep=1)

    def update(val):
        episode = int(slider.val)
        heatmap = create_heatmap(Q_snapshots[episode], visit_snapshots[episode], env)
        render_grid(ax, heatmap, env)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    ax_play = plt.axes([0.1, 0.1, 0.2, 0.075])
    ax_pause = plt.axes([0.4, 0.1, 0.2, 0.075])
    button_play = Button(ax_play, 'Play')
    button_pause = Button(ax_pause, 'Pause')

    playing = False

    def play(event):
        nonlocal playing
        playing = True
        ani.event_source.start()

    def pause(event):
        nonlocal playing
        playing = False
        ani.event_source.stop()

    button_play.on_clicked(play)
    button_pause.on_clicked(pause)

    def animate(i):
        if playing:
            episode = i % len(Q_snapshots)
            slider.set_val(episode)
            update(episode)
        return []

    ani = FuncAnimation(fig, animate, frames=len(Q_snapshots), interval=200, blit=True)

    plt.show()

# Create environment
env = GridWorldEnv()

# Parameters
episodes = 50000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Run SARSA to train the Q-values and track visits
Q_snapshots_sarsa, visit_snapshots_sarsa = run_sarsa(env, episodes, alpha, gamma, epsilon)

# Run Q-learning to train the Q-values and track visits
Q_snapshots_q_learning, visit_snapshots_q_learning = run_q_learning(env, episodes, alpha, gamma, epsilon)

# Animate the heatmap over episodes for SARSA
print("SARSA:")
animate_heatmap(Q_snapshots_sarsa, visit_snapshots_sarsa, env)

# Animate the heatmap over episodes for Q-learning
print("Q-learning:")
animate_heatmap(Q_snapshots_q_learning, visit_snapshots_q_learning, env)

# Show just the grid by itself
render_static_grid(env)

