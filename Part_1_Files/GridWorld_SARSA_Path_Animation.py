import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.colors as mcolors

class GridWorldEnv:
    def __init__(self):
        self.grid_size = 5
        self.action_space = ['up', 'down', 'left', 'right']  # 4 possible actions
        self.red_states = [(2, 0), (2, 1), (2, 3), (2, 4)]
        self.blue_state = (4, 0)
        self.black_states = [(0, 0), (0, 4)]
        
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

def render_grid(ax, state, env, heatmap=None):
    ax.clear()
    grid = np.zeros((env.grid_size, env.grid_size), dtype=int)
    
    for red_state in env.red_states:
        grid[red_state] = 1  # Red state
    
    grid[env.blue_state] = 2  # Blue state
    
    for black_state in env.black_states:
        grid[black_state] = 3  # Black state
    
    current_position = state
    grid[current_position] = 4  # Agent's position
    
    # Color map
    color_map = mcolors.ListedColormap(['white', 'red', 'blue', 'black', 'green'])
    
    if heatmap is not None:
        ax.imshow(heatmap, cmap='hot', interpolation='none', alpha=0.5)
    
    ax.imshow(grid, cmap=color_map, interpolation='none', alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Agent's Position: {current_position}")

def epsilon_greedy_policy(Q, state, epsilon, action_space):
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    else:
        return action_space[np.argmax([Q[state][action] for action in action_space])]

def run_sarsa(env, episodes, alpha, gamma, epsilon):
    Q = {state: {action: 0 for action in env.action_space} for state in [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]}
    visit_counter = {state: 0 for state in [(i, j) for i in range(env.grid_size) for j in range(env.grid_size)]}
    
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

    return Q, visit_counter

def run_episode(env, Q):
    state = env.reset()
    env.history = [state]
    
    while not env.done:
        action = max(Q[state], key=Q[state].get)  # Select action with the highest Q-value
        next_state, reward, done = env.step(action)
        state = next_state

    return env.history

def create_heatmap(env, visit_counter):
    heatmap = np.zeros((env.grid_size, env.grid_size))
    
    for state, count in visit_counter.items():
        heatmap[state] = count
        
    return heatmap

def animate_episode(history, env, heatmap):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.3)

    # Initial rendering
    render_grid(ax, history[0], env, heatmap)

    # Slider setup
    ax_slider = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Step', 0, len(history) - 1, valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)
        render_grid(ax, history[step], env, heatmap)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Play button setup
    ax_play = plt.axes([0.1, 0.1, 0.2, 0.075])
    ax_pause = plt.axes([0.4, 0.1, 0.2, 0.075])
    button_play = Button(ax_play, 'Play')
    button_pause = Button(ax_pause, 'Pause')

    # Global variable for playback control
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

    # Animation function
    def animate(i):
        if playing:
            step = i % len(history)
            slider.set_val(step)
            update(step)
        return []

    # Create an animation
    ani = FuncAnimation(fig, animate, frames=len(history), interval=200, blit=True)

    plt.show()

# Create environment
env = GridWorldEnv()

# Run SARSA to train the Q-values and track visits
episodes = 300
alpha = 0.1
gamma = 0.9
epsilon = 0.1

Q, visit_counter = run_sarsa(env, episodes, alpha, gamma, epsilon)

# Generate heatmap from visit counter
heatmap = create_heatmap(env, visit_counter)

# Animate an episode using the learned Q-values
history = run_episode(env, Q)
animate_episode(history, env, heatmap)

