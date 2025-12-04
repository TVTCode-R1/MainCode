import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.path import Path
from matplotlib.patches import Circle
import pickle
import os

try:
    from Agent_Traning import (
        WORLD_SIZE, OBSTACLE_SIZE, PREDICTED_TRAJ_LENGTH,
        UNCERTAINTY_RADIUS_INIT, UNCERTAINTY_GROWTH, BOUNDARY_POINTS,
        BOUNDARY_PATH, STATE_DIM, ACTION_DIM, DT, L, AGENT_STATE_DIM,
        OBSTACLE_INFO_DIM, NUM_DYNAMIC_OBSTACLES, bicycle_model_vec,
        get_enhanced_state_vec, update_dynamic_obstacles
    )
except ImportError:
    WORLD_SIZE = 12
    OBSTACLE_SIZE = 0.8
    PREDICTED_TRAJ_LENGTH = 20
    UNCERTAINTY_RADIUS_INIT = 0.5
    UNCERTAINTY_GROWTH = 0.1

    def generate_boundary():
        theta = np.linspace(0, 2 * np.pi, 1000)
        r = 10 + 2 * np.sin(5 * theta) + np.cos(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack((x, y))

    BOUNDARY_POINTS = generate_boundary()
    BOUNDARY_PATH = Path(BOUNDARY_POINTS)

    AGENT_STATE_DIM = 4
    OBSTACLE_INFO_DIM = 6
    NUM_DYNAMIC_OBSTACLES = 2
    STATE_DIM = AGENT_STATE_DIM + NUM_DYNAMIC_OBSTACLES * OBSTACLE_INFO_DIM
    ACTION_DIM = 2
    DT = 0.1
    L = 0.5

TRAINING_GOALS = np.array([[0, 7.5], [7.5, 0], [-5, 0], [0, -5]])
TRAINING_STARTS = np.array([[-5, -5, 0], [-7.5, 2.5, 0], [-5, -10, np.pi / 2], [5, 0, np.pi]])

TEST_GOALS = np.array([[0, 8], [6, 1], [-4, 1], [1, -4]])
TEST_STARTS = np.array([[-4, -4, 0], [-6, 3, 0], [-4, -8, np.pi / 3], [4, 1, np.pi]])

class StateNormalizer:
    def __init__(self):
        self.agent_means = np.array([5.0, 0.0, 1.0, 0.0])
        self.agent_stds = np.array([8.0, np.pi, 2.0, np.pi])
        self.obstacle_means = np.array([0.0, 0.0, 0.0, 0.0, 8.0, 0.5])
        self.obstacle_stds = np.array([12.0, 12.0, 12.0, 12.0, 12.0, 1.0])

    def normalize_state(self, state):
        normalized = np.zeros_like(state)
        normalized[:4] = (state[:4] - self.agent_means) / self.agent_stds
        for i in range(NUM_DYNAMIC_OBSTACLES):
            start_idx = 4 + i * 6
            end_idx = start_idx + 6
            normalized[start_idx:end_idx] = (state[start_idx:end_idx] - self.obstacle_means) / self.obstacle_stds
        return np.clip(normalized, -3, 3)

    def load_parameters(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
            self.agent_means = params['agent_means']
            self.agent_stds = params['agent_stds']
            self.obstacle_means = params['obstacle_means']
            self.obstacle_stds = params['obstacle_stds']
            return True
        except FileNotFoundError:
            return False

def build_enhanced_actor():
    return tf.keras.Sequential([
        layers.Input(shape=(STATE_DIM,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        layers.Dense(ACTION_DIM, activation='tanh')
    ])

class TestTD3Agent:
    def __init__(self, agent_id, state_normalizer):
        self.agent_id = agent_id
        self.actor = build_enhanced_actor()
        self.state_normalizer = state_normalizer

    def get_action(self, state, training=False):
        normalized_state = self.state_normalizer.normalize_state(state)
        action = self.actor(normalized_state[np.newaxis])[0]
        return np.clip(action, -1, 1)

    def load_weights(self, weights_path):
        try:
            self.actor.load_weights(weights_path)
            return True
        except Exception as e:
            return False

def load_trained_agents():
    state_normalizer = StateNormalizer()
    state_normalizer.load_parameters('state_normalizer_params.pkl')

    agents = []
    success_count = 0

    for i in range(4):
        agent = TestTD3Agent(i, state_normalizer)
        weights_file = f'improved_shared_critic_actor_{i}_weights.h5'
        if agent.load_weights(weights_file):
            success_count += 1
        agents.append(agent)

    if success_count == 0:
        return None, None

    return agents, state_normalizer

def create_test_obstacles(config="slight_change"):
    obstacles = []
    configs = []

    if config == "training_identical":
        configs = [
            {"start": np.array([0, -5]), "end": np.array([0, 5])},
            {"start": np.array([-5, 0]), "end": np.array([5, -5])}
        ]
    elif config == "slight_change":
        configs = [
            {"start": np.array([0, -5]), "end": np.array([1, 6])},
            {"start": np.array([-5, 0]), "end": np.array([6, -4])}
        ]
    elif config == "moderate_change":
        configs = [
            {"start": np.array([-5, -10]), "end": np.array([5, 5])},
            {"start": np.array([-5, 5]), "end": np.array([5, -5])}
        ]
    elif config == "crossing":
        configs = [
            {"start": np.array([0, -5]), "end": np.array([0, 5])},
            {"start": np.array([-5, 5]), "end": np.array([5, -5])}
        ]
    elif config == "parallel":
        configs = [
            {"start": np.array([-8, 2]), "end": np.array([8, 2])},
            {"start": np.array([-8, -2]), "end": np.array([8, -2])}
        ]
    else:
        raise ValueError(f"Unknown obstacle configuration: {config}")

    for config_dict in configs:
        start_pos = config_dict["start"]
        end_pos = config_dict["end"]
        traj_points = []

        for t in range(PREDICTED_TRAJ_LENGTH):
            ratio = t / (PREDICTED_TRAJ_LENGTH - 1)
            pos = start_pos * (1 - ratio) + end_pos * ratio
            radius = UNCERTAINTY_RADIUS_INIT + t * UNCERTAINTY_GROWTH
            traj_points.append({'pos': pos.copy(), 'radius': radius})

        obstacles.append({
            'predicted_traj': traj_points,
            'current_step': 0,
            'speed': 0.8
        })

    return obstacles

def test_with_config(trained_agents, goals, starts, obstacles, config_name="test", max_steps=300):
    pos = starts.copy()
    vel = np.zeros(4)
    traj = [deque(maxlen=max_steps) for _ in range(4)]
    frozen = [False, False, False, False]
    success = [False, False, False, False]
    collision_counts = [0, 0, 0, 0]
    step_rewards = []
    
    obstacle_history = [[] for _ in range(len(obstacles))]

    for step in range(max_steps):
        obstacles = update_dynamic_obstacles(obstacles)

        for i, obs in enumerate(obstacles):
            current_pos = obs['predicted_traj'][obs['current_step']]['pos']
            obstacle_history[i].append(current_pos.copy())

        states = get_enhanced_state_vec(pos, vel, pos[:, 2], goals, obstacles)
        actions = np.zeros((4, 2))

        for i in range(4):
            if not frozen[i]:
                actions[i] = trained_agents[i].get_action(states[i])

        new_pos, new_vel, new_theta = bicycle_model_vec(pos.copy(), vel.copy(), pos[:, 2].copy(), actions)
        new_pos[:, 2] = new_theta

        in_boundary = BOUNDARY_PATH.contains_points(new_pos[:, :2])
        for i in range(4):
            if not in_boundary[i]:
                new_pos[i] = pos[i]
                new_vel[i] = 0

        for i in range(4):
            for obs in obstacles:
                obs_pos = obs['predicted_traj'][obs['current_step']]['pos']
                if np.linalg.norm(new_pos[i, :2] - obs_pos) < OBSTACLE_SIZE:
                    collision_counts[i] += 1

        for i in range(4):
            if not frozen[i]:
                goal_dist = np.linalg.norm(new_pos[i, :2] - goals[i, :2])
                if goal_dist < 0.3:
                    frozen[i] = True
                    success[i] = True
                    new_vel[i] = 0
                    actions[i] = np.zeros(2)
                traj[i].append(new_pos[i, :2].copy())
            else:
                traj[i].append(new_pos[i, :2].copy())
                new_vel[i] = 0

        pos, vel = new_pos, new_vel
        step_reward = np.mean([np.linalg.norm(pos[i, :2] - goals[i, :2]) for i in range(4)])
        step_rewards.append(step_reward)

        if all(frozen):
            break

    return {
        'trajectories': [list(t) for t in traj],
        'obstacles': obstacles,
        'obstacle_history': obstacle_history,
        'success': success,
        'collision_counts': collision_counts,
        'steps_taken': step + 1,
        'final_positions': pos[:, :2],
        'final_distances': [np.linalg.norm(pos[i, :2] - goals[i, :2]) for i in range(4)],
        'step_rewards': step_rewards,
        'config_name': config_name
    }

def visualize_test_results(results, goals):
    config_name = results['config_name']
    traj = results['trajectories']
    obstacles = results['obstacles']
    obstacle_history = results['obstacle_history']

    plt.figure(figsize=(14, 10))
    plt.plot(BOUNDARY_POINTS[:, 0], BOUNDARY_POINTS[:, 1], 'k-', linewidth=2)
    plt.fill(BOUNDARY_POINTS[:, 0], BOUNDARY_POINTS[:, 1], 'lightgray', alpha=0.3)

    colors = ['blue', 'red', 'green', 'purple']
    for i in range(4):
        t = np.array(traj[i])
        if len(t) > 0:
            success_text = "Success" if results["success"][i] else f"Failed (dist: {results['final_distances'][i]:.2f})"
            plt.plot(t[:, 0], t[:, 1], color=colors[i], label=f'Agent {i} ({success_text})', alpha=0.8, linewidth=2)
            plt.scatter(t[0, 0], t[0, 1], marker='o', s=120, color=colors[i], edgecolor='white', linewidth=2)
            plt.scatter(t[-1, 0], t[-1, 1], marker='*', s=250, color=colors[i], edgecolor='white', linewidth=2)

    obstacle_colors = ['darkgreen', 'darkmagenta', 'darkorange', 'darkred']
    for obs_idx, obstacle in enumerate(obstacles):
        traj_points = np.array([p['pos'] for p in obstacle['predicted_traj']])
        plt.plot(traj_points[:, 0], traj_points[:, 1], '--', color=obstacle_colors[obs_idx % len(obstacle_colors)], linewidth=2, alpha=0.7)

        if len(obstacle_history[obs_idx]) > 1:
            hist_points = np.array(obstacle_history[obs_idx])
            plt.plot(hist_points[:, 0], hist_points[:, 1], '-', color=obstacle_colors[obs_idx % len(obstacle_colors)], linewidth=3, alpha=0.9)

        current_pos = obstacle['predicted_traj'][obstacle['current_step']]['pos']
        plt.scatter(current_pos[0], current_pos[1], marker='s', s=150, color=obstacle_colors[obs_idx % len(obstacle_colors)], edgecolor='white', linewidth=2)
        
        obstacle_circle = Circle(current_pos, OBSTACLE_SIZE, color=obstacle_colors[obs_idx % len(obstacle_colors)], alpha=0.3, linewidth=0)
        plt.gca().add_patch(obstacle_circle)

    for i in range(4):
        plt.scatter(*goals[i, :2], marker='*', s=300, color=colors[i], edgecolor='black', linewidth=2, alpha=0.8)

    plt.gca().set_aspect('equal')
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.title(f"Test Results: {config_name}\nSteps: {results['steps_taken']}, Success Rate: {sum(results['success'])}/4")
    
    plt.tight_layout()
    plt.savefig(f"test_results_{config_name}.png", dpi=300, bbox_inches='tight')

def run_progressive_tests():
    agents_and_normalizer = load_trained_agents()
    if agents_and_normalizer[0] is None:
        return None

    trained_agents, state_normalizer = agents_and_normalizer
    
    test_configs = [
        {'name': 'training_identical', 'goals': TRAINING_GOALS, 'starts': TRAINING_STARTS, 'obstacles': 'training_identical'},
        {'name': 'new_goals', 'goals': TEST_GOALS, 'starts': TRAINING_STARTS, 'obstacles': 'training_identical'},
        {'name': 'obstacle_change', 'goals': TRAINING_GOALS, 'starts': TRAINING_STARTS, 'obstacles': 'slight_change'},
        {'name': 'moderate_change', 'goals': TEST_GOALS, 'starts': TEST_STARTS, 'obstacles': 'moderate_change'},
        {'name': 'crossing_obstacles', 'goals': TEST_GOALS, 'starts': TEST_STARTS, 'obstacles': 'crossing'}
    ]

    all_results = []
    print(f"{'Configuration':<25} {'Success Rate':<12} {'Avg Distance':<12} {'Steps':<8}")
    print("-" * 65)

    for config in test_configs:
        obstacles = create_test_obstacles(config['obstacles'])
        results = test_with_config(trained_agents, config['goals'], config['starts'], obstacles, config['name'])
        all_results.append(results)
        visualize_test_results(results, config['goals'])
        
        success_rate = sum(results['success']) / 4 * 100
        avg_distance = np.mean(results['final_distances'])
        print(f"{config['name']:<25} {success_rate:>8.1f}%     {avg_distance:>8.2f}      {results['steps_taken']:>5d}")

    return all_results

if __name__ == "__main__":
    run_progressive_tests()