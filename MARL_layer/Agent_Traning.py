import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import pickle
from collections import deque
from matplotlib.path import Path

WORLD_SIZE = 12
GOALS = np.array([[0, 7.5], [7.5, 0], [-5, 0], [0, -5]])
STARTS = np.array([[-5, -5, 0], [-7.5, 2.5, 0], [-5, -10, np.pi / 2], [5, 0, np.pi]])

NUM_DYNAMIC_OBSTACLES = 2
OBSTACLE_SIZE = 0.8
PREDICTED_TRAJ_LENGTH = 20
UNCERTAINTY_RADIUS_INIT = 0.5
UNCERTAINTY_GROWTH = 0.1

AGENT_STATE_DIM = 4
OBSTACLE_INFO_DIM = 6
STATE_DIM = AGENT_STATE_DIM + NUM_DYNAMIC_OBSTACLES * OBSTACLE_INFO_DIM

ACTION_DIM = 2
ACTOR_LR = 2e-4
CRITIC_LR = 1.5e-4
GAMMA = 0.99
TAU = 0.001
BUFFER_SIZE = 100000
BATCH_SIZE = 128
NOISE_SCALE_INIT = 0.3
NOISE_DECAY = 0.9995
MIN_NOISE = 0.01
EPISODES = 1000
STEPS = 300
POLICY_DELAY = 2
TARGET_NOISE = 0.1
NOISE_CLIP = 0.3

DT = 0.1
L = 0.5

def generate_boundary():
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = 10 + 2 * np.sin(5 * theta) + np.cos(3 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

BOUNDARY_POINTS = generate_boundary()
BOUNDARY_PATH = Path(BOUNDARY_POINTS)

class StateNormalizer:
    def __init__(self):
        self.agent_means = np.array([3.0, 0.0, 1.0, 0.0])
        self.agent_stds = np.array([6.0, np.pi, 2.0, np.pi])
        self.obstacle_means = np.array([0.0, 0.0, 0.0, 0.0, 8.0, 0.5])
        self.obstacle_stds = np.array([12.0, 12.0, 12.0, 12.0, 12.0, 1.0])
        self.distance_history = []
        self.agent_stats = []
        self.obstacle_stats = []
        self.update_count = 0

    def normalize_state(self, state):
        normalized = np.zeros_like(state)
        distance = state[0]

        if distance > 0.3:
            normalized_distance = np.log(distance + 1) / np.log(10)
        else:
            normalized_distance = distance / 0.1

        normalized[0] = np.clip(normalized_distance, -2, 2)

        agent_part = state[1:4]
        agent_means = self.agent_means[1:4]
        agent_stds = self.agent_stds[1:4]
        normalized[1:4] = (agent_part - agent_means) / agent_stds

        for i in range(NUM_DYNAMIC_OBSTACLES):
            start_idx = 4 + i * 6
            end_idx = start_idx + 6
            obstacle_part = state[start_idx:end_idx]
            normalized[start_idx:end_idx] = (obstacle_part - self.obstacle_means) / self.obstacle_stds

        return np.clip(normalized, -3, 3)

    def update_stats(self, states):
        if self.update_count % 1000 == 0 and len(states) > 0:
            distances = states[:, 0]
            self.distance_history.extend(distances)

            agent_parts = states[:, 1:4]
            self.agent_means[1:4] = 0.9 * self.agent_means[1:4] + 0.1 * np.mean(agent_parts, axis=0)
            self.agent_stds[1:4] = 0.9 * self.agent_stds[1:4] + 0.1 * np.std(agent_parts, axis=0)

            for i in range(NUM_DYNAMIC_OBSTACLES):
                start_idx = 4 + i * 6
                end_idx = start_idx + 6
                obstacle_parts = states[:, start_idx:end_idx]
                self.obstacle_means = 0.9 * self.obstacle_means + 0.1 * np.mean(obstacle_parts, axis=0)
                self.obstacle_stds = 0.9 * self.obstacle_stds + 0.1 * np.std(obstacle_parts, axis=0)

            self.agent_stds = np.maximum(self.agent_stds, 0.1)
            self.obstacle_stds = np.maximum(self.obstacle_stds, 0.1)

        self.update_count += 1

    def save_parameters(self, filepath):
        params = {
            'agent_means': self.agent_means,
            'agent_stds': self.agent_stds,
            'obstacle_means': self.obstacle_means,
            'obstacle_stds': self.obstacle_stds,
            'update_count': self.update_count,
            'distance_history': self.distance_history[-10000:]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)

    def load_parameters(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
            self.agent_means = params['agent_means']
            self.agent_stds = params['agent_stds']
            self.obstacle_means = params['obstacle_means']
            self.obstacle_stds = params['obstacle_stds']
            self.update_count = params.get('update_count', 0)
            self.distance_history = params.get('distance_history', [])
            return True
        except FileNotFoundError:
            return False

state_normalizer = StateNormalizer()

def init_dynamic_obstacles():
    obstacles = []
    start_pos1 = np.array([0, -5])
    end_pos1 = np.array([0, 5])
    traj_points1 = []
    for t in range(PREDICTED_TRAJ_LENGTH):
        ratio = t / (PREDICTED_TRAJ_LENGTH - 1)
        pos = start_pos1 * (1 - ratio) + end_pos1 * ratio
        radius = UNCERTAINTY_RADIUS_INIT + t * UNCERTAINTY_GROWTH
        traj_points1.append({'pos': pos.copy(), 'radius': radius})

    obstacles.append({
        'predicted_traj': traj_points1,
        'current_step': 0,
        'speed': 0.8
    })

    start_pos2 = np.array([-5, 0])
    end_pos2 = np.array([5, -5])
    traj_points2 = []
    for t in range(PREDICTED_TRAJ_LENGTH):
        ratio = t / (PREDICTED_TRAJ_LENGTH - 1)
        pos = start_pos2 * (1 - ratio) + end_pos2 * ratio
        radius = UNCERTAINTY_RADIUS_INIT + t * UNCERTAINTY_GROWTH
        traj_points2.append({'pos': pos.copy(), 'radius': radius})

    obstacles.append({
        'predicted_traj': traj_points2,
        'current_step': 0,
        'speed': 0.8
    })
    return obstacles

def update_dynamic_obstacles(obstacles):
    for obs in obstacles:
        obs['current_step'] = min(obs['current_step'] + 1, PREDICTED_TRAJ_LENGTH - 1)
    return obstacles

def build_actor():
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

def build_shared_critic():
    state_input = layers.Input(shape=(STATE_DIM,))
    action_input = layers.Input(shape=(ACTION_DIM,))

    state_branch = layers.Dense(256, activation='relu')(state_input)
    state_branch = layers.BatchNormalization()(state_branch)
    state_branch = layers.Dropout(0.1)(state_branch)
    state_branch = layers.Dense(256, activation='relu')(state_branch)
    state_branch = layers.BatchNormalization()(state_branch)

    action_branch = layers.Dense(64, activation='relu')(action_input)

    x = layers.Concatenate()([state_branch, action_branch])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1)(x)

    return tf.keras.Model([state_input, action_input], out)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), BATCH_SIZE, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(map(np.array, zip(*samples)))
        return batch[0], batch[1], batch[2], batch[3], indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

class MATD3System:
    def __init__(self, num_agents=4):
        self.num_agents = num_agents
        self.actors = [build_actor() for _ in range(num_agents)]
        self.target_actors = [build_actor() for _ in range(num_agents)]

        self.shared_critic1 = build_shared_critic()
        self.shared_critic2 = build_shared_critic()
        self.target_shared_critic1 = build_shared_critic()
        self.target_shared_critic2 = build_shared_critic()

        for i in range(num_agents):
            self.target_actors[i].set_weights(self.actors[i].get_weights())

        self.target_shared_critic1.set_weights(self.shared_critic1.get_weights())
        self.target_shared_critic2.set_weights(self.shared_critic2.get_weights())

        self.actor_optimizers = [tf.keras.optimizers.Adam(ACTOR_LR) for _ in range(num_agents)]
        self.critic1_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
        self.critic2_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)

        self.buffer = PrioritizedReplayBuffer()
        self.noise_scales = [NOISE_SCALE_INIT] * num_agents
        self.policy_update_counter = 0

        self.critic_losses = []
        self.actor_losses = []

    def update_targets(self):
        for i in range(self.num_agents):
            for t, s in zip(self.target_actors[i].variables, self.actors[i].variables):
                t.assign(TAU * s + (1 - TAU) * t)

        for t, s in zip(self.target_shared_critic1.variables, self.shared_critic1.variables):
            t.assign(TAU * s + (1 - TAU) * t)
        for t, s in zip(self.target_shared_critic2.variables, self.shared_critic2.variables):
            t.assign(TAU * s + (1 - TAU) * t)

    def get_actions(self, states, training=True):
        actions = []
        for i in range(self.num_agents):
            normalized_state = state_normalizer.normalize_state(states[i])
            action = self.actors[i](normalized_state[np.newaxis])[0]
            if training:
                action += self.noise_scales[i] * np.random.randn(ACTION_DIM)
            actions.append(np.clip(action, -1, 1))
        return np.array(actions)

    @tf.function
    def _train_step_internal(self, states, actions, rewards, next_states, weights):
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        td_errors = []

        for i in range(self.num_agents):
            next_action = self.target_actors[i](next_states[i:i + 1])
            noise = tf.random.normal(shape=tf.shape(next_action), stddev=TARGET_NOISE)
            noise = tf.clip_by_value(noise, -NOISE_CLIP, NOISE_CLIP)
            next_action = tf.clip_by_value(next_action + noise, -1, 1)

            target_Q1 = self.target_shared_critic1([next_states[i:i + 1], next_action])
            target_Q2 = self.target_shared_critic2([next_states[i:i + 1], next_action])
            target_Q = tf.minimum(target_Q1, target_Q2)
            target = rewards[i:i + 1, tf.newaxis] + GAMMA * target_Q

            with tf.GradientTape() as tape1:
                Q1 = self.shared_critic1([states[i:i + 1], actions[i:i + 1]])
                td_error = target - Q1
                critic1_loss = tf.reduce_mean(weights[i:i + 1, tf.newaxis] * tf.square(td_error))

            critic1_grads = tape1.gradient(critic1_loss, self.shared_critic1.trainable_variables)
            critic1_grads = [tf.clip_by_norm(grad, 1.0) for grad in critic1_grads]
            self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.shared_critic1.trainable_variables))

            with tf.GradientTape() as tape2:
                Q2 = self.shared_critic2([states[i:i + 1], actions[i:i + 1]])
                critic2_loss = tf.reduce_mean(weights[i:i + 1, tf.newaxis] * tf.square(target - Q2))

            critic2_grads = tape2.gradient(critic2_loss, self.shared_critic2.trainable_variables)
            critic2_grads = [tf.clip_by_norm(grad, 1.0) for grad in critic2_grads]
            self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.shared_critic2.trainable_variables))

            total_critic_loss += critic1_loss + critic2_loss
            td_errors.append(tf.abs(td_error))

        if tf.equal(tf.math.mod(self.policy_update_counter, POLICY_DELAY), 0):
            for i in range(self.num_agents):
                with tf.GradientTape() as tape:
                    actor_actions = self.actors[i](states[i:i + 1])
                    actor_loss = -tf.reduce_mean(self.shared_critic1([states[i:i + 1], actor_actions]))

                actor_grads = tape.gradient(actor_loss, self.actors[i].trainable_variables)
                actor_grads = [tf.clip_by_norm(grad, 1.0) for grad in actor_grads]
                self.actor_optimizers[i].apply_gradients(zip(actor_grads, self.actors[i].trainable_variables))
                total_actor_loss += actor_loss

            self.update_targets()

        return total_critic_loss / self.num_agents, total_actor_loss / self.num_agents, td_errors

    def train(self):
        if len(self.buffer.buffer) < BATCH_SIZE:
            return 0, 0

        states, actions, rewards, next_states, indices, weights = self.buffer.sample()

        norm_states = np.array([state_normalizer.normalize_state(s) for s in states])
        norm_next_states = np.array([state_normalizer.normalize_state(s) for s in next_states])

        state_normalizer.update_stats(states)

        states_tf = tf.convert_to_tensor(norm_states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(norm_next_states, dtype=tf.float32)
        weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)

        critic_loss, actor_loss, td_errors = self._train_step_internal(
            states_tf, actions_tf, rewards_tf, next_states_tf, weights_tf)

        new_priorities = np.mean([td_error.numpy().flatten() for td_error in td_errors], axis=0) + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        self.policy_update_counter += 1

        self.critic_losses.append(critic_loss.numpy())
        self.actor_losses.append(actor_loss.numpy())

        return critic_loss.numpy(), actor_loss.numpy()

def bicycle_model_vec(pos, vel, theta, actions):
    a, delta = actions[:, 0], actions[:, 1]
    vel = np.clip(vel + a * DT, -1, 3)
    delta = np.clip(delta, -0.5, 0.5)
    theta += vel / L * np.tan(delta) * DT
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    pos[:, 0] += vel * np.cos(theta) * DT
    pos[:, 1] += vel * np.sin(theta) * DT
    return pos, vel, theta

def get_enhanced_state_vec(pos, vel, theta, goals, dynamic_obstacles):
    n_agents = pos.shape[0]
    enhanced_states = []

    for i in range(n_agents):
        dx = goals[i, 0] - pos[i, 0]
        dy = goals[i, 1] - pos[i, 1]
        distance = np.hypot(dx, dy)
        angle_to_goal = np.arctan2(dy, dx)
        dtheta_to_goal = (angle_to_goal - theta[i] + np.pi) % (2 * np.pi) - np.pi

        agent_state = [distance, dtheta_to_goal, vel[i], theta[i]]

        obstacle_info = []
        for obs in dynamic_obstacles:
            current_pos = obs['predicted_traj'][obs['current_step']]['pos']
            next_step = min(obs['current_step'] + 1, len(obs['predicted_traj']) - 1)
            next_pos = obs['predicted_traj'][next_step]['pos']

            rel_x = current_pos[0] - pos[i, 0]
            rel_y = current_pos[1] - pos[i, 1]
            next_rel_x = next_pos[0] - pos[i, 0]
            next_rel_y = next_pos[1] - pos[i, 1]
            dist_to_obstacle = np.linalg.norm([rel_x, rel_y])
            uncertainty = obs['predicted_traj'][obs['current_step']]['radius']

            obstacle_info.extend([rel_x, rel_y, next_rel_x, next_rel_y, dist_to_obstacle, uncertainty])

        full_state = agent_state + obstacle_info
        enhanced_states.append(full_state)

    return np.array(enhanced_states)

def calculate_uncertainty_penalty(agent_pos, obstacle):
    penalty = 0
    for i, point in enumerate(obstacle['predicted_traj']):
        if i >= obstacle['current_step']:
            dist = np.linalg.norm(agent_pos - point['pos'])
            time_weight = (i - obstacle['current_step'] + 1) / PREDICTED_TRAJ_LENGTH
            penalty += time_weight * np.exp(-0.5 * (dist / point['radius']) ** 2)
    return penalty

def get_reward_vec(pos, vel, theta, actions, goals, dynamic_obstacles):
    n_agents = pos.shape[0]
    in_boundary = BOUNDARY_PATH.contains_points(pos[:, :2])
    dist_to_goal = np.linalg.norm(pos[:, :2] - goals[:, :2], axis=1)

    rewards = -dist_to_goal * 0.2

    acceleration = actions[:, 0]
    steering_rate = actions[:, 1]
    rewards -= 0.05 * np.square(acceleration)
    rewards -= 0.1 * np.square(steering_rate)

    goal_direction = np.arctan2(goals[:, 1] - pos[:, 1], goals[:, 0] - pos[:, 0])
    heading_diff = (goal_direction - theta + np.pi) % (2 * np.pi) - np.pi
    rewards += 0.3 * np.cos(heading_diff)

    close_to_goal = dist_to_goal < 2
    rewards[close_to_goal] -= 0.3 * np.sin(heading_diff)[close_to_goal]

    goal_reached = (dist_to_goal < 0.3)
    rewards[goal_reached] += 100

    rewards[~in_boundary] -= 30

    for i in range(n_agents):
        for obs in dynamic_obstacles:
            uncertainty_penalty = calculate_uncertainty_penalty(pos[i, :2], obs)
            rewards[i] -= 2 * uncertainty_penalty

            obs_pos = obs['predicted_traj'][obs['current_step']]['pos']
            obs_dist = np.linalg.norm(pos[i, :2] - obs_pos)
            if obs_dist < OBSTACLE_SIZE:
                rewards[i] -= 20
            elif obs_dist < 2 * OBSTACLE_SIZE:
                rewards[i] -= 2 * (2 * OBSTACLE_SIZE - obs_dist)

    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                dist = np.linalg.norm(pos[i, :2] - pos[j, :2])
                if dist < 0.5:
                    rewards[i] -= 10 * (0.5 - dist)

    return np.clip(rewards, -30, 150)

def train():
    global state_normalizer
    matd3_system = MATD3System(num_agents=4)
    rewards_history = []
    critic_loss_history = []
    actor_loss_history = []
    start_time = time.time()
    dynamic_obstacles = init_dynamic_obstacles()

    for ep in range(EPISODES):
        pos = STARTS.copy()
        vel = np.zeros(4)
        traj = [deque(maxlen=STEPS) for _ in range(4)]
        ep_rewards = np.zeros(4)
        frozen = [False, False, False, False]

        dynamic_obstacles = init_dynamic_obstacles()

        for step in range(STEPS):
            dynamic_obstacles = update_dynamic_obstacles(dynamic_obstacles)

            states = get_enhanced_state_vec(pos, vel, pos[:, 2], GOALS, dynamic_obstacles)
            actions = np.zeros((4, 2))

            for i in range(4):
                if not frozen[i]:
                    normalized_state = state_normalizer.normalize_state(states[i])
                    actions[i] = matd3_system.actors[i](normalized_state[np.newaxis])[0]
                    if ep < EPISODES * 0.8:
                        actions[i] += matd3_system.noise_scales[i] * np.random.randn(ACTION_DIM)
                    actions[i] = np.clip(actions[i], -1, 1)

            new_pos, new_vel, new_theta = bicycle_model_vec(pos.copy(), vel.copy(), pos[:, 2].copy(), actions)
            new_pos[:, 2] = new_theta

            in_boundary = BOUNDARY_PATH.contains_points(new_pos[:, :2])
            for i in range(4):
                if not in_boundary[i]:
                    new_pos[i] = pos[i]
                    new_vel[i] = 0

            rewards = get_reward_vec(new_pos, new_vel, new_pos[:, 2], actions, GOALS, dynamic_obstacles)
            new_states = get_enhanced_state_vec(new_pos, new_vel, new_pos[:, 2], GOALS, dynamic_obstacles)

            for i in range(4):
                if not frozen[i]:
                    goal_dist = np.linalg.norm(new_pos[i, :2] - GOALS[i, :2])
                    if goal_dist < 0.3:
                        frozen[i] = True
                        rewards[i] += 50
                        new_vel[i] = 0
                        actions[i] = np.zeros(2)

                    matd3_system.buffer.add((states[i], actions[i], rewards[i], new_states[i]))
                    traj[i].append(new_pos[i, :2].copy())
                    ep_rewards[i] += rewards[i]
                else:
                    traj[i].append(new_pos[i, :2].copy())
                    new_vel[i] = 0

            if step % 4 == 0:
                critic_loss, actor_loss = matd3_system.train()
                if critic_loss > 0:
                    critic_loss_history.append(critic_loss)
                    actor_loss_history.append(actor_loss)

            pos, vel = new_pos, new_vel

            if all(frozen):
                break

        for i in range(4):
            matd3_system.noise_scales[i] = max(MIN_NOISE, matd3_system.noise_scales[i] * NOISE_DECAY)

        avg_reward = np.mean(ep_rewards)
        rewards_history.append(avg_reward)

        if ep % 20 == 0:
            recent_critic_loss = np.mean(critic_loss_history[-100:]) if critic_loss_history else 0
            recent_actor_loss = np.mean(actor_loss_history[-100:]) if actor_loss_history else 0
            success_rate = sum(frozen) / 4

            print(f"Episode {ep:3d} | Avg Reward: {avg_reward:7.1f} | "
                  f"Success: {success_rate:.2f} | "
                  f"Critic Loss: {recent_critic_loss:.4f} | "
                  f"Actor Loss: {recent_actor_loss:.4f} | "
                  f"Noise: {matd3_system.noise_scales[0]:.3f} | "
                  f"Time: {time.time() - start_time:.1f}s")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    ax1.plot(rewards_history)
    ax1.set_title("Training Rewards")
    
    if critic_loss_history:
        ax2.plot(critic_loss_history)
        ax2.set_title("Critic Loss")
        
    if actor_loss_history:
        ax3.plot(actor_loss_history)
        ax3.set_title("Actor Loss")

    recent_rewards = rewards_history[-50:] if len(rewards_history) >= 50 else rewards_history
    ax4.hist(recent_rewards, bins=20, alpha=0.7)
    ax4.set_title("Recent Rewards Distribution")

    plt.tight_layout()
    plt.savefig('training_results.png')

    for i in range(4):
        matd3_system.actors[i].save_weights(f'improved_shared_critic_actor_{i}_weights.h5')

    matd3_system.shared_critic1.save_weights('improved_shared_critic1_weights.h5')
    matd3_system.shared_critic2.save_weights('improved_shared_critic2_weights.h5')

    state_normalizer.save_parameters('state_normalizer_params.pkl')

    return matd3_system, [list(t) for t in traj], dynamic_obstacles

if __name__ == "__main__":
    train()
