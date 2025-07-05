import base64
import random
from itertools import zip_longest
from collections import deque
import imageio
import IPython
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.iolib.table import SimpleTable


SEED = 0              # seed for pseudo-random number generator
MINIBATCH_SIZE = 64   # mini-batch size
TAU = 0.005           # soft update parameter
E_DECAY = 0.9995      # ε decay rate for ε-greedy policy
E_MIN = 0.1          # minimum ε value for ε-greedy policy


random.seed(SEED)


def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.int32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8),
                                     dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)


def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > 10000:
        return True
    else:
        return False
    
    
def get_new_eps(epsilon):
    return max(E_MIN, E_DECAY*epsilon)


def get_action(q_values, epsilon, num_actions):
    if random.random() > epsilon:
        return np.argmax(q_values[0])
    else:
        return random.choice(np.arange(num_actions))
    
    
def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
    

def plot_history(reward_history, rolling_window=20, lower_limit=None,
                 upper_limit=None, plot_rw=True, plot_rm=True):
    
    if lower_limit is None or upper_limit is None:
        rh = reward_history
        xs = [x for x in range(len(reward_history))]
    else:
        rh = reward_history[lower_limit:upper_limit]
        xs = [x for x in range(lower_limit,upper_limit)]
    
    df = pd.DataFrame(rh)
    rollingMean = df.rolling(rolling_window).mean()

    plt.figure(figsize=(10,7), facecolor='white')
    
    if plot_rw:
        plt.plot(xs, rh, linewidth=1, color='cyan')
    if plot_rm:
        plt.plot(xs, rollingMean, linewidth=2, color='magenta')

    text_color = 'black'
        
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.grid()
#     plt.title("Total Point History", color=text_color, fontsize=40)
    plt.xlabel('Episode', color=text_color, fontsize=30)
    plt.ylabel('Total Points', color=text_color, fontsize=30)
    yNumFmt = mticker.StrMethodFormatter('{x:,}')
    ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis='x', colors=text_color)
    ax.tick_params(axis='y', colors=text_color)
    plt.show()


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="840" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())
    return IPython.display.HTML(tag)
        
        
def create_video(filename, q_network, fps=30.0):
    import gym
    import numpy as np
    import imageio
    from gym.wrappers import AtariPreprocessing, FrameStack

    # Setup environment
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = FrameStack(env, num_stack=4)

    with imageio.get_writer(filename, fps=float(fps)) as video:
        state = env.reset()
        state = np.array(state)
        if state.shape == (4, 84, 84):
            state = np.transpose(state, (1, 2, 0))
        frame = env.render(mode='rgb_array')
        video.append_data(frame)

        # Force FIRE action once
        state, _, done, _ = env.step(1)
        state = np.array(state)
        if state.shape == (4, 84, 84):
            state = np.transpose(state, (1, 2, 0))
        frame = env.render(mode='rgb_array')
        video.append_data(frame)

        done = False
        step = 0
        MAX_STEPS = 2000
        lives = env.unwrapped.ale.lives()

        while not done and step < MAX_STEPS:
            # Prepare input for model
            state_input = np.expand_dims(state, axis=0)

            # Get action from Q-network
            q_values = q_network(state_input)
            action = np.argmax(q_values.numpy()[0])

            # Step environment
            state, reward, done, _ = env.step(action)
            state = np.array(state)
            if state.shape == (4, 84, 84):
                state = np.transpose(state, (1, 2, 0))
            frame = env.render(mode='rgb_array')
            video.append_data(frame)

            # Check for life loss
            new_lives = env.unwrapped.ale.lives()
            if new_lives < lives:
                print(f"\n⚠️ Life lost! Forcing FIRE.")
                lives = new_lives
                state, _, done, _ = env.step(1)
                state = np.array(state)
                if state.shape == (4, 84, 84):
                    state = np.transpose(state, (1, 2, 0))
                frame = env.render(mode='rgb_array')
                video.append_data(frame)

            print(f"Step: {step:4d} | Action: {action} | Reward: {reward} | Done: {done}", end="\r")
            step += 1

    env.close()
    print("\n✅ Video generation complete.")
