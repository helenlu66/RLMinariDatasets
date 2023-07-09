import pickle
#import gymnasium as gym
#import minari
import numpy as np

import pickle
import collections
import argparse
import os
from PIL import Image
import io
import gzip
import gym
import h5py
from minari import MinariDataset


def deserialize_state(serialized_state):
    #byte_stream = io.BytesIO(serialized_state)
    state = pickle.loads(serialized_state, fix_imports=True, encoding='bytes')
    return state


def explore_dataset(filename):

    # Open the .pkl file in read binary mode
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), "rb") as file:
        # Load the data from the .pkl file
        expert_trajectories = pickle.load(file)

    for trajectory in expert_trajectories[:5]:
        task_description = trajectory[0]
        state = trajectory[1]
        rewards = trajectory[2]
        actions = trajectory[3]

        # Process and analyze the trajectory data
        print("Task Description:", task_description)
        print("State:", deserialize_state(state))
        print("Rewards:", rewards)
        print("Actions:", actions)
        print("----")
    

def port_dataset_to_minari(filename):
    # Open the .pkl file in read binary mode
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), filename), "rb") as file:
        # Load the data from the .pkl file
        expert_trajectories = pickle.load(file)
    

    buffer = []

    for trajectory in expert_trajectories:
        episode_dict = collections.defaultdict(list)
        task_desc = trajectory[0]
        serialized_state = trajectory[1]
        episode_dict['observations']
        rewards = trajectory[2]
        actions = trajectory[3]

        episode = {
                "observations": np.array([serialized_state] + [None] * len(actions)),  # Replace None with the appropriate initial observation
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "terminations": np.zeros(len(actions)),  # Replace zeros with appropriate termination values if available
                "truncations": np.zeros(len(actions))  # Replace zeros with appropriate truncation values if available
            }
        buffer.append(episode)

    env = gym.make(filename)
    # Specify the output HDF5 file path
    output_file = f'{filename}.h5'
    hf = h5py.File(output_file, 'w')
    # Create an HDF5 file and save the dataset
    # Create HDF5 datasets for each component of the Minari dataset
    hf.create_dataset('observations', data=buffer)
    hf.close()
    # Confirm successful saving
    print('Minari dataset saved as HDF5 successfully.')


if __name__ == '__main__':
    filename = 'BabyAI-GoToLocal-v0.pkl'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./old_datasets')
    parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
    parser.add_argument("--author_email", type=str, help="email of the author of the dataset", default=None)
    parser.add_argument("--code_permalink", type=str, help="link to the code used to generate the dataset", default=None)
    args = parser.parse_args()

    explore_dataset(filename)
    port_dataset_to_minari(filename)

