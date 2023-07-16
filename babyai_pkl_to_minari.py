import pickle
#import gymnasium as gym
#import minari
import numpy as np

import pickle
import argparse
import os
import gymnasium as gym
import minari
import pickle5 as pickle
import blosc
from copy import deepcopy

def deserialize_state(serialized_state):
    # Unpickling
    decompressed_data = blosc.decompress(serialized_state)
    unpickled_obj = pickle.loads(decompressed_data)
    return unpickled_obj


def explore_dataset(filename):

    # Open the .pkl file in read binary mode
    pkl_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{filename}.pkl')
    with open(pkl_filepath, "rb") as file:
        # Load the data from the .pkl file
        expert_trajectories = pickle.load(file)

    for trajectory in expert_trajectories[:5]:
        task_description = trajectory[0]
        state = deserialize_state(trajectory[1])
        rewards = trajectory[2]
        actions = trajectory[3]

        # Process and analyze the trajectory data
        print("Task Description:", task_description)
        print("State:", state)
        print("Rewards:", rewards)
        print("Actions:", actions)
        print("----")
    

def load_dataset(filename):
    # Open the .pkl file in read binary mode
    pkl_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{filename}.pkl')
    with open(pkl_filepath, "rb") as file:
        # Load the data from the .pkl file
        expert_trajectories = pickle.load(file)
    return expert_trajectories
    
    # hdf5_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{filename}.hf')
def port_dataset_to_minari(expert_trajectories, dataset_id):
    buffer = []

    # 1000,000 total episodes in GoToLocal
    N = len(expert_trajectories)
    for trajectory in expert_trajectories[:N//100000]:
        task_desc = trajectory[0]
        deserialized_state = deserialize_state(trajectory[1])
        rewards = trajectory[2]
        actions = trajectory[3]

        terminations = np.zeros((len(actions), 1))
        terminations[-1, 0] = 1
        truncations = np.zeros((len(actions), 1))
        if rewards[-1] == 0:
            truncations[-1, 0] = 1
        
        initial_state = deepcopy(deserialized_state[0])
        initial_state = initial_state[np.newaxis, :]
        deserialized_state = np.concatenate((initial_state,deserialized_state), axis=0)

        episode = {
                "observations": np.array(deserialized_state),  # Replace None with the appropriate initial observation
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "terminations": terminations, 
                "truncations": truncations,
                #"task":np.array([task_desc])
            }
        buffer.append(episode)
    
    env = gym.make(dataset_id)
    os.environ['MINARI_DATASETS_PATH'] = args.dir
    minari.create_dataset_from_buffers(
        dataset_id = dataset_id,
        env=env,
        buffer=buffer
    )
    print('Minari dataset saved as HDF5 successfully.')


if __name__ == '__main__':
    filename = 'BabyAI-GoToLocal-v0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./datasets')
    parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
    parser.add_argument("--author_email", type=str, help="email of the author of the dataset", default=None)
    parser.add_argument("--code_permalink", type=str, help="link to the code used to generate the dataset", default=None)
    args = parser.parse_args()
 
    #explore_dataset(filename)
    expert_trajectories = load_dataset(filename)
    port_dataset_to_minari(expert_trajectories, filename)

