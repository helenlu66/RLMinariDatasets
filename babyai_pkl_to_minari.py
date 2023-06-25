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


def deserialize_state(serialized_state):
    byte_stream = io.BytesIO(serialized_state)
    state = pickle.load(byte_stream, protocol=pickle.HIGHEST_PROTOCOL)
    return state


def convert_file():
    # Define the file path of the .pkl file
    file_path = "BabyAI-GoToLocal-v0.pkl"

    # Open the .pkl file in read binary mode
    
    with open(file_path, "rb") as file:
        # Load the data from the .pkl file
        expert_trajectories = pickle.load(file)
    
    # Access the expert trajectories
    # Assuming 'expert_trajectories' is the list of expert trajectories

    for trajectory in expert_trajectories[:1]:
        task_description = trajectory[0]
        serialized_state = trajectory[1]
        rewards = trajectory[2]
        actions = trajectory[3]

        # Deserialize the state
        #state = deserialize_state(serialized_state)

        # Process and analyze the trajectory data
        print("Task Description:", task_description)
        #print("State:", state)
        print("Rewards:", rewards)
        print("Actions:", actions)
        print("----")



if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dir', type=str, default='./old_datasets')
#     parser.add_argument("--author", type=str, help="name of the author of the dataset", default=None)
#     parser.add_argument("--author_email", type=str, help="email of the author of the dataset", default=None)
#     parser.add_argument("--code_permalink", type=str, help="link to the code used to generate the dataset", default=None)
#     args = parser.parse_args()
#     convert_files(args)
    convert_file()