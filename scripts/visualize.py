import os
import argparse
import numpy
import torch

import utils
from utils import device
from PIL import Image


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=1000000,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")

parser.add_argument("--visibility", type=int, default=7,
                    help="Number of visibility (default: 7)")
parser.add_argument("--il_visibility", type=int, default=7,
                    help="Number of visibility for il agent (default: 7)")
parser.add_argument("--save", action="store_true", default=False,
                    help="episodes (il state and expert action sequences) will be saved in model dir")
parser.add_argument("--nowin", action="store_true", default=False,
                    help="no window")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

render_mode = "None"
if not args.save:
    if args.nowin:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
env = utils.make_env(args.env, args.seed, render_mode=render_mode, agent_view_size_param = args.visibility)

if args.save:
    env = utils.ILDatasetWrapper(env, il_view_size=args.il_visibility)
    trajectories = {'states': [], 'actions': []}
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

# Run the agent

frames = []

# Create a window to view the environment
if not args.save:
    print("no save")
    env.render()

for episode in range(args.episodes):
    obs, _ = env.reset()
    if args.save:
        states, actions = [], []
    if episode % 100 == 0: 
        print(episode,end="\r")

    # if episode % 5000 == 4999:
    #     print("saving progress\n\n")
    #     torch.save(
    #     trajectories,
    #     os.path.join(model_dir, f"expert_vis{args.visibility}_il_vis{args.visibility}_first{episode+1}.pt")
    #     )
    #     trajectories = {'states': [], 'actions': []}
    while True:
        if not args.save:
            env.render()
        if args.gif and episode % 50 == 49:
            frames.append(env.get_frame())

        action = agent.get_action(obs)
        if args.save:
            states.append(obs['il_image'])
            actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        agent.analyze_feedback(reward, done)

        if done and args.save:
            trajectories['states'].append(states)
            trajectories['actions'].append(actions)

        if done or (env.window and env.window.closed):
            break


    if env.window and env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    #write_gif(numpy.array(frames), f"{args.gif}{episode}.gif", fps=1/args.pause)
    imgs = [Image.fromarray(frame,mode='RGB') for frame in frames]
    imgs[0].save(f"{args.gif}{episode}.gif",save_all=True,append_images=imgs[1:],duration=args.pause,loop=0)
    print("Done.")

if args.save:
    torch.save(
        trajectories,
        os.path.join(model_dir, f"expert_vis{args.visibility}_il_vis{args.visibility}.pt")
    )
