# command to run this script
# python -m scripts.expert_knowledge --num_eps 1000 --dir FourRoomWithVisibility/FourRoomVisibility

import os
import sys

import argparse
import gymnasium as gym
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
import torch
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt


import utils

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument('--num_eps', type=int, required=True,
                help='number of trajectories to save')
parser.add_argument('--dir', type=str, required=True,
                help='path to save trajectories')
parser.add_argument('--render', action="store_true", default=False)

args = parser.parse_args()
args.agent_view_sizes = [7]

num_episodes = args.num_eps
data_dir = args.dir
render_image = args.render

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def pretty_print_graph(graph, agent_pos, goal_pos):
    for j in range(graph.shape[1]):
        row_str = ''
        for i in range(graph.shape[0]):
            if (i,j) == agent_pos:
                row_str += 'A'
            elif (i,j) == goal_pos:
                row_str += 'G'
            else:
                row_str += str(int(graph[i,j]))
        print(row_str)

def convert_to_graph(grid_state):
    width, height, _ = grid_state.shape
    graph = np.zeros((width, height))
    agent_pos = None
    goal_pos = None
    for i in range(width):
        for j in range(height):
            grid_val = grid_state[i, j, :]
            node_type = np.argmax(grid_val)
            if node_type == 2:
                # Goal
                goal_pos = (i, j)
                node_val = 1
            elif node_type == 3:
                # Agent start
                agent_pos = (i, j)
                node_val = 1
            elif node_type == 1:
                # Wall
                node_val = 0
            elif node_type == 0:
                # Empty
                node_val = 1
            else:
                print(grid_val)
                raise ValueError('Unrecognized grid val')
            graph[j, i] = node_val
    return graph, agent_pos, goal_pos


class MomentumAStarFinder(AStarFinder):
    def calc_cost(self, node_a, node_b):
        """
        get the distance between current node and the neighbor (cost)
        """
        if node_a.parent is not None:
            last_direction = (node_a.x - node_a.parent.x, node_a.y - node_a.parent.y)
        else:
            last_direction = None

        if last_direction:
            turned = last_direction != (node_b.x - node_a.x, node_b.y - node_a.y)
        else:
            turned = False

        if node_b.x - node_a.x == 0 or node_b.y - node_a.y == 0:
            # direct neighbor - distance is 1
            ng = 1
        else:
            # not a direct neighbor - diagonal movement
            ng = SQRT2

        ng += turned

        # weight for weighted algorithms
        if self.weighted:
            ng *= node_b.weight

        return node_a.g + ng


class GridWorldExpert:
    def _solve_env(self, state, direction):
        graph, agent_pos, goal_pos = convert_to_graph(state)

        grid = Grid(matrix=graph.tolist())
        start_node = grid.node(*agent_pos)
        end_node = grid.node(*goal_pos)
        finder = MomentumAStarFinder()
        path, _ = finder.find_path(start_node, end_node, grid)
        # For debugging print this out.
        # print(grid.grid_str(path=path, start=start_node, end=end_node))
        path_diffs = [(b[0]-a[0], b[1]-a[1]) for a,b in zip(path, path[1:])]

        # Found through brute force.
        diff_translate = {
                # left
                (-1,0): 2,
                # up
                (0,-1): 3,
                # right
                (1,0): 0,
                # down
                (0,1): 1,
                }

        path_diffs = [diff_translate[x] for x in path_diffs]
        return post_process(path_diffs, direction)

def post_process(path, direction):
    new_path = []
    for p in path:
        if direction == p:
            new_path.append(2)
        elif (direction + 1) % 4 == p:
            # turn right, move forward
            new_path.extend([1, 2])
        elif (direction + 3) % 4 == p:
            # turn left, move forward
            new_path.extend([0, 2])
        else:
            # turn back, move forward
            new_path.extend([1, 1, 2])
        direction = p
    return new_path


NODE_TO_ONE_HOT = {
    # Empty square
    (1, 0, 0): [1, 0, 0, 0],
    # Wall
    (2, 5, 0): [0, 1, 0, 0],
    # Goal
    (8, 1, 0): [0, 0, 1, 0],
    # Agent
    (10, 0, 0): [0, 0, 0, 1],
    (10, 0, 1): [0, 0, 0, 1],
    (10, 0, 2): [0, 0, 0, 1],
    (10, 0, 3): [0, 0, 0, 1],
}


class ExpertKnowledgeWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_sizes=[9, 11, 15]):
        super().__init__(env)

        assert all(agent_view_size % 2 == 1 for agent_view_size in agent_view_sizes)
        self.agent_view_sizes = agent_view_sizes

        # Compute observation space with specified view size
        obs_space_dict = {**self.observation_space.spaces}
        for agent_view_size in agent_view_sizes:
            new_image_space = gym.spaces.Box(
                low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
            )
            obs_space_dict['expert%d_image' % agent_view_size] = new_image_space

        # Override the environment's observation spaceexit
        self.observation_space = gym.spaces.Dict(obs_space_dict)

    def observation(self, obs):
        new_obs = {**obs}
        env = self.unwrapped

        for agent_view_size in self.agent_view_sizes:
            grid, vis_mask = env.gen_obs_grid(agent_view_size)

            # Encode the partially observable view into a numpy array
            image = grid.encode(vis_mask)
            new_obs['expert%d_image' % agent_view_size] = image

        return new_obs

    def reset(self, **kwargs):
        obs, _ = super().reset(**kwargs)
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        grid_shape = full_grid.shape
        full_grid = full_grid.reshape(-1, 3)
        full_grid = np.array(list(map(lambda x: NODE_TO_ONE_HOT[tuple(x)], full_grid)))
        grid_shape = grid_shape[:-1] + (4,) # last dim of NODE_TO_ONE_HOT
        full_grid = full_grid.reshape(grid_shape)

        obs['planner_image'] = full_grid
        obs['direction'] = self.agent_dir
        return obs, {}


def render(env, fn):
    img = Image.fromarray(env.render())
    img.save(os.path.join(data_dir, fn))

def reset_data():
    d = {}
    for size in args.agent_view_sizes:
        d['expert%d_image' % size] = []
    return d

def extract_data(env, obs, direction, actions, render_image=False):
    episode = reset_data()
    done = False
    i = 0
    if render_image:
        render(env, '%05d_initial.png' % i)

    while not done and i < len(actions):
        for size in args.agent_view_sizes:
            key = 'expert%d_image' % size
            episode[key].append(obs[key])
        next_obs, reward, done, truncated, info = env.step(actions[i])
        i += 1
        if render_image:
            render(env, '%05d_action_%d.png' % (i, actions[i-1]))
        obs = next_obs

    if i == len(actions):
        return episode, reward > 0 # info['ep_found_goal'] == 1
    else:
        return None, False

utils.seed(args.seed)
env_name = 'MiniGrid-FourRooms-v0'
env = ExpertKnowledgeWrapper(utils.make_env(env_name, render_mode='rgb_array'),
        agent_view_sizes=args.agent_view_sizes
)
print('Environment loaded')
expert = GridWorldExpert()
cur_count = 0
success_count = 0
data = reset_data()
trajectories = {'states': [], 'actions': []}
while cur_count < num_episodes:
    obs, _ = env.reset()
    planner_obs, direction = obs['planner_image'], obs['direction']
    actions = expert._solve_env(planner_obs, direction)
    episode, success = extract_data(env, obs, direction, actions)
    if episode is None:
        continue
    else:
        success_count += success
        cur_count += 1

        trajectories['states'].append(episode['expert%d_image' % args.agent_view_sizes[0]])
        trajectories['actions'].append(actions)
    if cur_count % 100 == 0:
        print(cur_count)

torch.save(
    trajectories,
    os.path.join(args.dir, f"expert_astar_il_vis%d.pt" % args.agent_view_sizes[0])
)
