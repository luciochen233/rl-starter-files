import gymnasium as gym


def make_env(env_key, seed=None, render_mode=None,agent_view_size_param=7):
    #env = gym.make(env_key, render_mode=render_mode)
    env = gym.make(env_key, render_mode=render_mode, agent_view_size=agent_view_size_param)
    env.reset(seed=seed)
    return env


class ILDatasetWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, il_view_size=7):
        super().__init__(env)

        assert il_view_size % 2 == 1
        assert il_view_size >= 3

        self.il_view_size = il_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(il_view_size, il_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = gym.spaces.Dict(
            {**self.observation_space.spaces, "il_image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.il_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        return {**obs, "il_image": image}
