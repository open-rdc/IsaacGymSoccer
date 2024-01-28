from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch

"""
   Cartpole environment built on top of Isaac Gym.
   Based on the official implementation here: 
   https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/cartpole.py
"""

class Soccer:
    def __init__(self, args):
        self.args = args

        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 30.
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        # task-specific parameters
        self.num_obs = 13 # self pos 3 + ball 2 + robot 4 * 2
        self.num_act = 1 #
        self.actions = torch.tensor([[1,0,0,0,0], [-1,0,0,0,0], [0,1,0,0,0], [0,-1,0,0,0], [0,0,1,0,0], [0,0,-1,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0]], device=self.args.sim_device)
        #self.actions = torch.tensor([[0.3,0,0,0,0], [0.3,0,0,0,0], [0,0.2,0,0,0], [0,-0.2,0,0,0], [0,0,0.5,0,0], [0,0,-0.5,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0]], device=self.args.sim_device)
        #foward, backword, left, right, cw, ccw, left kick, right kick, stop

        self.reset_dist = 3.0  # when to reset
        self.max_episode_length = 500  # maximum episode length

        # allocate buffers
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialise envs and state tensors
        self.envs, self.num_dof = self.create_envs()

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_states = dof_states.view(self.args.num_envs, self.num_dof * 2)
        self.dof_pos = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 1]

        # generate viewer for visualisation
        if not self.args.headless:
            self.viewer = self.create_viewer()

        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        self.reset()

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(12, 9, 0)
        num_per_row = int(np.sqrt(self.args.num_envs))

        self.actors_per_env = 2
        self.all_soccer_indices = self.actors_per_env * torch.arange(self.args.num_envs, dtype=torch.int32, device=self.args.sim_device)
        
        # create soccer asset
        asset_root = 'assets'
        asset_file = 'soccer.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        soccer_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(soccer_asset)

        # create ball asset
        ball_asset_file = "ball.urdf"
        ball_options = gymapi.AssetOptions()
        ball_options.angular_damping = 0.77
        ball_options.linear_damping = 0.77
        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, ball_options)
        ball_init_pose = gymapi.Transform()
        ball_init_pose.p = gymapi.Vec3(0, 0, 0.08)

        # define cartpole pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0.01)

        # define soccer dof properties
        dof_props = self.gym.get_asset_dof_properties(soccer_asset)
        dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
        dof_props['stiffness'][:] = 10000.0
        dof_props['damping'][:] = 500.0

        # generate environments
        envs = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add cartpole here in each environment
            soccer_handle = self.gym.create_actor(env, soccer_asset, pose, "soccer", i, 0, 0)
            self.gym.set_actor_dof_properties(env, soccer_handle, dof_props)
            
            ball_handle = self.gym.create_actor(env, ball_asset, ball_init_pose, "ball", i, 1, 0)

            envs.append(env)
        return envs, num_dof

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer

    def get_obs(self, env_ids=None):
        pass
        # get state observation from each environment id
        #if env_ids is None:
        #    env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)

        self.gym.refresh_dof_state_tensor(self.sim)
        #self.obs_buf[env_ids] = self.dof_states[env_ids]

    def get_reward(self):
        self.reward_buf[:], self.reset_buf[:] = compute_reward(self.dof_states,
                                                                        self.reset_dist,
                                                                        self.reset_buf,
                                                                        self.progress_buf,
                                                                        self.max_episode_length)

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities
        min_values = torch.tensor([-4, -2.5, 0, 0, 0, -4, -2.5, 0, 0, 0, 1, -2.5, 3.14, 0, 0, 1, -2.5, 3.14, 0, 0], device=self.args.sim_device)
        max_values = torch.tensor([-1, 2.5, 0, 0, 0, -1, 2.5, 0, 0, 0, 4, 2.5, 3.14, 0, 0, 4, 2.5, 3.14, 0, 0], device=self.args.sim_device)
        random_tensor = torch.rand((len(env_ids), self.num_dof), device=self.args.sim_device)
        positions = min_values + (max_values- min_values) * random_tensor

        self.dof_pos[env_ids, :] = positions[:]
        env_ids_int32 = self.all_soccer_indices[env_ids].flatten()

        # reset desired environments
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # refresh new observation after reset
        self.get_obs()

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def exit(self):
        # close the simulator in a graceful way
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def step(self, actions):
        # apply action
        each_dof_pos = self.dof_pos.view(self.args.num_envs * 4, 5)
        non_zero_rows = (each_dof_pos[:, 3] > 0.1) | (each_dof_pos[:, 4] > 0.1)
        actions[non_zero_rows] = 8
        actions_tensor = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        actions_tensor[:] = self.actions[actions].flatten()
        positions = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        positions0 = self.dof_pos[:].reshape(self.args.num_envs * 4, 5)
        positions0[non_zero_rows,3] = positions0[non_zero_rows,4] = 0
        positions[:] = positions0.reshape(-1)

        # simulate and render
        for i in range(10):
            positions += actions_tensor * 1 / 30
            target_pos = gymtorch.unwrap_tensor(positions)
            self.gym.set_dof_position_target_tensor(self.sim, target_pos)
            self.simulate()
            if not self.args.headless:
                self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        #self.get_reward()


# define reward function using JIT
#@torch.jit.script
def compute_reward(obs_buf, reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # retrieve each state from observation buffer
    cart_pos, cart_vel, pole_angle, pole_vel = torch.split(obs_buf, [1, 1, 1, 1], dim=1)

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reward[:, 0], reset[:, 0]
