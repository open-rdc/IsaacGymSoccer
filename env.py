from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import math
from gymnasium.spaces import Box, Discrete, Sequence
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
        sim_params.substeps = 1
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        self.dt = sim_params.dt
        self.walking_period = 0.34

        # task-specific parameters
        self.num_obs = 11 # self pos 3 + ball 2 + robot 3 * 2
        self.num_act = 1 #
        self.actions = torch.tensor([[1.0, 0.0 ,0.0 ,0.0 ,0.0], [-1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0, 0.0]], device=self.args.sim_device)
        #self.actions = torch.tensor([[0.3,0,0,0,0], [0.3,0,0,0,0], [0,0.2,0,0,0], [0,-0.2,0,0,0], [0,0,0.5,0,0], [0,0,-0.5,0,0], [0,0,0,1,0], [0,0,0,0,1], [0,0,0,0,0]], device=self.args.sim_device)
        #foward, backword, left, right, cw, ccw, left kick, right kick, stop
        self.num_player = 4

        self.reset_dist = 3.0  # when to reset
        self.max_episode_length = self.args.episode_length  # maximum episode length

        # allocate buffers
        self.obs_buf = torch.zeros((self.args.num_envs*self.num_player, self.num_obs), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs*self.num_player, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        self.observation_space = [Box(low=-100, high=100, shape = ([11]), dtype=np.float16) for _ in range(int(self.args.num_envs*self.num_player/2))]
        self.share_observation_space = [Box(low=float("-inf"), high=float("inf"), shape = ([11]), dtype=np.float32) for _ in range(int(self.args.num_envs*self.num_player/2))]
        self.action_space = [Discrete(9) for _ in range(self.args.num_envs*self.num_player)]

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialise envs and state tensors
        self.envs, self.num_dof = self.create_envs()

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.args.num_envs, self.actors_per_env, 13)
        self.root_init_state = torch.tensor([[0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=self.args.sim_device)
        self.ball_pos = self.root_states[:, 1, 0:2]
        self.ball_vel = self.root_states[:, 1, 7:9]
        dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_states = dof_states.view(self.args.num_envs, self.num_dof * 2)
        self.dof_pos = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 1]

        # generate viewer for visualisation
        if not self.args.headless:
            self.viewer = self.create_viewer()

        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        torch_zeros = lambda : torch.zeros(self.args.num_envs * int(self.num_player/2), dtype=torch.float, device=self.args.sim_device, requires_grad=False)
        self.episode_sums = {"goal": torch_zeros(), "ball_velocity": torch_zeros(), "out_of_field": torch_zeros(), "collision": torch_zeros()}
        self.reset()

        self.train_team_name = "blue"
        self.copy_team_name = "red"
        self.n_agents = 2
        self.extras = {}

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
        self.all_actor_indices = torch.arange(self.args.num_envs * int(self.actors_per_env), dtype=torch.int32, device=self.args.sim_device).view(self.args.num_envs, self.actors_per_env)

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
        self.soccer_handles = []
        self.ball_handles = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add cartpole here in each environment
            soccer_handle = self.gym.create_actor(env, soccer_asset, pose, "soccer", i, 0, 0)
            self.gym.set_actor_dof_properties(env, soccer_handle, dof_props)
            self.soccer_handles.append(soccer_handle)

            ball_handle = self.gym.create_actor(env, ball_asset, ball_init_pose, "ball", i, 1, 0)
            self.ball_handles.append(ball_handle)

            envs.append(env)
        return envs, num_dof

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer

    def local_pos(self, global_pos, robot_xy, rotation_matrix):
        rotated_translation = torch.matmul(rotation_matrix, (global_pos - robot_xy).unsqueeze(-1))
        return rotated_translation
        
    def get_obs(self, env_ids=None):
        # get state observation from each environment id
        if env_ids is None:
            env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        pos = self.dof_pos.view(self.args.num_envs*self.num_player, 5)
        global_pos = pos[:,:2]
        global_ball = torch.repeat_interleave(self.ball_pos[:,:2], self.num_player, dim=0)

        yaw = pos[:,2]
        cos_angles = torch.cos(yaw)
        sin_angles = torch.sin(yaw)
        rotation_matrix = torch.stack([cos_angles, -sin_angles, sin_angles, cos_angles], dim=1).reshape(-1, 2, 2)
        
        view_ratio = math.tan(math.radians(60))
        local_ball = self.local_pos(global_ball, global_pos, rotation_matrix).squeeze()
        out_of_view = (local_ball[:,0] * view_ratio) < torch.abs(local_ball[:,1])
        local_ball[out_of_view, :] = -100.0

        obs = torch.zeros((self.args.num_envs * self.num_player, self.num_obs), device=self.args.sim_device)
        obs[:,:2] = global_pos
        obs[:,2] = yaw
        obs[:,3:5] = local_ball
        global_pos3 = torch.repeat_interleave(global_pos, 3, dim=0)
        robot_pos = torch.zeros((self.args.num_envs*self.num_player*3,2), device=self.args.sim_device)
        robot_pos[0::12,:] = global_pos[1::4,:2]
        robot_pos[1::12,:] = global_pos[2::4,:2]
        robot_pos[2::12,:] = global_pos[3::4,:2]
        robot_pos[3::12,:] = global_pos[0::4,:2]
        robot_pos[4::12,:] = global_pos[2::4,:2]
        robot_pos[5::12,:] = global_pos[3::4,:2]
        robot_pos[6::12,:] = global_pos[3::4,:2]
        robot_pos[7::12,:] = global_pos[0::4,:2]
        robot_pos[8::12,:] = global_pos[1::4,:2]
        robot_pos[9::12,:] = global_pos[2::4,:2]
        robot_pos[10::12,:] = global_pos[0::4,:2]
        robot_pos[11::12,:] = global_pos[1::4,:2]
        rotation_matrix3 = torch.repeat_interleave(rotation_matrix, 3, dim=0)
        local_robot = self.local_pos(robot_pos, global_pos3, rotation_matrix3).squeeze()
        out_of_view = (local_robot[:,0] * view_ratio) < torch.abs(local_robot[:,1]) 
        local_robot[out_of_view, :] = -100.0
        obs[:,5:7] = local_robot[0::3,:]
        obs[:,7:9] = local_robot[1::3,:]
        obs[:,9:11] = local_robot[2::3,:]
        repeated_ids = torch.repeat_interleave(env_ids, self.num_player)
        increment_ids = torch.arange(self.num_player, device=self.args.sim_device).repeat(env_ids.numel())
        expanded_env_ids = repeated_ids * 4 + increment_ids
        self.obs_buf[expanded_env_ids] = obs[expanded_env_ids]

    def get_reward(self):
        self.reward_buf[:], self.reset_buf[:], rew_goal, rew_ball_vel, rew_out_of_field, rew_collision= compute_reward(
            self.obs_buf,
            self.ball_pos,
            self.ball_vel,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )
        self.episode_sums["goal"] += rew_goal.reshape(-1, 4)[:, :2].flatten()
        self.episode_sums["ball_velocity"] += rew_ball_vel.reshape(-1, 4)[:, :2].flatten()
        self.episode_sums["out_of_field"] += rew_out_of_field.reshape(-1, 4)[:, :2].flatten()
        self.episode_sums["collision"] += rew_collision.reshape(-1, 4)[:, :2].flatten()

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities
        min_values = torch.tensor([-0.4, -0.2, 0, 0, 0, -4, -2.5, 0, 0, 0, 1.0, -0.2, 3.14, 0, 0, 1, -2.5, 3.14, 0, 0], device=self.args.sim_device)
        max_values = torch.tensor([-0.2, 0.2, 0, 0, 0, -1, 2.5, 0, 0, 0, 1.2, 0.2, 3.14, 0, 0, 4, 2.5, 3.14, 0, 0], device=self.args.sim_device)
        random_tensor = torch.rand((len(env_ids), self.num_dof), device=self.args.sim_device)
        positions = min_values + (max_values- min_values) * random_tensor

        self.root_states[env_ids] = self.root_init_state
        actor_indices = self.all_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

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

        obs_mask = (torch.arange(self.obs_buf.shape[0]) % 4 <= 1)
        obs = self.obs_buf[obs_mask, :].cpu().numpy().reshape(-1, 2, 11)
        c_obs = self.obs_buf[~obs_mask, :].cpu().numpy().reshape(-1, 2, 11)
        available = np.tile(np.array([1] * self.actions.shape[0]),(self.args.num_envs,int(self.num_player/2),1))
        return obs, obs, available, c_obs, c_obs, available

    def each_reward(self):
        env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids])
            self.episode_sums[key][env_ids] = 0.
        return self.extras["episode"]

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
        each_dof_pos = self.dof_pos.view(self.args.num_envs*self.num_player, 5)
        angles = each_dof_pos[:, 2]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        rotation_matrix = torch.stack([cos_angles, -sin_angles, sin_angles, cos_angles], dim=1).reshape(-1, 2, 2)
        non_zero_rows = (each_dof_pos[:, 3] > 0.1) | (each_dof_pos[:, 4] > 0.1)
        actions_flat = torch.tensor(actions[0].flatten())
        actions_flat[non_zero_rows] = 8
        actions_tensor = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        actions0 = self.actions[actions_flat]
        translation = actions0[:, :2].unsqueeze(-1)
        rotated_translation = torch.matmul(rotation_matrix, translation).squeeze(-1)
        actions0[:,:2] = rotated_translation
        actions_tensor[:] = actions0.flatten()
        positions = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        positions0 = self.dof_pos[:].reshape(self.args.num_envs*self.num_player, 5)
        positions0[non_zero_rows,3] = positions0[non_zero_rows,4] = 0
        positions[:] = positions0.reshape(-1)

        # simulate and render
        for i in range(int(self.walking_period / self.dt)):
            positions += actions_tensor * self.dt
            target_pos = gymtorch.unwrap_tensor(positions)
            self.gym.set_dof_position_target_tensor(self.sim, target_pos)
            self.simulate()
            if not self.args.headless:
                self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()

        obs_mask = (torch.arange(self.obs_buf.shape[0]) % 4 <= 1)
        obs = self.obs_buf[obs_mask, :].cpu().numpy().reshape(-1, 2, 11)
        c_obs = self.obs_buf[~obs_mask, :].cpu().numpy().reshape(-1, 2, 11)
        rewards = self.reward_buf[obs_mask].cpu().numpy().reshape(-1, 2, 1)
        c_rewards = self.reward_buf[~obs_mask].cpu().numpy().reshape(-1, 2, 1)
        dones = np.repeat(np.array([self.reset_buf.cpu().numpy()]).reshape(-1, 1), 2, axis=1)
        infos = np.tile(np.array([{"score_reward": 0} for _ in range(int(self.num_player/2))]),(self.args.num_envs,1))
        c_infos = np.tile(np.array([{"score_reward": 0} for _ in range(int(self.num_player/2))]),(self.args.num_envs,1))
        available = np.tile(np.array([1] * self.actions.shape[0]),(self.args.num_envs,int(self.num_player/2),1))
        c_available = np.tile(np.array([1] * self.actions.shape[0]),(self.args.num_envs,int(self.num_player/2),1))

        return obs, obs, rewards, dones, infos, available, c_obs, c_obs, c_rewards, dones, c_infos, c_available

# define reward function using JIT
#@torch.jit.script
def compute_reward(obs_buf, ball_pos, ball_vel, reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    num_player = 4
    goal_reward = 1000.0
    velocity_reward = 100.0
    out_of_field_reward = -0.1
    collision_reward = -0.1
    
    # goal reward
    extended_ball_pos = torch.repeat_interleave(ball_pos[:,:], num_player, dim=0)
    extended_ball_pos[2::num_player, 0] *= -1
    extended_ball_pos[3::num_player, 0] *= -1
    rew_goal = torch.zeros(extended_ball_pos.shape[0], device=obs_buf.device)
    rew_goal = torch.where((extended_ball_pos[:,0] > 4.5) & (torch.abs(extended_ball_pos[:,1]) < 1.3), torch.ones_like(rew_goal)*goal_reward, rew_goal)
    rew_goal = torch.where((extended_ball_pos[:,0] < -4.5) & (torch.abs(extended_ball_pos[:,1]) < 1.3), torch.ones_like(rew_goal)*(-goal_reward), rew_goal)
    
    # ball velocity reward
    extended_ball_vel = torch.repeat_interleave(ball_vel[:,:], num_player, dim=0)
    extended_ball_vel[2::num_player, 0] *= -1
    extended_ball_vel[3::num_player, 0] *= -1
    backward_ball = extended_ball_vel[:, 0] < 0
    extended_ball_vel[backward_ball, :] = 0.0
    goal_pos = torch.tensor([4.5, 0.0], device=obs_buf.device).repeat(extended_ball_pos.shape[0], 1)
    vectors = (goal_pos - extended_ball_pos)
    norm = torch.norm(vectors)
    unit_vectors = vectors / norm
    unit_vectors_3d = unit_vectors.unsqueeze(1)
    extended_ball_vel_3d = extended_ball_vel.unsqueeze(2)
    dot_products = torch.bmm(unit_vectors_3d, extended_ball_vel_3d).squeeze()
    local_ball = obs_buf[:,3:5]
    ball_distances = torch.sum(local_ball**2, dim=1)
    without_1_0m = ball_distances > 1.0**2
    dot_products[without_1_0m] = 0.0
    rew_ball_vel = dot_products * velocity_reward

    # out of field reward
    rew_out_of_field = torch.zeros(obs_buf.shape[0], device=obs_buf.device)
    robot_pos = obs_buf[:, :2]
    out_of_field = (torch.abs(robot_pos[:, 0]) > 5.0) | (torch.abs(robot_pos[:, 1]) > 3.5)
    rew_out_of_field[out_of_field] += out_of_field_reward

    # collision reward
    rew_collision = torch.zeros(obs_buf.shape[0], device=obs_buf.device)
    collision_robot1 = torch.sum(obs_buf[:,5:7]**2, dim=1) < (0.3**2)
    collision_robot2 = torch.sum(obs_buf[:,7:9]**2, dim=1) < (0.3**2)
    collision_robot3 = torch.sum(obs_buf[:,9:11]**2, dim=1) < (0.3**2)
    collision = collision_robot1 | collision_robot2 | collision_robot3
    rew_collision[collision] += collision_reward

    reward = rew_goal + rew_ball_vel + rew_out_of_field + rew_collision

    # reset
    reset = torch.where((torch.abs(ball_pos[:,0]) > 4.5) | (torch.abs(ball_pos[:,1]) > 3), torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset)
    return reward, reset, rew_goal, rew_ball_vel, rew_out_of_field, rew_collision
