import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import time
import math
import os
from collections import deque
from typing import Optional, List, Dict, Any, Tuple, Literal, Union

from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon # For mapper visualization
import imageio.v2 as imageio
from PIL import Image
import glob
from scipy.spatial import ConvexHull, Delaunay
import warnings

# --- Core Constants (from original configs.py) ---
# These are fundamental to state and trajectory dimensions.
# If these change in your main project, they might need updating here too,
# or ideally, AppSimConfig should be flexible enough or derive them.
# For now, hardcoding based on the provided configs.py.
ORIG_CORE_STATE_DIM = 8
ORIG_CORE_ACTION_DIM = 1
ORIG_TRAJECTORY_REWARD_DIM = 1
ORIG_DEFAULT_TRAJECTORY_FEATURE_DIM = ORIG_CORE_STATE_DIM + ORIG_CORE_ACTION_DIM + ORIG_TRAJECTORY_REWARD_DIM

# --- Configuration Models (Simplified for Simulation) ---

class LocationSimConfig(BaseModel): # Based on original Position
    x: float = 0.0
    y: float = 0.0
    # No depth for 2D oil spill problem

class VelocitySimConfig(BaseModel): # Based on original Velocity
    x: float = 0.0
    y: float = 0.0

class RandomizationRangeSimConfig(BaseModel): # Based on original RandomizationRange
    x_range: Tuple[float, float] = Field((10.0, 90.0))
    y_range: Tuple[float, float] = Field((10.0, 90.0))

class MapperSimConfig(BaseModel): # Based on original MapperConfig
    min_oil_points_for_estimate: int = Field(3)

class SACSimConfig(BaseModel): # Subset of original SACConfig
    state_dim: int = Field(ORIG_CORE_STATE_DIM)
    action_dim: int = Field(ORIG_CORE_ACTION_DIM)
    hidden_dims: List[int] = Field([128, 128])
    log_std_min: int = Field(-20)
    log_std_max: int = Field(1)
    use_rnn: bool = Field(False)
    rnn_type: Literal['lstm', 'gru'] = Field('lstm')
    rnn_hidden_size: int = Field(68)
    rnn_num_layers: int = Field(1)

class PPOSimConfig(BaseModel): # Subset of original PPOConfig
    state_dim: int = Field(ORIG_CORE_STATE_DIM)
    action_dim: int = Field(ORIG_CORE_ACTION_DIM)
    hidden_dim: int = Field(256) # Original PPO uses single hidden_dim
    log_std_min: int = Field(-20)
    log_std_max: int = Field(1)
    use_rnn: bool = Field(False)
    rnn_type: Literal['lstm', 'gru'] = Field('gru') # Note: PPO default is gru
    rnn_hidden_size: int = Field(64)
    rnn_num_layers: int = Field(1)

class WorldSimConfig(BaseModel): # Based on original WorldConfig
    # Core dimensions for this sim config instance (should align with agent needs)
    CORE_STATE_DIM: int = Field(ORIG_CORE_STATE_DIM)
    CORE_ACTION_DIM: int = Field(ORIG_CORE_ACTION_DIM)
    TRAJECTORY_REWARD_DIM: int = Field(ORIG_TRAJECTORY_REWARD_DIM)

    dt: float = Field(1.0)
    world_size: Tuple[float, float] = Field((125.0, 125.0))
    normalize_coords: bool = Field(True)
    agent_speed: float = Field(3.0)
    yaw_angle_range: Tuple[float, float] = Field((-math.pi / 6, math.pi / 6))
    num_sensors: int = Field(5)
    sensor_distance: float = Field(2.5)
    sensor_radius: float = Field(4.0)
    agent_initial_location: LocationSimConfig = Field(default_factory=lambda: LocationSimConfig(x=50.0, y=10.0))
    randomize_agent_initial_location: bool = Field(True)
    agent_randomization_ranges: RandomizationRangeSimConfig = Field(
        default_factory=lambda: RandomizationRangeSimConfig(x_range=(25.0, 100.0), y_range=(25.0, 100.0))
    )
    num_oil_points: int = Field(200)
    num_water_points: int = Field(400) # Added for completeness, though not directly in UI
    oil_cluster_std_dev_range: Tuple[float, float] = Field((8.0, 10.0))
    randomize_oil_cluster: bool = Field(True)
    oil_center_randomization_range: RandomizationRangeSimConfig = Field(
        default_factory=lambda: RandomizationRangeSimConfig(x_range=(25.0, 100.0), y_range=(25.0, 100.0))
    )
    initial_oil_center: LocationSimConfig = Field(default_factory=lambda: LocationSimConfig(x=50.0, y=50.0))
    initial_oil_std_dev: float = Field(10.0)
    min_initial_separation_distance: float = Field(40.0)
    trajectory_length: int = Field(10)
    trajectory_feature_dim: int = Field(ORIG_DEFAULT_TRAJECTORY_FEATURE_DIM)
    
    # Reward params (to match original world.py _calculate_reward)
    success_metric_threshold: float = Field(0.95)
    terminate_on_success: bool = Field(True) # For `done` logic
    terminate_out_of_bounds: bool = Field(True) # For `done` logic
    metric_improvement_scale: float = Field(50.0)
    step_penalty: float = Field(0.0)
    new_oil_detection_bonus: float = Field(0.0)
    out_of_bounds_penalty: float = Field(20.0)
    success_bonus: float = Field(55.0)
    uninitialized_mapper_penalty: float = Field(0.0)
    
    mapper_config: MapperSimConfig = Field(default_factory=MapperSimConfig)
    seeds: List[int] = Field([]) # For reproducible runs

class VisualizationSimConfig(BaseModel): # Based on original VisualizationConfig
    figure_size: tuple = Field((10, 10))
    max_trajectory_points: int = Field(100) # Increased for smoother sim viz
    gif_frame_duration: float = Field(0.1) # Corresponds to 10 FPS
    delete_frames_after_gif: bool = Field(True)
    sensor_marker_size: int = Field(10)
    sensor_color_oil: str = Field("red")
    sensor_color_water: str = Field("blue")
    plot_oil_points: bool = Field(True)
    plot_water_points: bool = Field(False) # Usually off by default
    point_marker_size: int = Field(2)


class AppSimConfig(BaseModel):
    sac: Optional[SACSimConfig] = None
    ppo: Optional[PPOSimConfig] = None
    world: WorldSimConfig
    # Mapper config is part of WorldSimConfig, no separate top-level mapper here.
    visualization: VisualizationSimConfig
    cuda_device: str = Field("cpu") # Simulation bundle always runs on CPU
    algorithm: str = Field("sac")

    class Config:
        arbitrary_types_allowed = True

# --- World Objects (from original world_objects.py) ---
class VelocitySim:
    def __init__(self, x: float, y: float):
        self.x = x; self.y = y
    def is_moving(self) -> bool: return self.x != 0 or self.y != 0
    def get_heading(self) -> float:
        if not self.is_moving(): return 0.0
        return math.atan2(self.y, self.x)
    def __str__(self) -> str: return f"VelSim:(vx:{self.x:.2f}, vy:{self.y:.2f})"

class LocationSim:
    def __init__(self, x: float, y: float):
        self.x = x; self.y = y
    def update(self, velocity: VelocitySim, dt: float = 1.0):
        self.x += velocity.x * dt; self.y += velocity.y * dt
    def distance_to(self, other_loc: 'LocationSim') -> float:
        return math.sqrt((self.x - other_loc.x)**2 + (self.y - other_loc.y)**2)
    def get_normalized(self, world_size: Tuple[float, float]) -> Tuple[float, float]:
        norm_x = max(0.0, min(1.0, self.x / world_size[0] if world_size[0] > 0 else 0.0))
        norm_y = max(0.0, min(1.0, self.y / world_size[1] if world_size[1] > 0 else 0.0))
        return norm_x, norm_y
    def __str__(self) -> str: return f"LocSim:(x:{self.x:.2f}, y:{self.y:.2f})"

class ObjectSim:
    def __init__(self, location: LocationSim, velocity: VelocitySim = None, name: str = None):
        self.name = name if name else "SimObject"
        self.location = location
        self.velocity = velocity if velocity is not None else VelocitySim(0.0, 0.0)
    def update_position(self, dt: float = 1.0):
        if self.velocity and self.velocity.is_moving():
            self.location.update(self.velocity, dt)
    def get_heading(self) -> float: return self.velocity.get_heading()
    def __str__(self) -> str: return f"{self.name}: {self.location}, {self.velocity}"

# --- Mapper (from original mapper.py) ---
class MapperSim:
    def __init__(self, config: MapperSimConfig):
        self.config = config
        self.oil_sensor_locations: List[LocationSim] = []
        self.water_sensor_locations: List[LocationSim] = [] # Not directly used by original Mapper logic but good for sim
        self.estimated_hull: Optional[ConvexHull] = None
        self.hull_vertices: Optional[np.ndarray] = None

    def reset(self):
        self.oil_sensor_locations = []; self.water_sensor_locations = []
        self.estimated_hull = None; self.hull_vertices = None

    def add_measurement(self, sensor_location: LocationSim, is_oil_detected: bool):
        if is_oil_detected:
            if not any(abs(p.x - sensor_location.x) < 1e-6 and abs(p.y - sensor_location.y) < 1e-6 for p in self.oil_sensor_locations):
                self.oil_sensor_locations.append(sensor_location)
        else:
             if not any(abs(p.x - sensor_location.x) < 1e-6 and abs(p.y - sensor_location.y) < 1e-6 for p in self.water_sensor_locations):
                self.water_sensor_locations.append(sensor_location)

    def estimate_spill(self):
        self.estimated_hull = None; self.hull_vertices = None
        if len(self.oil_sensor_locations) < self.config.min_oil_points_for_estimate: return
        oil_points_np = np.array([[p.x, p.y] for p in self.oil_sensor_locations])
        unique_oil_points = np.unique(oil_points_np, axis=0)
        if unique_oil_points.shape[0] < 3: return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hull = ConvexHull(unique_oil_points, qhull_options='QJ')
            self.estimated_hull = hull
            self.hull_vertices = unique_oil_points[hull.vertices]
        except Exception: self.estimated_hull = None; self.hull_vertices = None

    def is_inside_estimate(self, point: LocationSim) -> bool:
        if self.estimated_hull is None or self.hull_vertices is None or len(self.hull_vertices) < 3: return False
        point_np = np.array([point.x, point.y])
        try:
            if len(self.hull_vertices) < 3: return False
            delaunay_hull = Delaunay(self.hull_vertices, qhull_options='QJ')
            return delaunay_hull.find_simplex(point_np) >= 0
        except Exception: return False

# --- RL AGENT NETWORKS (EVALUATION VERSIONS) ---
# SAC Actor
class ActorNetEvalSAC(nn.Module):
    def __init__(self, config: SACSimConfig):
        super(ActorNetEvalSAC, self).__init__()
        self.config = config; self.use_rnn = config.use_rnn
        self.state_dim = config.state_dim; self.action_dim = config.action_dim
        
        if self.use_rnn:
            self.rnn_hidden_size = config.rnn_hidden_size; self.rnn_num_layers = config.rnn_num_layers
            rnn_input_dim = self.state_dim
            if config.rnn_type == 'lstm': self.rnn = nn.LSTM(rnn_input_dim, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True)
            elif config.rnn_type == 'gru': self.rnn = nn.GRU(rnn_input_dim, config.rnn_hidden_size, config.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN: {config.rnn_type}")
            mlp_input_dim = config.rnn_hidden_size
        else: mlp_input_dim = self.state_dim; self.rnn = None
        
        self.layers = nn.ModuleList(); current_dim = mlp_input_dim
        hidden_dims = config.hidden_dims if config.hidden_dims else [256, 256]
        for hidden_dim_val in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim_val))
            self.layers.append(nn.ReLU()); current_dim = hidden_dim_val
        
        # --- CORRECTED LAYER NAMES ---
        self.mean = nn.Linear(current_dim, self.action_dim) # Was mean_layer
        self.log_std = nn.Linear(current_dim, self.action_dim) # Was log_std_layer
        # --- END CORRECTION ---

        self.log_std_min = config.log_std_min; self.log_std_max = config.log_std_max

    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn and self.rnn: 
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            mlp_input = rnn_output[:, -1, :]
        else: mlp_input = network_input
        
        x = mlp_input
        for layer_module in self.layers: x = layer_module(x) 
        mean = self.mean(x); log_std = self.log_std(x) # Use corrected names
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std, next_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None, evaluate: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        mean, log_std, next_hidden_state = self.forward(network_input, hidden_state)
        if evaluate: 
            action_normalized = torch.tanh(mean)
            log_prob = torch.zeros_like(action_normalized) 
            return action_normalized, log_prob, action_normalized, next_hidden_state 
        else: 
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action_normalized = torch.tanh(x_t)
            log_prob_unbounded = normal.log_prob(x_t)
            clamped_tanh = action_normalized.clamp(-0.999999, 0.999999)
            log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
            log_prob = log_prob_unbounded - log_det_jacobian
            log_prob = log_prob.sum(1, keepdim=True)
            return action_normalized, log_prob, torch.tanh(mean), next_hidden_state


    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.config.rnn_num_layers, batch_size, self.config.rnn_hidden_size).to(device)
        if self.config.rnn_type == 'lstm': return (h_zeros, torch.zeros(self.config.rnn_num_layers, batch_size, self.config.rnn_hidden_size).to(device))
        return h_zeros

# PPO Policy Network
class PolicyNetworkNetEvalPPO(nn.Module):
    def __init__(self, ppo_config: PPOSimConfig):
        super(PolicyNetworkNetEvalPPO, self).__init__()
        self.ppo_config = ppo_config
        self.use_rnn = ppo_config.use_rnn; self.state_dim = ppo_config.state_dim; self.action_dim = ppo_config.action_dim
        
        if self.use_rnn:
            self.rnn_hidden_size = ppo_config.rnn_hidden_size; self.rnn_num_layers = ppo_config.rnn_num_layers
            rnn_input_dim = self.state_dim
            if ppo_config.rnn_type == 'lstm': self.rnn = nn.LSTM(rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True)
            elif ppo_config.rnn_type == 'gru': self.rnn = nn.GRU(rnn_input_dim, self.rnn_hidden_size, self.rnn_num_layers, batch_first=True)
            else: raise ValueError(f"Unsupported RNN: {ppo_config.rnn_type}")
            mlp_input_dim = self.rnn_hidden_size
        else: self.rnn = None; mlp_input_dim = self.state_dim

        self.fc1 = nn.Linear(mlp_input_dim, ppo_config.hidden_dim)
        self.fc2 = nn.Linear(ppo_config.hidden_dim, ppo_config.hidden_dim)
        self.mean_layer = nn.Linear(ppo_config.hidden_dim, self.action_dim)
        # Log_std is usually a parameter in PPO, not output of a layer, for eval it's fine to fix or load
        self.log_std_val = nn.Parameter(torch.zeros(1, self.action_dim)) # Matching original PPO.py structure
        self.log_std_min = ppo_config.log_std_min; self.log_std_max = ppo_config.log_std_max


    def forward(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        next_hidden_state = None
        if self.use_rnn and self.rnn:
            rnn_output, next_hidden_state = self.rnn(network_input, hidden_state)
            mlp_features = rnn_output[:, -1, :]
        else: mlp_features = network_input
        
        x = F.relu(self.fc1(mlp_features)); x = F.relu(self.fc2(x))
        action_mean = self.mean_layer(x)
        action_log_std = torch.clamp(self.log_std_val, self.log_std_min, self.log_std_max) # Use the parameter
        action_std = action_log_std.exp().expand_as(action_mean) # Expand to match mean shape
        return action_mean, action_std, next_hidden_state

    def sample(self, network_input: torch.Tensor, hidden_state: Optional[Tuple] = None, evaluate: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        mean, std, next_hidden_state = self.forward(network_input, hidden_state)
        if evaluate: # Deterministic action
            action_normalized = torch.tanh(mean)
            log_prob = torch.zeros_like(action_normalized) # Placeholder for eval
            return action_normalized, log_prob, next_hidden_state
        else: # Stochastic action (original PPO.py sample logic)
            distribution = Normal(mean, std)
            x_t = distribution.sample()
            action_normalized = torch.tanh(x_t)
            log_prob_unbounded = distribution.log_prob(x_t)
            clamped_tanh = action_normalized.clamp(-0.999999, 0.999999)
            log_det_jacobian = torch.log(1.0 - clamped_tanh.pow(2) + 1e-7)
            log_prob = log_prob_unbounded - log_det_jacobian
            log_prob = log_prob.sum(1, keepdim=True)
            return action_normalized, log_prob, next_hidden_state
        
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Tuple]:
        if not self.use_rnn: return None
        h_zeros = torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device)
        if self.ppo_config.rnn_type == 'lstm': return (h_zeros, torch.zeros(self.ppo_config.rnn_num_layers, batch_size, self.ppo_config.rnn_hidden_size).to(device))
        return h_zeros

# --- RL AGENT MAIN CLASSES (EVALUATION VERSIONS) ---
class SACAgentEval:
    def __init__(self, agent_config: SACSimConfig, device_str: str):
        self.config = agent_config
        self.device = torch.device("cpu") 
        self.use_rnn = agent_config.use_rnn
        self.actor = ActorNetEvalSAC(agent_config).to(self.device)

    def select_action(self, state_dict: Dict[str, Any], actor_hidden_state: Optional[Tuple] = None) -> Tuple[float, Optional[Tuple]]:
        with torch.no_grad():
            if self.use_rnn:
                # SAC RNN Actor expects a sequence of basic states: (batch=1, seq_len, state_dim)
                # state_dict['full_trajectory'] is (seq_len, feature_dim)
                # We need the state part: (seq_len, state_dim)
                actor_input_seq = torch.FloatTensor(state_dict['full_trajectory'][:, :self.config.state_dim]).to(self.device).unsqueeze(0)
            else:
                # SAC MLP Actor expects the last basic state: (batch=1, state_dim)
                actor_input_seq = torch.FloatTensor(state_dict['basic_state']).to(self.device).unsqueeze(0)
            
            self.actor.eval()
            # For SAC eval, we use the deterministic action (tanh of mean). The third element from sample() is this.
            _, _, action_normalized_deterministic, next_actor_hidden_state = self.actor.sample(actor_input_seq, actor_hidden_state, evaluate=True)
        
        return action_normalized_deterministic.detach().cpu().numpy()[0, 0], next_actor_hidden_state

    def load_model(self, path: str):
        if not os.path.exists(path): print(f"SAC Model file not found: {path}"); return
        checkpoint = torch.load(path, map_location=self.device) # weights_only=False by default
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()
        print(f"SAC Eval Agent loaded actor from {path}")

class PPOAgentEval:
    def __init__(self, agent_config: PPOSimConfig, device_str: str):
        self.config = agent_config
        self.device = torch.device("cpu") 
        self.use_rnn = agent_config.use_rnn
        self.actor = PolicyNetworkNetEvalPPO(agent_config).to(self.device)

    def select_action(self, basic_state_tuple: Tuple, actor_hidden_state: Optional[Tuple]=None) -> Tuple[float, Optional[Tuple]]:
        with torch.no_grad():
            if self.use_rnn:
                # PPO RNN Actor expects current basic state shaped as (batch=1, seq_len=1, state_dim)
                network_input_tensor = torch.FloatTensor(basic_state_tuple).to(self.device).unsqueeze(0).unsqueeze(0)
            else:
                # PPO MLP Actor expects current basic state shaped as (batch=1, state_dim)
                network_input_tensor = torch.FloatTensor(basic_state_tuple).to(self.device).unsqueeze(0)
            
            self.actor.eval()
            # PPO evaluate usually means taking the mean of the distribution
            action_normalized, _, next_actor_h_detached = self.actor.sample(network_input_tensor, actor_hidden_state, evaluate=True)
        
        return action_normalized.detach().cpu().numpy().item(), next_actor_h_detached

    def load_model(self, path: str):
        if not os.path.exists(path): print(f"PPO Model file not found: {path}"); return
        checkpoint = torch.load(path, map_location=self.device) # weights_only=False by default
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        # PPO might store log_std in checkpoint if it's a parameter of the network
        if 'log_std' in checkpoint and hasattr(self.actor, 'log_std_val'): # Ensure log_std_val exists
             self.actor.log_std_val.data.copy_(checkpoint['log_std'].to(self.device))
        elif hasattr(self.actor, 'log_std_val'): # Checkpoint might be older
             print("PPO eval: log_std not found in checkpoint for actor, using default network parameter.")

        self.actor.eval()
        print(f"PPO Eval Agent loaded actor from {path}")

# --- World Simulation (adapted from original world.py) ---
class WorldSim:
    def __init__(self, world_sim_config: WorldSimConfig,
                 initial_agent_loc_override: Optional[Tuple[float, float]] = None,
                 initial_oil_center_override: Optional[Tuple[float, float]] = None,
                 initial_oil_std_dev_override: Optional[float] = None,
                 seed_override: Optional[int] = None
                ):
        self.world_config = world_sim_config
        self.initial_agent_loc_override = initial_agent_loc_override
        self.initial_oil_center_override = initial_oil_center_override
        self.initial_oil_std_dev_override = initial_oil_std_dev_override
        self.seed_override = seed_override # For specific seed runs

        # Directly use dimensions from the passed config
        self.CORE_STATE_DIM = world_sim_config.CORE_STATE_DIM
        self.CORE_ACTION_DIM = world_sim_config.CORE_ACTION_DIM
        self.TRAJECTORY_REWARD_DIM = world_sim_config.TRAJECTORY_REWARD_DIM
        # self.feature_dim should be consistent with these
        self.feature_dim = self.CORE_STATE_DIM + self.CORE_ACTION_DIM + self.TRAJECTORY_REWARD_DIM
        if self.feature_dim != world_sim_config.trajectory_feature_dim:
            print(f"Warning (WorldSim): Calculated feature_dim {self.feature_dim} mismatches config {world_sim_config.trajectory_feature_dim}. Using calculated.")


        self.dt = world_sim_config.dt
        self.agent_speed = world_sim_config.agent_speed
        self.max_yaw_change = world_sim_config.yaw_angle_range[1]
        self.num_sensors = world_sim_config.num_sensors
        self.sensor_distance = world_sim_config.sensor_distance
        self.sensor_radius = world_sim_config.sensor_radius
        self.trajectory_length = world_sim_config.trajectory_length
        self.world_size = world_sim_config.world_size
        self.normalize_coords = world_sim_config.normalize_coords

        self.agent: Optional[ObjectSim] = None
        self.true_oil_points: List[LocationSim] = []
        self.true_water_points: List[LocationSim] = []
        self.mapper: MapperSim = MapperSim(world_sim_config.mapper_config)
        
        self.seeds = world_sim_config.seeds # Store seeds from config
        self.seed_index = 0
        self.current_seed: Optional[int] = None

        self.reward: float = 0.0
        self.performance_metric: float = 0.0
        self.previous_performance_metric: float = 0.0
        self.done: bool = False
        self.current_step: int = 0
        self.last_sensor_reads: List[bool] = [False] * self.num_sensors
        self.reward_components: Dict[str, float] = {} # Initialized in reset

        self._trajectory_history = deque(maxlen=self.trajectory_length)
        self.reset(seed=self.seed_override)


    def _seed_environment(self, seed: Optional[int] = None):
        if seed is None: seed = random.randint(0, 2**32 - 1)
        self.current_seed = seed
        random.seed(self.current_seed); np.random.seed(self.current_seed)

    def reset(self, seed: Optional[int] = None):
        self.current_step = 0; self.done = False; self.reward = 0.0
        self.performance_metric = 0.0; self.previous_performance_metric = 0.0
        self.reward_components = {
            "metric_improvement": 0.0, "new_oil_detection": 0.0, "step_penalty": 0.0,
            "uninitialized_penalty": 0.0, "out_of_bounds_penalty": 0.0,
            "success_bonus": 0.0, "total": 0.0
        }

        reset_seed = seed
        if reset_seed is None: # If no override, use internal seed list
            if self.seeds:
                if self.seed_index >= len(self.seeds): self.seed_index = 0
                reset_seed = self.seeds[self.seed_index]; self.seed_index += 1
        self._seed_environment(reset_seed)

        world_w, world_h = self.world_size
        self.true_oil_points = []; self.true_water_points = []

        # Determine Oil Center
        if self.initial_oil_center_override:
            oil_center = LocationSim(x=self.initial_oil_center_override[0], y=self.initial_oil_center_override[1])
            oil_std_dev = self.initial_oil_std_dev_override if self.initial_oil_std_dev_override is not None else self.world_config.initial_oil_std_dev
        elif self.world_config.randomize_oil_cluster:
            ranges = self.world_config.oil_center_randomization_range
            oil_center = LocationSim(x=random.uniform(*ranges.x_range), y=random.uniform(*ranges.y_range))
            oil_std_dev = random.uniform(*self.world_config.oil_cluster_std_dev_range)
        else:
            cfg_loc = self.world_config.initial_oil_center
            oil_center = LocationSim(x=cfg_loc.x, y=cfg_loc.y)
            oil_std_dev = self.world_config.initial_oil_std_dev
        
        for _ in range(self.world_config.num_oil_points):
            px = np.random.normal(oil_center.x, oil_std_dev); py = np.random.normal(oil_center.y, oil_std_dev)
            self.true_oil_points.append(LocationSim(x=max(0.0, min(world_w, px)), y=max(0.0, min(world_h, py))))
        for _ in range(self.world_config.num_water_points):
            px = random.uniform(0, world_w); py = random.uniform(0, world_h)
            if not any(abs(p.x - px) < 1e-6 and abs(p.y - py) < 1e-6 for p in self.true_oil_points):
                 self.true_water_points.append(LocationSim(px,py))

        # Initialize Agent
        min_dist = self.world_config.min_initial_separation_distance; attempts = 0; max_attempts = 100
        agent_location: Optional[LocationSim] = None
        while attempts < max_attempts:
            potential_agent_loc: LocationSim
            if self.initial_agent_loc_override:
                potential_agent_loc = LocationSim(x=self.initial_agent_loc_override[0], y=self.initial_agent_loc_override[1])
            elif self.world_config.randomize_agent_initial_location:
                ranges = self.world_config.agent_randomization_ranges
                potential_agent_loc = LocationSim(x=random.uniform(*ranges.x_range), y=random.uniform(*ranges.y_range))
            else:
                cfg_loc = self.world_config.agent_initial_location
                potential_agent_loc = LocationSim(x=cfg_loc.x, y=cfg_loc.y)
            
            potential_agent_loc.x = max(0.0, min(world_w, potential_agent_loc.x))
            potential_agent_loc.y = max(0.0, min(world_h, potential_agent_loc.y))
            if potential_agent_loc.distance_to(oil_center) >= min_dist: agent_location = potential_agent_loc; break
            attempts += 1
            if self.initial_agent_loc_override or not self.world_config.randomize_agent_initial_location: # If fixed and too close, use it anyway
                 agent_location = potential_agent_loc; break 
        if agent_location is None: agent_location = potential_agent_loc # Fallback

        initial_heading = random.uniform(-math.pi, math.pi)
        agent_velocity = VelocitySim(x=self.agent_speed * math.cos(initial_heading), y=self.agent_speed * math.sin(initial_heading))
        self.agent = ObjectSim(location=agent_location, velocity=agent_velocity, name="agent")

        self.mapper.reset()
        sensor_locs_t0, sensor_reads_t0 = self._get_sensor_readings()
        for loc, read in zip(sensor_locs_t0, sensor_reads_t0): self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill(); self._calculate_performance_metric()
        self.previous_performance_metric = self.performance_metric
        self.last_sensor_reads = sensor_reads_t0
        
        self._calculate_reward(sensor_reads_t0) # Initial reward (r1)
        self.reward = self.reward_components["total"]

        self._initialize_trajectory_history()
        return self.encode_state()

    def _get_sensor_locations(self) -> List[LocationSim]:
        sensor_locations = []
        if not self.agent: return []
        agent_loc = self.agent.location; agent_heading = self.agent.get_heading()
        angle_offsets = [0.0] if self.num_sensors == 1 else np.linspace(-math.pi / 2, math.pi / 2, self.num_sensors)
        for angle_offset in angle_offsets:
            sensor_angle = agent_heading + angle_offset
            sx = agent_loc.x + self.sensor_distance * math.cos(sensor_angle)
            sy = agent_loc.y + self.sensor_distance * math.sin(sensor_angle)
            sensor_locations.append(LocationSim(x=max(0.0, min(self.world_size[0], sx)), y=max(0.0, min(self.world_size[1], sy))))
        return sensor_locations

    def _get_sensor_readings(self) -> Tuple[List[LocationSim], List[bool]]:
        sensor_locations = self._get_sensor_locations(); sensor_readings = [False] * self.num_sensors
        if not self.true_oil_points: return sensor_locations, sensor_readings
        for i, sensor_loc in enumerate(sensor_locations):
            for oil_point in self.true_oil_points:
                if sensor_loc.distance_to(oil_point) <= self.sensor_radius:
                    sensor_readings[i] = True; break
        return sensor_locations, sensor_readings

    def _calculate_performance_metric(self):
        if self.mapper.estimated_hull is None or not self.true_oil_points:
            self.performance_metric = 0.0; return
        points_inside = sum(1 for oil_point in self.true_oil_points if self.mapper.is_inside_estimate(oil_point))
        self.performance_metric = points_inside / len(self.true_oil_points)

    def _get_basic_state_tuple_normalized(self) -> Tuple:
        if not self.agent: raise RuntimeError("Agent not initialized in WorldSim")
        _, sensor_reads_bool = self._get_sensor_readings()
        sensor_reads_float = [1.0 if read else 0.0 for read in sensor_reads_bool]
        agent_loc_norm = self.agent.location.get_normalized(self.world_size)
        agent_heading_norm = self.agent.get_heading() / math.pi
        state_list = sensor_reads_float + list(agent_loc_norm) + [agent_heading_norm]
        
        if len(state_list) != self.CORE_STATE_DIM:
            # This should ideally not happen if num_sensors and config CORE_STATE_DIM are aligned
            # Example: if CORE_STATE_DIM = 8 and num_sensors = 5, then 5 + 2 + 1 = 8. Correct.
            # If num_sensors changes without CORE_STATE_DIM changing, this will fail.
            # For simulation, we assume AppSimConfig passes consistent num_sensors and CORE_STATE_DIM.
            print(f"Warning (WorldSim): Basic state dim mismatch. Expected {self.CORE_STATE_DIM}, got {len(state_list)}. Check num_sensors.")
            # Pad or truncate if absolutely necessary, but this indicates a config issue.
            if len(state_list) < self.CORE_STATE_DIM:
                state_list.extend([0.0] * (self.CORE_STATE_DIM - len(state_list)))
            else:
                state_list = state_list[:self.CORE_STATE_DIM]
                
        return tuple(state_list)


    def _initialize_trajectory_history(self):
        if self.agent is None: raise ValueError("Agent must be initialized before trajectory history.")
        initial_basic_state_norm = self._get_basic_state_tuple_normalized()
        initial_feature = np.concatenate([
            np.array(initial_basic_state_norm, dtype=np.float32),
            np.array([0.0], dtype=np.float32), # initial_action
            np.array([self.reward], dtype=np.float32) # initial_reward (r1)
        ])
        if len(initial_feature) != self.feature_dim:
             raise ValueError(f"Sim Feature dim mismatch. Expected {self.feature_dim}, got {len(initial_feature)}")
        self._trajectory_history.clear()
        for _ in range(self.trajectory_length): self._trajectory_history.append(initial_feature)

    def step(self, yaw_change_normalized: float, terminal_step: bool = False):
        if self.done: return self.encode_state()
        if not self.agent: raise RuntimeError("Agent not initialized in WorldSim step")

        prev_basic_state_norm = self._get_basic_state_tuple_normalized(); prev_action = yaw_change_normalized
        sensor_locs_t, sensor_reads_t = self._get_sensor_readings()

        yaw_change = yaw_change_normalized * self.max_yaw_change
        current_heading = self.agent.get_heading(); new_heading = (current_heading + yaw_change + math.pi) % (2 * math.pi) - math.pi
        self.agent.velocity = VelocitySim(self.agent_speed * math.cos(new_heading), self.agent_speed * math.sin(new_heading))
        self.agent.update_position(self.dt)

        ax, ay = self.agent.location.x, self.agent.location.y; wx, wy = self.world_size
        terminated_by_bounds = False
        if self.world_config.terminate_out_of_bounds and not (0 <= ax <= wx and 0 <= ay <= wy):
            terminated_by_bounds = True
            self.reward_components["out_of_bounds_penalty"] = -self.world_config.out_of_bounds_penalty
        
        if terminated_by_bounds:
            self.done = True; self.reward = sum(self.reward_components.values())
            self.reward_components["total"] = self.reward
            current_feature_vector = np.concatenate([np.array(prev_basic_state_norm, dtype=np.float32), np.array([prev_action], dtype=np.float32), np.array([self.reward], dtype=np.float32)])
            self._trajectory_history.append(current_feature_vector)
            return self.encode_state()

        for loc, read in zip(sensor_locs_t, sensor_reads_t): self.mapper.add_measurement(loc, read)
        self.mapper.estimate_spill(); self._calculate_performance_metric()
        
        self._calculate_reward(sensor_reads_t) # Calculates r_{t+1} components

        self.current_step += 1
        success = self.performance_metric >= self.world_config.success_metric_threshold
        terminated_by_success = success and self.world_config.terminate_on_success
        terminated_by_steps = terminal_step or self.current_step >= 350 # Hardcoded max steps from original TrainingConfig as fallback
        
        self.done = terminated_by_success or terminated_by_steps
        if terminated_by_success and not self.previous_performance_metric >= self.world_config.success_metric_threshold:
             self.reward_components["success_bonus"] = self.world_config.success_bonus
        
        self.reward = sum(self.reward_components.values()); self.reward_components["total"] = self.reward
        current_feature_vector = np.concatenate([np.array(prev_basic_state_norm, dtype=np.float32), np.array([prev_action], dtype=np.float32), np.array([self.reward], dtype=np.float32)])
        if len(current_feature_vector) != self.feature_dim:
             raise ValueError(f"Sim Step Feature dim mismatch. Expected {self.feature_dim}, got {len(current_feature_vector)}")
        self._trajectory_history.append(current_feature_vector)
        
        self.previous_performance_metric = self.performance_metric
        self.last_sensor_reads = sensor_reads_t
        return self.encode_state()

    def _calculate_reward(self, current_sensor_readings: List[bool]): # Matches original world.py
        cfg = self.world_config; current_metric_value = self.performance_metric
        components_to_reset = [k for k in self.reward_components if k not in ["out_of_bounds_penalty", "success_bonus", "total"]]
        for key in components_to_reset: self.reward_components[key] = 0.0
        self.reward_components["step_penalty"] = -cfg.step_penalty
        if self.mapper.estimated_hull is None:
            self.reward_components["uninitialized_penalty"] = -cfg.uninitialized_mapper_penalty
        else:
            metric_delta = current_metric_value - self.previous_performance_metric
            self.reward_components["metric_improvement"] = cfg.metric_improvement_scale * max(0, metric_delta)
            new_detections = sum(1 for i in range(self.num_sensors) if current_sensor_readings[i] and not self.last_sensor_reads[i])
            if new_detections > 0: self.reward_components["new_oil_detection"] = cfg.new_oil_detection_bonus
        self.reward_components["total"] = sum(v for k, v in self.reward_components.items() if k not in ["success_bonus", "out_of_bounds_penalty", "total"])


    def encode_state(self) -> Dict[str, Any]:
        basic_state_t_plus_1_norm = self._get_basic_state_tuple_normalized()
        full_trajectory_norm = np.array(self._trajectory_history, dtype=np.float32)
        if full_trajectory_norm.shape != (self.trajectory_length, self.feature_dim):
            # Attempt to recover if shape is wrong (e.g. history not full yet on first few steps)
            # This usually happens if _initialize_trajectory_history wasn't called or deque isn't full
            # Pad with copies of the first valid entry if needed
            if len(self._trajectory_history) > 0:
                padded_history = list(self._trajectory_history)
                while len(padded_history) < self.trajectory_length:
                    padded_history.insert(0, self._trajectory_history[0]) # Pad with oldest available
                full_trajectory_norm = np.array(padded_history[-self.trajectory_length:], dtype=np.float32)
            else: # Should not happen if reset correctly calls _initialize_trajectory_history
                 raise ValueError("Trajectory history is empty in encode_state.")

            if full_trajectory_norm.shape != (self.trajectory_length, self.feature_dim):
                 raise ValueError(f"Sim Encoded trajectory shape mismatch. Got {full_trajectory_norm.shape}, expected {(self.trajectory_length, self.feature_dim)}")
        return {"basic_state": basic_state_t_plus_1_norm, "full_trajectory": full_trajectory_norm}


# --- VISUALIZATION (adapted from original visualization.py) ---
_agent_trajectory_sim = [] # Module-level storage for trajectory points

def reset_trajectories_sim():
    global _agent_trajectory_sim; _agent_trajectory_sim = []

def visualize_world_sim(world: WorldSim, vis_config: VisualizationSimConfig, fig, ax, show_trajectories: bool = True):
    global _agent_trajectory_sim
    if not world.agent: return # Should not happen if world is initialized
    
    _agent_trajectory_sim.append((world.agent.location.x, world.agent.location.y))
    if len(_agent_trajectory_sim) > vis_config.max_trajectory_points:
        _agent_trajectory_sim = _agent_trajectory_sim[-vis_config.max_trajectory_points:]
    ax.clear()

    if show_trajectories and len(_agent_trajectory_sim) > 1:
        ax.plot(*zip(*_agent_trajectory_sim), 'g-', lw=1.0, alpha=0.6, label='Agent Traj.')
    if vis_config.plot_oil_points and world.true_oil_points:
        ax.scatter(*zip(*[(p.x, p.y) for p in world.true_oil_points]), color='black', marker='.', s=vis_config.point_marker_size, alpha=0.7, label='True Oil')
    if vis_config.plot_water_points and world.true_water_points: # Not usually plotted
        ax.scatter(*zip(*[(p.x, p.y) for p in world.true_water_points]), color='lightblue', marker='.', s=vis_config.point_marker_size, alpha=0.5, label='Water')

    if world.mapper and world.mapper.hull_vertices is not None:
        hull_poly = Polygon(world.mapper.hull_vertices, edgecolor='red', facecolor='red', alpha=0.2, lw=1.5, ls='--', label=f'Est. Hull (Pts In: {world.performance_metric:.2%})')
        ax.add_patch(hull_poly)

    ax.scatter(world.agent.location.x, world.agent.location.y, c='blue', marker='o', s=60, zorder=5, label='Agent')
    heading = world.agent.get_heading()
    ax.arrow(world.agent.location.x, world.agent.location.y, 3.0 * math.cos(heading), 3.0 * math.sin(heading), head_width=1.0, head_length=1.5, fc='blue', ec='blue', alpha=0.7, zorder=5)

    sensor_locs, sensor_reads = world._get_sensor_readings()
    for i, loc in enumerate(sensor_locs):
        color = vis_config.sensor_color_oil if sensor_reads[i] else vis_config.sensor_color_water
        ax.scatter(loc.x, loc.y, c=color, marker='s', s=vis_config.sensor_marker_size, edgecolors='black', lw=0.5, zorder=4, label='Sensors' if i == 0 else "")
        ax.add_patch(Circle((loc.x, loc.y), world.sensor_radius, edgecolor=color, facecolor='none', lw=0.5, ls=':', alpha=0.4, zorder=3))

    wx, wy = world.world_size
    ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
    title1 = f"Step: {world.current_step}, Reward: {world.reward:.3f}"
    if world.current_seed is not None: title1 += f", Seed: {world.current_seed}"
    title2 = f"Metric (Pts In): {world.performance_metric:.3f}"
    ax.set_title(f'Oil Spill Mapping Simulation\n{title1} | {title2}')
    padding = 5.0
    ax.set_xlim(-padding, wx + padding); ax.set_ylim(-padding, wy + padding)
    ax.set_aspect('equal', adjustable='box'); ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(rect=[0, 0, 0.85, 1])


def save_gif_sim(output_filename: str, vis_config: VisualizationSimConfig, vis_output_dir:str, frame_paths: list):
    if not frame_paths: return None
    output_path = os.path.join(vis_output_dir, output_filename)
    frame_duration_sec = vis_config.gif_frame_duration # Already in seconds
    try:
        images = []
        for fp in frame_paths:
             if os.path.exists(fp):
                  try: images.append(imageio.imread(fp))
                  except Exception as e_im: print(f"Error reading frame {fp} for GIF: {e_im}")
        if not images: print("No valid frames for GIF."); return None
        
        # Ensure all images have the same shape by resizing if necessary
        first_shape = images[0].shape
        processed_images = []
        for img in images:
            if img.shape == first_shape:
                processed_images.append(img)
            else: # Resize, assumes PIL.Image.open and then convert to np.array for imageio
                pil_img = Image.fromarray(img)
                pil_img_resized = pil_img.resize((first_shape[1], first_shape[0])) # (width, height)
                processed_images.append(np.array(pil_img_resized))

        imageio.mimsave(output_path, processed_images, duration=frame_duration_sec * 1000) # imageio duration is in ms

        if vis_config.delete_frames_after_gif:
            for fp in frame_paths:
                try:
                    if os.path.exists(fp): os.remove(fp)
                except OSError as e_del: print(f"Could not delete frame {fp}: {e_del}")
        return output_path
    except Exception as e: print(f"Error creating GIF for oil spill: {e}"); return None

# --- MAIN SIMULATION RUNNER ---
def run_simulation_for_streamlit(
    app_sim_config: AppSimConfig,
    model_path: str,
    initial_agent_loc_data: Optional[Tuple[float, float]], # (x,y)
    initial_oil_center_data: Optional[Tuple[float, float]], # (x,y)
    initial_oil_std_dev_data: Optional[float],
    seed_data: Optional[int],
    num_simulation_steps: int,
    visualization_output_dir: str
) -> Tuple[Optional[str], float, float, List[float]]:

    world_cfg_sim = app_sim_config.world # This is WorldSimConfig
    
    # The overrides are now passed to WorldSim constructor
    world = WorldSim(
        world_sim_config=world_cfg_sim,
        initial_agent_loc_override=initial_agent_loc_data,
        initial_oil_center_override=initial_oil_center_data,
        initial_oil_std_dev_override=initial_oil_std_dev_data,
        seed_override=seed_data
    )

    agent: Union[SACAgentEval, PPOAgentEval]
    if app_sim_config.algorithm.lower() == "sac":
        if not app_sim_config.sac: raise ValueError("SACSimConfig missing in AppSimConfig")
        agent = SACAgentEval(app_sim_config.sac, app_sim_config.cuda_device)
    elif app_sim_config.algorithm.lower() == "ppo":
        if not app_sim_config.ppo: raise ValueError("PPOSimConfig missing in AppSimConfig")
        agent = PPOAgentEval(app_sim_config.ppo, app_sim_config.cuda_device)
    else:
        raise ValueError(f"Unknown algorithm for simulation: {app_sim_config.algorithm}")
    
    agent.load_model(model_path)

    reset_trajectories_sim() # Reset global trajectory for visualization
    frame_paths = []
    episode_rewards_raw = [] # Store raw rewards per step
    
    state_dict = world.encode_state() # Initial state
    actor_hidden_state = None

    # Initialize hidden state for RNN if used
    if agent.use_rnn:
        # The get_initial_hidden_state is part of the Eval Actor Network now
        actor_hidden_state = agent.actor.get_initial_hidden_state(1, agent.device)


    # Initial frame visualization
    fig_init, ax_init = plt.subplots(figsize=app_sim_config.visualization.figure_size)
    visualize_world_sim(world, app_sim_config.visualization, fig_init, ax_init)
    initial_frame_filename = f"sim_frame_000_initial.png"
    initial_frame_path = os.path.join(visualization_output_dir, initial_frame_filename)
    os.makedirs(visualization_output_dir, exist_ok=True)
    fig_init.savefig(initial_frame_path); plt.close(fig_init)
    if os.path.exists(initial_frame_path): frame_paths.append(initial_frame_path)


    for step_num in range(num_simulation_steps):
        action_norm: float
        next_actor_hidden_state: Optional[Tuple] = None
        
        if app_sim_config.algorithm.lower() == "sac":
            action_norm, next_actor_hidden_state = agent.select_action(state_dict, actor_hidden_state)
        elif app_sim_config.algorithm.lower() == "ppo":
            # PPO's select_action in eval mode takes the basic_state tuple
            basic_state_for_ppo = state_dict['basic_state']
            action_norm, next_actor_hidden_state = agent.select_action(basic_state_for_ppo, actor_hidden_state)
        else:
            action_norm = 0.0 # Should not happen due to earlier check

        world.step(action_norm, terminal_step=(step_num == num_simulation_steps - 1))
        state_dict = world.encode_state() # Get s_{t+1}
        episode_rewards_raw.append(world.reward) # Store r_{t+1}

        if agent.use_rnn:
            actor_hidden_state = next_actor_hidden_state

        # Visualize current world state (s_{t+1})
        fig_step, ax_step = plt.subplots(figsize=app_sim_config.visualization.figure_size)
        visualize_world_sim(world, app_sim_config.visualization, fig_step, ax_step)
        current_frame_filename = f"sim_frame_{step_num+1:03d}.png"
        current_frame_path = os.path.join(visualization_output_dir, current_frame_filename)
        fig_step.savefig(current_frame_path); plt.close(fig_step)
        if os.path.exists(current_frame_path): frame_paths.append(current_frame_path)


        if world.done:
            print(f"Simulation ended early at step {step_num+1} due to 'done' flag.")
            break
            
    gif_filename = f"simulation_run_oilspill_{time.strftime('%Y%m%d-%H%M%S')}.gif"
    gif_path = save_gif_sim(gif_filename, app_sim_config.visualization, visualization_output_dir, frame_paths)
    
    final_metric = world.performance_metric
    total_raw_reward = sum(episode_rewards_raw) if episode_rewards_raw else 0.0

    return gif_path, final_metric, total_raw_reward, episode_rewards_raw