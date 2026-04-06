"""Non-stationary Brax/MJX environment wrapper.

Ports ESCP's NonstationaryEnv from gym.Wrapper to Brax MJX,
supporting domain randomization via dynamic parameter modification.
Supports: HalfCheetah, Hopper, Walker2d, Ant (MuJoCo v2 equivalents).
"""
import numpy as np
import copy
from typing import List, Dict, Optional, Tuple

# Use standard Gymnasium + MuJoCo for simulation (not Brax pipelines).
# JAX acceleration comes from vectorized training, not env stepping.
import gymnasium as gym


# Map v2 names to modern gymnasium names
ENV_MAP = {
    "HalfCheetah-v2": "HalfCheetah-v5",
    "Hopper-v2": "Hopper-v5",
    "Walker2d-v2": "Walker2d-v5",
    "Ant-v2": "Ant-v5",
    # Allow v4/v5 names directly
    "HalfCheetah-v4": "HalfCheetah-v4",
    "Hopper-v4": "Hopper-v4",
    "Walker2d-v4": "Walker2d-v4",
    "Ant-v4": "Ant-v4",
    "HalfCheetah-v5": "HalfCheetah-v5",
    "Hopper-v5": "Hopper-v5",
    "Walker2d-v5": "Walker2d-v5",
    "Ant-v5": "Ant-v5",
}

RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction',
               'gravity', 'density', 'wind', 'geom_friction_1_dim', 'dof_damping_1_dim']


class NonstationaryEnv:
    """Non-stationary MuJoCo environment with parametric mode switching.

    Ported from ESCP's envs/nonstationary_env.py.
    Supports dynamic modification of MuJoCo model parameters and
    periodic switching between sampled task configurations.
    """

    def __init__(self, env_name: str, rand_params: List[str] = None,
                 log_scale_limit: float = 3.0, seed: int = 0):
        actual_name = ENV_MAP.get(env_name, env_name)
        try:
            self.env = gym.make(actual_name)
        except Exception:
            # Fallback: try original name
            self.env = gym.make(env_name)

        self.rand_params = rand_params or ['gravity']
        self.log_scale_limit = log_scale_limit
        self.rng = np.random.default_rng(seed)

        # Save initial parameters
        self.init_params = {}
        self._save_parameters()
        self.cur_params = copy.deepcopy(self.init_params)

        # Non-stationary switching state
        self.setted_env_params = None
        self.setted_env_changing_period = None
        self.setted_env_changing_interval = None
        self.cur_step_ind = 0
        self.current_task_id = 0

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def obs_dim(self):
        return self.env.observation_space.shape[0]

    @property
    def act_dim(self):
        return self.env.action_space.shape[0]

    def _get_model(self):
        """Get the MuJoCo model from the unwrapped env."""
        return self.env.unwrapped.model

    def _save_parameters(self):
        """Save initial MuJoCo model parameters."""
        model = self._get_model()
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = np.array(model.body_mass).copy()
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = np.array(model.body_inertia).copy()
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = np.array(model.dof_damping).copy()
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = np.array(model.geom_friction).copy()
        if 'geom_friction_1_dim' in self.rand_params:
            self.init_params['geom_friction_1_dim'] = np.array(model.geom_friction).copy()
        if 'dof_damping_1_dim' in self.rand_params:
            self.init_params['dof_damping_1_dim'] = np.array(model.dof_damping).copy()
        if 'gravity' in self.rand_params:
            self.init_params['gravity'] = np.array(model.opt.gravity).copy()
        if 'density' in self.rand_params:
            self.init_params['density'] = np.array([model.opt.density]).copy()

    def sample_tasks(self, n_tasks: int) -> List[Dict]:
        """Generate n_tasks random parameter sets."""
        param_sets = []
        for _ in range(n_tasks):
            new_params = {}
            for param in self.rand_params:
                if param in ('geom_friction_1_dim', 'dof_damping_1_dim'):
                    multiplier = 1.5 ** self.rng.uniform(
                        -self.log_scale_limit, self.log_scale_limit, (1,))
                    new_params[param] = multiplier
                elif param == 'gravity':
                    multiplier = 1.5 ** self.rng.uniform(
                        -self.log_scale_limit, self.log_scale_limit,
                        self.init_params[param].shape)
                    new_params[param] = self.init_params[param] * multiplier
                elif param == 'density':
                    multiplier = 1.5 ** self.rng.uniform(
                        -self.log_scale_limit, self.log_scale_limit, (1,))
                    new_params[param] = self.init_params[param] * multiplier
                elif param == 'wind':
                    new_params[param] = self.rng.uniform(
                        -self.log_scale_limit, self.log_scale_limit, (2,))
                elif param in self.init_params:
                    multiplier = 1.5 ** self.rng.uniform(
                        -self.log_scale_limit, self.log_scale_limit,
                        self.init_params[param].shape)
                    new_params[param] = self.init_params[param] * multiplier
            param_sets.append(new_params)
        return param_sets

    def set_task(self, task: Dict):
        """Apply a task parameter set to the MuJoCo model."""
        model = self._get_model()
        for param, param_val in task.items():
            if param == 'gravity':
                model.opt.gravity[:] = param_val
            elif param == 'density':
                model.opt.density = float(param_val[0])
            elif param == 'wind':
                model.opt.wind[:2] = param_val
            elif param == 'geom_friction_1_dim':
                model.geom_friction[:] = self.init_params['geom_friction_1_dim'] * param_val
            elif param == 'dof_damping_1_dim':
                model.dof_damping[:] = self.init_params['dof_damping_1_dim'] * param_val
            else:
                param_variable = getattr(model, param)
                param_variable[:] = param_val
        self.cur_params = task

    def set_nonstationary_para(self, tasks, changing_period, changing_interval):
        """Setup periodic parameter switching between task configs."""
        self.setted_env_params = tasks
        self.setted_env_changing_period = changing_period
        self.setted_env_changing_interval = changing_interval

    def reset(self, **kwargs):
        self.cur_step_ind = 0
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        self.cur_step_ind += 1
        # Handle periodic parameter switching
        if (self.setted_env_params is not None
                and self.cur_step_ind % self.setted_env_changing_interval == 0):
            weight_origin = self.cur_step_ind / self.setted_env_changing_period
            ind = int(weight_origin) % len(self.setted_env_params)
            self.set_task(self.setted_env_params[ind])
            self.current_task_id = ind

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def get_env_parameter_vector(self):
        """Return flattened current parameter vector (for EP ground truth)."""
        vecs = []
        for key in sorted(self.cur_params.keys()):
            vecs.append(np.array(self.cur_params[key]).ravel())
        if len(vecs) == 0:
            return np.array([])
        return np.concatenate(vecs)

    @property
    def env_parameter_length(self):
        return len(self.get_env_parameter_vector())

    def close(self):
        self.env.close()
