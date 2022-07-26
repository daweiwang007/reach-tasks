import numpy as np
import gym
from gym.spaces import Box as SamplingSpace
from alr_sim.core import Scene
from alr_sim.gyms.gym_utils.helpers import obj_distance
from alr_sim.sims.universal_sim.PrimitiveObjects import Sphere


_DEFAULT_VALUE_AT_MARGIN = 0.1

class ReachEnv(gym.Env):
    def __init__(
            self,
            scene: Scene,
            controller,
            n_substeps: int = 25,
            max_steps_per_episode: int = 2e3,
            debug: bool = False,
            random_env: bool = False,
    ):

        self.scene = scene
        self.controller = controller
        self.n_substeps = n_substeps
        self.max_steps_per_episode = max_steps_per_episode
        self.debug = debug
        self.random_env = random_env

        self.robot = scene.robots[0]
        self.env_step_counter = 0

        self.init_robot_c_pos = np.array([5.50899712e-01, -1.03382391e-08,  6.99822168e-01])

        self.episode = 0
        self.terminated = False

        self.goal = Sphere(
            name="goal",
            size=[0.01],
            init_pos=[0.5, 0, 0],
            init_quat=[1, 0, 0, 0],
            rgba=[1, 0, 0, 1],
            static=True,
        )
        self.goal_space = SamplingSpace(
            low=np.array([0.2, -0.3, 0.1]), high=np.array([0.5, 0.3, 0.5])
        )
        self.scene.add_object(self.goal)

        self.target_min_dist = 0.02

    def step(self, action):

        self.controller.set_action(action)
        self.controller.execute_action(n_time_steps=self.n_substeps)

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_finished()

        debug_info = {}
        if self.debug:
            debug_info = self.debug_msg()

        self.env_step_counter += 1
        return observation, reward, done, debug_info

    def get_observation(self) -> np.ndarray:
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, rel_goal_tcp_pos = obj_distance(goal_pos, tcp_pos)

        env_state = np.concatenate([goal_pos, [dist_tcp_goal], rel_goal_tcp_pos])
        robot_state = self.robot_state()
        return np.concatenate([robot_state, env_state])

    def get_reward(self):
        _TARGET_RADIUS = 0.05
        tcp = self.robot.current_c_pos
        target = self.scene.get_obj_pos(self.goal)

        tcp_to_target = np.linalg.norm(tcp - target)

        in_place_margin = (np.linalg.norm(self.init_robot_c_pos - target))
        in_place = self.tolerance(tcp_to_target,
                                  bounds=(0, _TARGET_RADIUS),
                                  margin=in_place_margin,
                                  sigmoid='long_tail', )
        reward = 10 * in_place
        return reward

    def start(self):
        self.scene.start()

    def _reset_env(self):
        if self.random_env:
            new_goal = [self.goal, self.goal_space.sample()]
            self.scene.reset([new_goal])
        else:
            self.scene.reset()

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        goal_pos = self.scene.get_obj_pos(self.goal)
        tcp_pos = self.robot.current_c_pos
        dist_tcp_goal, _ = obj_distance(goal_pos, tcp_pos)

        if dist_tcp_goal <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
            return True
        return False

    def is_finished(self):
        if (
                self.terminated
                or self._check_early_termination()
                or self.env_step_counter >= self.max_steps_per_episode - 1
        ):
            return True
        return False

    def robot_state(self):
        # Update Robot State
        self.robot.receiveState()

        # joint state
        joint_pos = self.robot.current_j_pos
        joint_vel = self.robot.current_j_vel

        # gripper state
        gripper_vel = self.robot.current_fing_vel
        gripper_width = [self.robot.gripper_width]

        # end effector state
        tcp_pos = self.robot.current_c_pos
        tcp_vel = self.robot.current_c_vel
        tcp_quad = self.robot.current_c_quat

        return np.concatenate(
            [
                joint_pos,
                joint_vel,
                gripper_vel,
                gripper_width,
                tcp_pos,
                tcp_vel,
                tcp_quad,
            ]
        )

    def _sigmoids(self, x, value_at_1, sigmoid):
        """Returns 1 when `x` == 0, between 0 and 1 otherwise.
        Args:
            x: A scalar or numpy array.
            value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
            sigmoid: String, choice of sigmoid type.
        Returns:
            A numpy array with values between 0.0 and 1.0.
        Raises:
            ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
            `quadratic` sigmoids which allow `value_at_1` == 0.
            ValueError: If `sigmoid` is of an unknown type.
        """
        if sigmoid in ('cosine', 'linear', 'quadratic'):
            if not 0 <= value_at_1 < 1:
                raise ValueError(
                    '`value_at_1` must be nonnegative and smaller than 1, '
                    'got {}.'.format(value_at_1))
        else:
            if not 0 < value_at_1 < 1:
                raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                                 'got {}.'.format(value_at_1))

        if sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_1))
            return np.exp(-0.5 * (x * scale) ** 2)

        elif sigmoid == 'hyperbolic':
            scale = np.arccosh(1 / value_at_1)
            return 1 / np.cosh(x * scale)

        elif sigmoid == 'long_tail':
            scale = np.sqrt(1 / value_at_1 - 1)
            return 1 / ((x * scale) ** 2 + 1)

        elif sigmoid == 'reciprocal':
            scale = 1 / value_at_1 - 1
            return 1 / (abs(x) * scale + 1)

        elif sigmoid == 'cosine':
            scale = np.arccos(2 * value_at_1 - 1) / np.pi
            scaled_x = x * scale
            return np.where(
                abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

        elif sigmoid == 'linear':
            scale = 1 - value_at_1
            scaled_x = x * scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

        elif sigmoid == 'quadratic':
            scale = np.sqrt(1 - value_at_1)
            scaled_x = x * scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)

        elif sigmoid == 'tanh_squared':
            scale = np.arctanh(np.sqrt(1 - value_at_1))
            return 1 - np.tanh(x * scale) ** 2

        else:
            raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

    def tolerance(self, x,
                  bounds=(0.0, 0.0),
                  margin=0.0,
                  sigmoid='gaussian',
                  value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
        """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
        Args:
            x: A scalar or numpy array.
            bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
            the target interval. These can be infinite if the interval is unbounded
            at one or both ends, or they can be equal to one another if the target
            value is exact.
            margin: Float. Parameter that controls how steeply the output decreases as
            `x` moves out-of-bounds.
            * If `margin == 0` then the output will be 0 for all values of `x`
                outside of `bounds`.
            * If `margin > 0` then the output will decrease sigmoidally with
                increasing distance from the nearest bound.
            sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
            'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
            value_at_margin: A float between 0 and 1 specifying the output value when
            the distance from `x` to the nearest bound is equal to `margin`. Ignored
            if `margin == 0`.
        Returns:
            A float or numpy array with values between 0.0 and 1.0.
        Raises:
            ValueError: If `bounds[0] > bounds[1]`.
            ValueError: If `margin` is negative.
        """
        lower, upper = bounds
        if lower > upper:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin < 0:
            raise ValueError('`margin` must be non-negative. Current value: {}'.format(margin))

        in_bounds = np.logical_and(lower <= x, x <= upper)
        if margin == 0:
            value = np.where(in_bounds, 1.0, 0.0)
        else:
            d = np.where(x < lower, lower - x, x - upper) / margin
            value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin,
                                                       sigmoid))

        return float(value) if np.isscalar(x) else value
