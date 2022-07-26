import time

import numpy as np

from alr_sim.gyms.gym_controllers import GymCartesianVelController, GymTorqueController
from alr_sim.controllers import Controller
from alr_sim.sims.SimFactory import SimRepository
#from envs.reach_env.reach import ReachEnv
from meta_world.reach_env import ReachEnv

from alr_sim.core.logger import RobotPlotFlags

import random


if __name__ == "__main__":

    simulator = "mujoco"

    sim_factory = SimRepository.get_factory(simulator)

    scene = sim_factory.create_scene()
    robot = sim_factory.create_robot(scene)
    ctrl = GymTorqueController(robot)
    robot.cartesianPosQuatTrackingController.neglect_dynamics = False
    env = ReachEnv(scene=scene, n_substeps=500, controller=ctrl, random_env=True)

    env.start()

    env.seed(10)
    scene.start_logging()
    for i in range(20):
        print('iNIT POS', robot.current_c_pos)
        action = ctrl.action_space().sample()
        print(action)
        env.step(action)
        if (i + 1) % 100 == 0:
            env.reset()

    scene.stop_logging()
