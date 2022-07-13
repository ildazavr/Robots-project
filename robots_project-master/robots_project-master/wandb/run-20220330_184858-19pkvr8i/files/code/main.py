import numpy as np
from controller import Controller


from env_Drones.env_Drones import EnvDrones
import wandb
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


if __name__ == '__main__':
    env = EnvDrones(50, 4, 10, 30, 5)
    env.rand_reset_drone_pos()
    controller = Controller()

    # max_MC_iter = 100
    # fig = plt.figure()
    # gs = GridSpec(1, 2, figure=fig)
    # ax1 = fig.add_subplot(gs[0:1, 0:1])
    # ax2 = fig.add_subplot(gs[0:1, 1:2])
    #
    # # ax1.imshow(env.get_joint_obs())
    # # ax2.imshow(env.get_drone_obs(env.drone_list[0]))
    # # print(env.get_drone_obs(env.drone_list[0]).shape)
    # # print(env.get_joint_obs().shape)
    # plt.show()

    wandb.init(project='robots_proj', group='big_model', job_type='test')
    controller.train(env)

    # human_act_traj = [[], [], [], [], []]
    # for MC_iter in range(10):
    #     ax1.imshow(env.get_full_obs())
    #     ax2.imshow(env.get_joint_obs())
    #
    #     drone_act_list = []
    #     for i in range(env.drone_num):
    #         drone_act_list.append(random.randint(0, 4))
    #
    #     obs, rew, done = env.step(drone_act_list)
    #
    #     print(obs.shape)
    #
    #     print(rew)
    #     plt.pause(1)
    #     plt.show()
