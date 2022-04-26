import torch
import robosuite
from robosuite import load_controller_config
import numpy as np
import copy
import time
import argparse
import os
from robosuite.wrappers.gym_wrapper import GymWrapper

skill_config = dict(
                skills=['atomic', 'open', 'reach_osc', 'grasp', 'push'],
                aff_penalty_fac=15.0,

                base_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.95]
                    ],
                    lift_height=0.95,
                    binary_gripper=True,
                    ignore_aff_rew = True,
                    aff_threshold=0.06,
                    aff_type='dense',
                    aff_tanh_scaling=10.0,
                ),
                atomic_config=dict(
                    use_ori_params=True,
                ),
                reach_config=dict(
                    use_gripper_params=False,
                    local_xyz_scale=[0.0, 0.0, 0.06],
                    use_ori_params=False,
                    max_ac_calls=25,
                    num_reach_steps=10
                ),
                grasp_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    aff_threshold=0.03,

                    local_xyz_scale=[0.0, 0.0, 0.0],
                    use_ori_params=True,
                    max_ac_calls=20,
                    num_reach_steps=10,
                    num_grasp_steps=10,
                ),
                push_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    delta_xyz_scale=[0.25, 0.25, 0.05],

                    max_ac_calls=20,
                    use_ori_params=True,

                    aff_threshold=[0.12, 0.12, 0.04],
                ),
            )      


def rollout(
        env,
        image_obs_in_info=False,
     
):

    o = env.reset()
    env.viewer.set_camera(camera_id=0)
    env.render()
   

    # # Resetting state capability
    # mj_state = env.get_mj_state()
    # env.sim.set_state(mj_state)
    # env.sim.forward()
    # env.render()
    a = 'reach_osc'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info,specific_skill='reach-bread')
    
 
    a = 'grasp'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info,specific_skill='grasp-cube_bread')

  
    a = 'reach_osc'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info, specific_skill='reach-pot')

    

    step_basic(env, skill = 'release-cube_bread')
    env.render()

    a = 'grasp'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info, specific_skill='grasp-PotObject')

    a = 'reach_osc'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info,specific_skill='reach-stove')
    

    step_basic(env, skill = 'release-PotObject')

    a = 'reach_osc'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info,specific_skill='reach-button')

    step_basic(env, skill = 'slide-button')
    env.render()

   
    a = 'reach_osc'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info,specific_skill='reach-cabinet')
    
    a = 'grasp'
    next_o, r, d, env_info, ob_dict = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info,specific_skill='grasp-cabinet')

   
    step_basic(env, skill = 'slide-cabinet')
    
  
def step_basic(env, skill):    
    env.skill_controller._pos_is_delta = True
    env.skill_controller._ori_is_delta = True
    
    info = env._get_skill_info(skill)
    if 'release' in skill:
        for i in range(5):
            obs, reward, done, env_info, ob_dict = env.step_basic(info['delta1'])
            env.render()
        for i in range(20):
            obs, reward, done, env_info, ob_dict= env.step_basic(info['delta2'])
            env.render()
        
    
    elif 'slide' in skill:
        for i in range(20):
            obs, reward, done, env_info, ob_dict = env.step_basic(info['delta'])
            env.render()
    
    else:
        env_info = {'skill_complete': False}
        return None, 0., False, env_info, None

    env_info['skill_complete'] = True

    return obs, reward, done, env_info, ob_dict
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    robots = "Panda"

    options = {}
    options["controller_configs"] = robosuite.load_controller_config(
        default_controller="OSC_POSITION_YAW")
    options["skill_config"] = skill_config
    
    
    env = robosuite.environments.manipulation.multitask_kitchen.MultitaskKitchenDomain(
            robots,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            hard_reset = False,
            #initialization_noise=None,

            **options
        )
    obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'object-state']
    env = GymWrapper(env, keys=obs_keys)
    rollout(env)
    