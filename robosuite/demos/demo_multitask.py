import numpy as np

import robosuite
import h5py
import argparse
import os

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
                    max_ac_calls=15,
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
                    num_reach_steps=2,
                    num_grasp_steps=3,
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
def visitor_func(name, obj):
    print(name)
    for key, val in obj.attrs.iteritems():
        print(f'{key}:{val}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    )
    parser.add_argument(
        "--use-actions",
        action='store_true',
    )

    args = parser.parse_args()

    robots = "Panda"

    options = {}
    options["controller_configs"] = robosuite.load_controller_config(
        default_controller="OSC_POSITION")
    options["skill_config"] = skill_config
    args.folder = 'multi'
    
    if 'multi' in args.folder:
        env = robosuite.environments.manipulation.multitask_kitchen.MultitaskKitchenDomain(
            robots,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,

            **options
        )
        # env = robosuite.environments.manipulation.Lift(
        #     robots,
        #     has_renderer=True,
        #     has_offscreen_renderer=False,
        #     use_camera_obs=False,
        #     control_freq=20,

        #     **options
        # )
        demo_path = '/home/nuria/phd/BUDS/datasets/MultitaskKitchenDomain_training_set'
    elif 'single' in args.folder:
        env = robosuite.environments.manipulation.KitchenEnv(
            robots,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
            **options
        )
        demo_path = '/home/nuria/phd/BUDS/datasets/KitchenEnv_training_set'
    # reset env
    env.reset()
    env.viewer.set_camera(camera_id=0)
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")

    for num_ep in range(10):

        env.reset()
        actions = np.array(f["data/ep_{}/actions".format(num_ep)][()])
        states = f["data/ep_{}/gt_states".format(num_ep)][()]
        # get action limits
        low, high = env.action_spec

        if args.use_actions:
            for j, action in enumerate(actions):
                obs, reward, done, info = env.step(action)
                env.render()

        else:
            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()
