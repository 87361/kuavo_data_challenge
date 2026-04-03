import cv2
from torchvision.transforms.functional import to_tensor
from kuavo_data.CvtRosbag2Lerobot import load_raw_episode_data
from client import PolicyClient
import numpy as np
import torch,os,shutil
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.policies.factory import make_pre_post_processors

from pathlib import Path
import kuavo_data.common.kuavo_dataset as kuavo

def read_and_process_episode_data(ep_path):
    def init_param():
        #Manually load configuration
        from omegaconf import OmegaConf
        config_path = Path(__file__).parents[2] / "configs" / "data" / "KuavoRosbag2Lerobot.yaml"
        cfg = OmegaConf.load(config_path)
        
        global DEFAULT_JOINT_NAMES_LIST
        kuavo.init_parameters(cfg)

        n = cfg.rosbag.num_used
        raw_dir = cfg.rosbag.rosbag_dir
        version = cfg.rosbag.lerobot_dir

        task_name = os.path.basename(raw_dir)
        repo_id = f'lerobot/{task_name}'
        lerobot_dir = os.path.join(raw_dir,"../",version,"lerobot")
        if os.path.exists(lerobot_dir):
            shutil.rmtree(lerobot_dir)
        
        half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
        half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
        half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
        UP_START_INDEX = 12
        if kuavo.ONLY_HALF_UP_BODY:
            if kuavo.USE_LEJU_CLAW:
                DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                        + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
                arm_slice = [
                    (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_CLAW[0][0] + half_arm, kuavo.SLICE_CLAW[0][-1] + half_arm), 
                    (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_claw, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_claw), (kuavo.SLICE_CLAW[1][0] + half_arm * 2, kuavo.SLICE_CLAW[1][-1] + half_arm * 2)
                    ]
            elif kuavo.USE_QIANGNAO:  
                DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                        + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]               
                arm_slice = [
                    (kuavo.SLICE_ROBOT[0][0] - UP_START_INDEX, kuavo.SLICE_ROBOT[0][-1] - UP_START_INDEX),(kuavo.SLICE_DEX[0][0] + half_arm, kuavo.SLICE_DEX[0][-1] + half_arm), 
                    (kuavo.SLICE_ROBOT[1][0] - UP_START_INDEX + half_dexhand, kuavo.SLICE_ROBOT[1][-1] - UP_START_INDEX + half_dexhand), (kuavo.SLICE_DEX[1][0] + half_arm * 2, kuavo.SLICE_DEX[1][-1] + half_arm * 2)
                    ]
            DEFAULT_JOINT_NAMES_LIST = [DEFAULT_ARM_JOINT_NAMES[k] for l, r in arm_slice for k in range(l, r)]  
        else:
            if kuavo.USE_LEJU_CLAW:
                DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw] \
                                        + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
            elif kuavo.USE_QIANGNAO:
                DEFAULT_ARM_JOINT_NAMES = kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand] \
                                        + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:] + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]             
            DEFAULT_JOINT_NAMES_LIST = kuavo.DEFAULT_LEG_JOINT_NAMES + DEFAULT_ARM_JOINT_NAMES + kuavo.DEFAULT_HEAD_JOINT_NAMES

    init_param()
    imgs_per_cam, state, action, velocity, effort ,claw_state, claw_action,qiangnao_state,qiangnao_action, rq2f85_state, rq2f85_action = load_raw_episode_data(ep_path)
    if len(claw_action)==0 and len(qiangnao_action) == 0:
        claw_action = rq2f85_action
        claw_state = rq2f85_state

    #Binarize the hands
    if kuavo.IS_BINARY:
        qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
        qiangnao_action = np.where(qiangnao_action > 50, 1, 0)
        claw_state = np.where(claw_state > 50, 1, 0)
        claw_action = np.where(claw_action > 50, 1, 0)
        rq2f85_state = np.where(rq2f85_state > 0.4, 1, 0)
        rq2f85_action = np.where(rq2f85_action > 70, 1, 0)
    else:
        #Perform data normalization processing
        claw_state = claw_state / 100
        claw_action = claw_action / 100
        qiangnao_state = qiangnao_state / 100
        qiangnao_action = qiangnao_action / 100
        rq2f85_state = rq2f85_state / 0.8
        rq2f85_action = rq2f85_action / 140
    ########################
    #delta processing
    ########################
    # =====================
    #To solve the zero point problem, subtract each frame from the first frame
    if kuavo.RELATIVE_START:
        #Each state, action is subtracted from their first frame
        state = state - state[0]
        action = action - action[0]
        
    #===Only handle delta actions
    if kuavo.DELTA_ACTION:
        # delta_action = action[1:] - state[:-1]
        # trim = lambda x: x[1:] if (x is not None) and (len(x) > 0) else x
        # state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action = \
        #     map(
        #         trim, 
        #         [state, action, velocity, effort, claw_state, claw_action, qiangnao_state, qiangnao_action]
        #         )
        # for camera, img_array in imgs_per_cam.items():
        #     imgs_per_cam[camera] = img_array[1:]
        # action = delta_action

        # delta_action = np.concatenate(([action[0]-state[0]], action[1:] - action[:-1]), axis=0)
        # action = delta_action

        delta_action = action-state
        action = delta_action
    
    num_frames = state.shape[0]
    frames = []
    for i in range(num_frames):
        if kuavo.ONLY_HALF_UP_BODY:
            if kuavo.USE_LEJU_CLAW:
                #Use lejuclaw to convert upper body joint data
                if kuavo.CONTROL_HAND_SIDE == "left" or kuavo.CONTROL_HAND_SIDE == "both":
                    output_state = state[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[0][0]:kuavo.SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                    output_action = action[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[0][0]:kuavo.SLICE_CLAW[0][-1]].astype(np.float32)), axis=0)
                if kuavo.CONTROL_HAND_SIDE == "right" or kuavo.CONTROL_HAND_SIDE == "both":
                    if kuavo.CONTROL_HAND_SIDE == "both":
                        output_state = np.concatenate((output_state, state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                        output_action = np.concatenate((output_action, action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                    else:
                        output_state = state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_state = np.concatenate((output_state, claw_state[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)
                        output_action = action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_action = np.concatenate((output_action, claw_action[i, kuavo.SLICE_CLAW[1][0]:kuavo.SLICE_CLAW[1][-1]].astype(np.float32)), axis=0)

            elif kuavo.USE_QIANGNAO:
                #Type: kuavo_sdk/robotHandPosition
                #left_hand_position (list of float): left hand position, contains 6 elements, the value range of each element is [0, 100], 0 is open, 100 is closed.
                #right_hand_position (list of float): Right hand position, including 6 elements, the value range of each element is [0, 100], 0 is open, 100 is closed.
                #The length of the data structure constructing the output_state of the qiangnao type should be 26
                if kuavo.CONTROL_HAND_SIDE == "left" or kuavo.CONTROL_HAND_SIDE == "both":
                    output_state = state[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[0][0]:kuavo.SLICE_DEX[0][-1]].astype(np.float32)), axis=0)

                    output_action = action[i, kuavo.SLICE_ROBOT[0][0]:kuavo.SLICE_ROBOT[0][-1]]
                    output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[0][0]:kuavo.SLICE_DEX[0][-1]].astype(np.float32)), axis=0)
                if kuavo.CONTROL_HAND_SIDE == "right" or kuavo.CONTROL_HAND_SIDE == "both":
                    if kuavo.CONTROL_HAND_SIDE == "both":
                        output_state = np.concatenate((output_state, state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                        output_action = np.concatenate((output_action, action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]), axis=0)
                        output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                    else:
                        output_state = state[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_state = np.concatenate((output_state, qiangnao_state[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                        output_action = action[i, kuavo.SLICE_ROBOT[1][0]:kuavo.SLICE_ROBOT[1][-1]]
                        output_action = np.concatenate((output_action, qiangnao_action[i, kuavo.SLICE_DEX[1][0]:kuavo.SLICE_DEX[1][-1]].astype(np.float32)), axis=0)
                # output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)
        else:
            if kuavo.USE_LEJU_CLAW:
                #Using lejuclaw for whole body joint data conversion
                #The original data is the data of 28 joints, and the length of the original state and action data is 28
                #Data order:
                #The first 12 data are lower limb motor data:
                #0~5 is left lower limb data (l_leg_roll, l_leg_yaw, l_leg_pitch, l_knee, l_foot_pitch, l_foot_roll)
                #6~11 is the right lower limb data (r_leg_roll, r_leg_yaw, r_leg_pitch, r_knee, r_foot_pitch, r_foot_roll)
                #The next 14 data are arm motor data:
                #12~18 Left arm motor data ("l_arm_pitch", "l_arm_roll", "l_arm_yaw", "l_forearm_pitch", "l_hand_yaw", "l_hand_pitch", "l_hand_roll")
                #19~25 is the right arm motor data ("r_arm_pitch", "r_arm_roll", "r_arm_yaw", "r_forearm_pitch", "r_hand_yaw", "r_hand_pitch", "r_hand_roll")
                #The last 2 are head motor data: head_yaw and head_pitch
                
                #TODO: Construct target slice
                output_state = state[i, 0:19]
                output_state = np.insert(output_state, 19, claw_state[i, 0].astype(np.float32))
                output_state = np.concatenate((output_state, state[i, 19:26]), axis=0)
                output_state = np.insert(output_state, 19, claw_state[i, 1].astype(np.float32))
                output_state = np.concatenate((output_state, state[i, 26:28]), axis=0)

                output_action = action[i, 0:19]
                output_action = np.insert(output_action, 19, claw_action[i, 0].astype(np.float32))
                output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                output_action = np.insert(output_action, 19, claw_action[i, 1].astype(np.float32))
                output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)

            elif kuavo.USE_QIANGNAO:
                output_state = state[i, 0:19]
                output_state = np.concatenate((output_state, qiangnao_state[i, 0:6].astype(np.float32)), axis=0)
                output_state = np.concatenate((output_state, state[i, 19:26]), axis=0)
                output_state = np.concatenate((output_state, qiangnao_state[i, 6:12].astype(np.float32)), axis=0)
                output_state = np.concatenate((output_state, state[i, 26:28]), axis=0)

                output_action = action[i, 0:19]
                output_action = np.concatenate((output_action, qiangnao_action[i, 0:6].astype(np.float32)),axis=0)
                output_action = np.concatenate((output_action, action[i, 19:26]), axis=0)
                output_action = np.concatenate((output_action, qiangnao_action[i, 6:12].astype(np.float32)), axis=0)
                output_action = np.concatenate((output_action, action[i, 26:28]), axis=0)  
        frame = {
            "observation.state": torch.from_numpy(output_state).type(torch.float32),
            "action": torch.from_numpy(output_action).type(torch.float32),
        }
        
        for camera, img_array in imgs_per_cam.items():
            if "depth" in camera:
                # frame[f"observation.{camera}"] = img_array[i]
                min_depth, max_dpeth = kuavo.DEPTH_RANGE[0], kuavo.DEPTH_RANGE[1]
                frame[f"observation.{camera}"] = np.clip(img_array[i], min_depth, max_dpeth)
                print("[info]: Clip depth in range %d ~ %d"%(min_depth, max_dpeth))
            else:
                frame[f"observation.images.{camera}"] = img_array[i]
        
        if velocity is not None:
            frame["observation.velocity"] = velocity[i]
        if effort is not None:
            frame["observation.effort"] = effort[i]
        frames.append(frame)
    
    return frames

def img_preprocess(image, device="cpu"):
    return to_tensor(image).unsqueeze(0).to(device, non_blocking=True)

def depth_preprocess(depth, device="cpu",depth_range=[0,1000]):
    depth_uint16 =  torch.tensor(depth,dtype=torch.float32).clamp(*depth_range).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    max_depth = depth_uint16.max()
    min_depth = depth_uint16.min()
    depth_normalized = (depth_uint16 - min_depth) / (max_depth - min_depth + 1e-9)  #normalized to [0, 1]
    # depth_normalized = (depth_normalized * 255).astype(np.uint8)
    return depth_normalized
    
# Convert raw observations into the observations required by the model
def hardware_obses_to_policy_obs_dict(obs):
    device = 'cuda'
    obs_dict = {}
    for k,v in obs.items():
        if "images" in k:
            obs_dict[k] = img_preprocess(v, device=device)
        elif "state" in k:
            obs_dict[k] = torch.tensor(v,dtype=torch.float32).unsqueeze(0).to(device, non_blocking=True)
            print(obs_dict[k].shape)
        elif "depth" in k:
            obs_dict[k] = depth_preprocess(v, device=device, depth_range=[0,1500])
    return obs_dict

def main(ep_path="/home/ubun-new/go_bag/bag_for_handover/A10-A01-206-208-92-71-dex_hand-20250930100830-v1.bag"):
    policy_client = PolicyClient()
    frames = read_and_process_episode_data(ep_path) #Load a piece of remote operation data, each step is a dictionary including observation.state, observation.images.head_cam_h, observation.images.wrist_cam_l, observation.images.wrist_cam_r, action
    preprocessor, postprocessor = make_pre_post_processors(None,"outputs/train/test_handover/state_fuse/run_1008/epochbest")
    for i in range(len(frames)):
    	#Process the observations of the real machine into a dictionary
        obs_dict = hardware_obses_to_policy_obs_dict(frames[i])
        # print("head img",obs_dict["observation.depth_h"].max())

        obs_dict = preprocessor(obs_dict)

        print("head img",obs_dict.keys())
        print("head img",obs_dict["observation.images.head_cam_h"].min(), obs_dict["observation.images.head_cam_h"].max())
        print("head img",obs_dict["observation.depth_h"].min(), obs_dict["observation.depth_h"].max())
        # raise ValueError()

        action_pred = policy_client.select_action(obs_dict) #The numpy(26,) dimensional action is composed of 7-dimensional left arm + 6-dimensional left arm + 7-dimensional right arm + 6-dimensional right hand according to the method of kuavo to lerobot. The movement of the arm is the target joint angle. The movement of the dexterous hand is between [0,1] and needs to be multiplied by 100 before sending it to the real machine.
        action_pred = postprocessor(action_pred).squeeze(0).cpu().numpy()
        action_groundtruth = frames[i]["action"].cpu().numpy()
        print("Timestep:", i, "Action MSE:", ((action_pred - action_groundtruth)**2).mean())
        # print("action_pred:",action_pred)
        # print("action_groundtruth:",action_groundtruth)

if __name__ == "__main__":
    main()
