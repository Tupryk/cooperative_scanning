import os
import sys
import cv2
import torch
import numpy as np
import robotic as ry
import matplotlib.pyplot as plt
sys.path.append('../FoundationStereo')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import depth2xyzmap
from core.foundation_stereo import FoundationStereo


def foundation_stereo_call(
        model_path: str="./checkpoints/foundationstereo/pretrained_models/23-51-11/model_best_bp2.pth"
        ) -> tuple[np.ndarray, np.ndarray]:
    
    cfg = OmegaConf.load(f'{os.path.dirname(model_path)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)

    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()
    ### MODEL SETUP END ###

    ### CAMERA INIT START ###
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
    
    scale = .5
    # Zed 2 instrisics
    K = np.array([420.0, 0.0, 336.0, 0.0, 420.0, 188.0, 0.0, 0.0, 1.0]).astype(np.float32).reshape(3,3)
    stereo_width = 0.12
    K[:2] *= scale
    ### CAMERA INIT END ###

    ### IMAGE CAPTURE START ###
    for i in range(100):
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to grab frame")
            exit()
        print(f"BUFFERINGGGGGGGGGGG {i}")
    cap.release()
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    height, width = frame.shape[:2]
    center_x = width // 2

    img0 = frame[:, :center_x]
    img1 = frame[:, center_x:]

    assert scale<=1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()

    print(torch.cuda.memory_summary())
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    ### IMAGE CAPTURE END ###

    print(torch.cuda.memory_summary())
    ### IMAGE FORWARD START ###
    with torch.cuda.amp.autocast(True):
        disp = model.forward(img0, img1, iters=32, test_mode=True)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    ### IMAGE FORWARD END ###

    ### POINT CLOUD START ###
    depth = K[0,0]*stereo_width/disp
    xyz_map = depth2xyzmap(depth, K)
    ### POINT CLOUD END ###

    return xyz_map, img0_ori


def get_point_clouds(C: ry.Config, camera_frames: list[str],
                     on_real: bool=False, verbose: int=0
                     )-> tuple[list[np.ndarray], list[np.ndarray]]:

    pcs = []
    rgbs = []
    sim = ry.Simulation(C, ry.SimulationEngine.physx, verbose=0)

    for i, cam_frame_name in enumerate(camera_frames):
        
        if not on_real:
            sim.addSensor(cam_frame_name)
            rgb, depth = sim.getImageAndDepth()
            if verbose >= 2:
                plt.imsave(f"{cam_frame_name}.png", rgb)

            fxycx = [869.1169166564941, 869.1169166564941, 320.0, 180.0]  # Same as in BotOp
            pc = sim.depthData2pointCloud(depth, fxycx)
            pc = pc.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            non_zero_indices = np.where(~np.all(pc == 0, axis=1))[0]
            pc = pc[non_zero_indices]
            rgb = rgb[non_zero_indices] / 255
        
        else:
            # TODO: Object oriented stuff to avoid loading the model each time
            pc, rgb = foundation_stereo_call()
            pcs.append(pc)
            rgbs.append(rgb)
            break

        pcs.append(pc)
        rgbs.append(rgb)

        if verbose > 0:
            pc_frame = C.addFrame(f"{cam_frame_name}_pointCloud", cam_frame_name)

            if verbose == 3:
                pc_frame.setPointCloud(pc, rgb)
            else:
                pc_frame.setPointCloud(pc)
                if i%2:
                    pc_frame.setColor([1, 0, 0])
                else:
                    pc_frame.setColor([0, 0, 1])

    if verbose > 0:
        C.view(True)
    
    return pcs, rgbs
