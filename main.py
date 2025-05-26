import torch
import robotic as ry

from scenes import get_config
from motions import move_to_look
from vision import get_point_clouds
from grasping import sample_best_grasp, get_graspnet

print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

ON_REAL = True
camera_frames = ["l_cameraWrist", "r_cameraWrist"]

C = get_config()

obj_frame_name = C.getFrameNames()[-1]
path = move_to_look(C, obj_frame_name)
C.setJointState(path[-1])

pcs, rgbs = get_point_clouds(C, camera_frames, on_real=ON_REAL, verbose=1)

print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

graspnet = get_graspnet()
grasp_pos = sample_best_grasp(pcs[0], graspnet, rgbs[0], verbose=1)
