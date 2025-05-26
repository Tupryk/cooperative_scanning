import os
import sys
import torch
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup

GRASPNET_DIR = "../graspnet-baseline"
sys.path.append(os.path.join(GRASPNET_DIR, 'models'))
sys.path.append(os.path.join(GRASPNET_DIR, 'dataset'))
sys.path.append(os.path.join(GRASPNET_DIR, 'utils'))

from graspnet import GraspNet, pred_decode


def get_graspnet(model_path: str="./checkpoints/graspnet/checkpoint-rs.tar") -> GraspNet:

    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    return net

def process_point_cloud(point_cloud: np.ndarray,
                        rgb: np.ndarray=np.array([])) -> tuple[dict, o3d.geometry.PointCloud]:
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
    if len(rgb):
        print(rgb)
        cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))

    end_points = dict()
    point_cloud = torch.from_numpy(point_cloud[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_cloud = point_cloud.to(device)
    end_points['point_clouds'] = point_cloud
    end_points['cloud_colors'] = rgb

    return end_points, cloud

def get_grasps(net: GraspNet, end_points: dict) -> GraspGroup:
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def sample_best_grasp(
        point_cloud: np.ndarray,
        graspnet: GraspNet,
        rgb: np.ndarray=np.array([]),
        verbose: int=0) -> np.ndarray:
    
    #TODO: should maybe be a method in Graspnet
    
    end_points, cloud = process_point_cloud(point_cloud, rgb)
    grasps = get_grasps(graspnet, end_points)
    
    if verbose:
        vis_grasps(grasps, cloud)
    
    pose = np.zeros(7)
    
    return pose
