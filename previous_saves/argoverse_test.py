import argparse
import sys
import os
import time
import warnings
import json
import math
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import open3d as o3d
from plyfile import PlyData

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
from utils.visualization_utils import merge_rgb_to_bev

# Argoverse-specific configuration
class ArgoverseConfig:
    def __init__(self):
        # BEV parameters
        self.BEV_WIDTH = 608  # pixels
        self.BEV_HEIGHT = 608  # pixels
        self.DISCRETIZATION = 0.1  # meters per pixel
        
        # Boundary for point cloud filtering
        self.boundary = {
            "minX": -40.0,
            "maxX": 40.0,
            "minY": -40.0,
            "maxY": 40.0,
            "minZ": -3.0,
            "maxZ": 1.0
        }
        
        # Colors for visualization
        self.colors = {
            "Car": (0, 255, 0),        # Green
            "Pedestrian": (255, 0, 0), # Blue
            "Cyclist": (0, 0, 255)     # Red
        }

cnf = ArgoverseConfig()

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for Argoverse Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                       help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                       help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                       default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                       help='the path of the pretrained checkpoint')
    parser.add_argument('--K', type=int, default=50,
                       help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                       help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                       help='GPU index to use.')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='mini-batch size (default: 4)')
    parser.add_argument('--peak_thresh', type=float, default=0.02)
    parser.add_argument('--save_test_output', action='store_true',
                       help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                       help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                       help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=608,
                       help='the width of showing output, the height maybe vary')
    
    # Argoverse specific arguments
    parser.add_argument('--lidar_dir', type=str, 
                       default='D:/NEW_DOWNLOADS/av_1.0_sample_dataset/av_1.0_sample_dataset/argoverse-tracking/sample/samplefile/lidar',
                       help='Directory containing Argoverse LiDAR data')
    parser.add_argument('--image_dir', type=str, 
                       default='D:/NEW_DOWNLOADS/av_1.0_sample_dataset/av_1.0_sample_dataset/argoverse-tracking/sample/samplefile/ring_front_center',
                       help='Directory containing Argoverse camera images')
    parser.add_argument('--calib_file', type=str, 
                       default='D:/NEW_DOWNLOADS/av_1.0_sample_dataset/av_1.0_sample_dataset/argoverse-tracking/sample/samplefile/vehicle_calibration_info.json',
                       help='Path to Argoverse calibration file')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    
    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs

class ArgoverseDataLoader:
    def __init__(self, lidar_dir, image_dir, calib_file):
        self.lidar_dir = Path(lidar_dir)
        self.image_dir = Path(image_dir)
        self.calib_file = calib_file
        
        # Load calibration data
        with open(calib_file, 'r') as f:
            self.calib_data = json.load(f)
        
        # Get sorted list of files
        self.lidar_files = sorted(self.lidar_dir.glob('*.ply'))
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        
        # Ensure we have matching pairs
        #assert len(self.lidar_files) == len(self.image_files), "Mismatch between LiDAR and image files"
        self.num_samples = len(self.lidar_files)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load LiDAR data
        lidar_path = self.lidar_files[idx]
        points = self.load_lidar_data(lidar_path)
        
        # Create BEV map
        bev_map = makeBVFeature(points, cnf.DISCRETIZATION, cnf.boundary)
        
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Metadata
        metadata = {
            'img_path': str(img_path),
            'lidar_path': str(lidar_path),
            'frame_id': lidar_path.stem
        }
        
        return metadata, bev_map, img_rgb
    
    def load_lidar_data(self, path):
        """Load LiDAR data from Argoverse PLY file."""
        plydata = PlyData.read(path)
        vertex_data = plydata['vertex'].data
        
        # Get x, y, z coordinates
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']
        
        # Get intensity if available, otherwise use default
        if 'intensity' in vertex_data.dtype.names:
            intensity = vertex_data['intensity']
        else:
            intensity = np.ones_like(x) * 0.5
        
        return np.column_stack([x, y, z, intensity])

def makeBVFeature(points, discretization, boundary):
    """Generate BEV feature map from point cloud."""
    # Handle different point cloud formats
    if points.shape[1] == 3:  # x, y, z only
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        i = np.ones_like(x) * 0.5  # Default intensity
    elif points.shape[1] >= 4:  # x, y, z, intensity (and possibly more)
        x, y, z, i = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    else:
        raise ValueError(f"Invalid point cloud shape: {points.shape}")

    # Filter points within boundary
    mask = ((x >= boundary["minX"]) & (x <= boundary["maxX"]) &
            (y >= boundary["minY"]) & (y <= boundary["maxY"]) &
            (z >= boundary["minZ"]) & (z <= boundary["maxZ"]))
    
    if not np.any(mask):
        print("Warning: No points within boundary after filtering")
        H = int((boundary["maxX"] - boundary["minX"]) / discretization)
        W = int((boundary["maxY"] - boundary["minY"]) / discretization)
        return np.zeros((3, H, W), dtype=np.float32)
    
    x, y, z, i = x[mask], y[mask], z[mask], i[mask]
    
    # Calculate BEV dimensions
    H = int((boundary["maxX"] - boundary["minX"]) / discretization)
    W = int((boundary["maxY"] - boundary["minY"]) / discretization)
    
    # Convert to grid indices
    x_idx = np.clip(((boundary["maxX"] - x) / discretization).astype(np.int32), 0, H-1)
    y_idx = np.clip(((y - boundary["minY"]) / discretization).astype(np.int32), 0, W-1)
    
    # Initialize feature maps
    h_map = np.zeros((H, W), dtype=np.float32)
    d_map = np.zeros((H, W), dtype=np.float32)
    i_map = np.zeros((H, W), dtype=np.float32)
    
    # Fill maps using vectorized operations
    z_rel = z - boundary["minZ"]
    for xi, yi, zi, ii in zip(x_idx, y_idx, z_rel, i):
        h_map[xi, yi] = max(h_map[xi, yi], zi)
        i_map[xi, yi] = max(i_map[xi, yi], ii)
        d_map[xi, yi] += 1
    
    # Normalize density
    d_map = np.clip(d_map / 10.0, 0, 1)
    
        # Normalize height and intensity
    if h_map.max() > 0:
        h_map = h_map / (boundary["maxZ"] - boundary["minZ"])
    if i_map.max() > 0:
        i_map = i_map / i_map.max()

    # Stack the feature maps (channel-first for PyTorch compatibility)
    bev_map = np.stack([d_map, h_map, i_map], axis=0).astype(np.float32)
    return bev_map


class ArgoverseCalibration:
    def __init__(self, calib_data, camera_name='ring_front_center'):
        self.camera_name = camera_name
        
        # Find camera intrinsics
        for cam in calib_data['camera_data_']:
            if cam['key'] == f'image_raw_{camera_name}':
                self.camera_data = cam['value']
                break
        else:
            raise ValueError(f"Camera {camera_name} not found in calibration.")

        # Camera intrinsics matrix P2
        fx = self.camera_data['focal_length_x_px_']
        fy = self.camera_data['focal_length_y_px_']
        cx = self.camera_data['focal_center_x_px_']
        cy = self.camera_data['focal_center_y_px_']
        self.P2 = np.array([[fx, 0, cx, 0],
                            [0, fy, cy, 0],
                            [0,  0,  1, 0]], dtype=np.float32)

        # Extrinsics
        self.vehicle_SE3_lidar = calib_data['vehicle_SE3_down_lidar_']
        self._setup_transforms()

    def _quat_to_rot_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z,   2*x*z+2*w*y],
            [2*x*y+2*w*z,     1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y,     2*y*z+2*w*x,   1-2*x**2-2*y**2]
        ], dtype=np.float32)

    def _setup_transforms(self):
        """Setup transformation matrices between sensor frames."""
        # Lidar → Vehicle
        lidar_t = np.array(self.vehicle_SE3_lidar['translation'])
        lidar_q = np.array(self.vehicle_SE3_lidar['rotation']['coefficients'])
        R_l2v = self._quat_to_rot_matrix(lidar_q)

        T_l2v = np.eye(4)
        T_l2v[:3, :3] = R_l2v
        T_l2v[:3, 3] = lidar_t

        # Vehicle → Camera (via inverse of camera → vehicle)
        cam = self.camera_data['vehicle_SE3_camera_']
        cam_t = np.array(cam['translation'])
        cam_q = np.array(cam['rotation']['coefficients'])
        R_c2v = self._quat_to_rot_matrix(cam_q)
        R_v2c = R_c2v.T
        t_v2c = -R_v2c @ cam_t

        T_v2c = np.eye(4)
        T_v2c[:3, :3] = R_v2c
        T_v2c[:3, 3] = t_v2c

        # Lidar → Camera: T_v2c * T_l2v
        self.T_l2c = T_v2c @ T_l2v

    def project_to_image(self, points_3d_lidar):
        """Project 3D lidar points to 2D image coordinates."""
        points_hom = np.hstack([points_3d_lidar, np.ones((points_3d_lidar.shape[0], 1))]).T
        
        # Debug print: Check shape and values before transformation
        # print(f"points_hom shape: {points_hom.shape}, values (first 3): \n{points_hom[:, :3]}")
        # print(f"T_l2c matrix: \n{self.T_l2c}")

        points_cam = self.L2C_front_center @ points_hom
        # Debug print: Check points in camera coordinates
        # print(f"points_cam shape: {points_cam.shape}, values (first 3): \n{points_cam[:, :3]}")

        valid_mask = points_cam[2, :] > 0.1
        # Debug print: Check z-values and mask
        # print(f"points_cam z-values (first 10): {points_cam[2, :10]}")
        # print(f"valid_mask (sum): {np.sum(valid_mask)}")

        if not np.any(valid_mask):
            print("Warning: No points in front of camera after Lidar to Camera transformation.")
            return np.full((points_3d_lidar.shape[0], 2), np.nan)

        points_2d = np.full((points_3d_lidar.shape[0], 2), np.nan)
        points_2d[valid_mask] = (self.P2 @ points_cam[:, valid_mask])[:2].T / points_cam[2, valid_mask][:, np.newaxis]
            
            # Debug print: Check final 2D points
            # print(f"Final points_2d (first 3 valid): \n{points_2d[valid_mask][:3, :] if np.any(valid_mask) else 'No valid points'}")

        return points_2d

def create_3d_bbox_corners(dimensions, location, rotation_y):
    """
    Create 8 corners of a 3D bounding box in camera coordinates.
    
    Parameters:
        dimensions: tuple (h, w, l) — height, width, length of the box
        location: tuple (x, y, z) — center of the box in 3D space
        rotation_y: float — yaw angle around Y-axis in radians
    
    Returns:
        corners_3d: (8, 3) numpy array of corner points
    """
    h, w, l = dimensions
    x, y, z = location

    # 3D bounding box corners (before rotation and translation)
    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [  0,    0,    0,    0,  -h,  -h,   -h,   -h]
    z_corners = [w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2]

    corners = np.array([x_corners, y_corners, z_corners])

    # Rotation matrix around Y-axis
    R = np.array([
        [ np.cos(rotation_y), 0, np.sin(rotation_y)],
        [ 0,                 1, 0                ],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ])

    # Rotate and translate the corners
    corners_3d = np.dot(R, corners).T
    corners_3d += np.array(location)

    return corners_3d

def draw_3d_bbox_on_image(image, det, calib, boundary, discretization):
    """Draw 3D bbox on image using detection and calibration."""
    try:
        # Convert detection from grid to metric coordinates (if needed)
        cx_m = det[0] * discretization + boundary["minX"]
        cy_m = det[1] * discretization + boundary["minY"]
        cz_m = det[2] + boundary["minZ"]  # Add elevation offset

        l_m, w_m, h_m, yaw = det[3], det[4], det[5], det[6]

        print(f"Detection (metric): cx={cx_m:.2f}, cy={cy_m:.2f}, cz={cz_m:.2f}, l={l_m:.2f}, w={w_m:.2f}, h={h_m:.2f}, yaw={yaw:.2f}")

        # Get 8 corner coordinates
        corners_3d = create_3d_bbox_corners((h_m, w_m, l_m), (cx_m, cy_m, cz_m), yaw)
        print(f"Corners 3D (first 3): \n{corners_3d[:3, :]}") # Print first 3 corners for brevity

        # Project corners to image plane
        pts_2d = calib.project_to_image(corners_3d)
        print(f"Projected 2D points (first 3): \n{pts_2d[:3, :]}") # Print first 3 projected points
        print(f"Are any projected points valid? {np.any(~np.isnan(pts_2d))}")

        if np.any(~np.isnan(pts_2d)):
            # Also check if points are roughly within image dimensions
            img_h, img_w = image.shape[:2]
            valid_pts_count = np.sum((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & 
                                     (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h))
            print(f"Number of projected points within image bounds: {valid_pts_count}/8")
            if valid_pts_count > 0: # Even if some are out, if at least one is in, we can try to draw
                draw_3d_bbox_lines(image, pts_2d)
                return True
    except Exception as e:
        print(f"3D bbox draw error: {e}")
    return False

def draw_3d_bbox_lines(img, pts):
    """Draw lines between 3D bbox corners projected to 2D image."""
    if pts.shape[0] != 8:
        return

    h_img, w_img = img.shape[:2]
    pts = pts.astype(int)

    # 12 edges of a cuboid
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    for start, end in edges:
        pt1 = pts[start]
        pt2 = pts[end]

        # Check if points are within image bounds
        if (0 <= pt1[0] < w_img and 0 <= pt1[1] < h_img and
            0 <= pt2[0] < w_img and 0 <= pt2[1] < h_img):
            cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 0), 2)
"""
def show_image_with_boxes(img, detections, calib, boundary, discretization):
    img_with_boxes = img.copy()
    valid_detections = 0
    
    if detections is not None and len(detections) > 0:
        for det in detections:
            if draw_3d_bbox_on_image(img_with_boxes, det, calib, boundary, discretization):
                valid_detections += 1
                
    return img_with_boxes, valid_detections
"""
def show_image_with_boxes(img, detections, calib, boundary, discretization):
    """Draw 3D bounding boxes on image."""
    img_with_boxes = img.copy()
    valid_detections = 0
    
    if detections is not None and len(detections) > 0:
        filtered_detections = [det for det in detections if isinstance(det, (np.ndarray, list)) and len(det) >= 7]

        if len(filtered_detections) > 0:
            for det in filtered_detections:
                if draw_3d_bbox_on_image(img_with_boxes, det, calib, boundary, discretization):
                    valid_detections += 1
                
    return img_with_boxes, valid_detections
if __name__ == '__main__':
    configs = parse_test_configs()

    # Create model
    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    #configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    configs.device = torch.device('cpu')
    model = model.to(device=configs.device)

    # Create data loader
    dataloader = ArgoverseDataLoader(configs.lidar_dir, configs.image_dir, configs.calib_file)
    
    # Create calibration object
    with open(configs.calib_file, 'r') as f:
        calib_data = json.load(f)
    calib = ArgoverseCalibration(calib_data)

    out_cap = None
    model.eval()

    with torch.no_grad():
        for batch_idx in range(len(dataloader)):
            metadata, bev_map, img_rgb = dataloader[batch_idx]
            
            # Convert BEV map to tensor and add batch dimension
            bev_tensor = torch.from_numpy(bev_map).unsqueeze(0).to(configs.device).float()
            
            t1 = time_synchronized()
            outputs = model(bev_tensor)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], 
                              outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]  # only first batch
            
            # Draw prediction in the BEV image
            bev_map_vis = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
            bev_map_vis = cv2.resize(bev_map_vis, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map_vis = draw_predictions(bev_map_vis, detections.copy(), configs.num_classes)

            # Rotate the bev_map for visualization
            bev_map_vis = cv2.rotate(bev_map_vis, cv2.ROTATE_180)

            # Convert image to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw 3D boxes on image
            img_with_boxes, num_detections = show_image_with_boxes(
                img_bgr, detections, calib, cnf.boundary, cnf.DISCRETIZATION
            )

            # Create combined visualization
            out_img = merge_rgb_to_bev(img_with_boxes, bev_map_vis, output_width=configs.output_width)

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS, detections: {}'.format(
                batch_idx, (t2 - t1) * 1000, 1 / (t2 - t1), num_detections))
            
            if configs.save_test_output:
                if configs.output_format == 'image':
                    img_fn = os.path.basename(metadata['img_path'])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, '{}.jpg'.format(img_fn)), out_img)
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, '{}.avi'.format(configs.output_video_fn)),
                            fourcc, 30, (out_cap_w, out_cap_h))

                    out_cap.write(out_img)
                else:
                    raise TypeError

            cv2.imshow('Argoverse Detection', out_img)
            print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            if cv2.waitKey(0) & 0xFF == 27:
                break
                
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()