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
            "Car": (0, 255, 0),       
            "Pedestrian": (255, 0, 0), 
            "Cyclist": (0, 0, 255)    
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
    parser.add_argument('--peak_thresh', type=float, default=0.0001)
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
    configs.num_input_features = 4 # This is often 4 (x,y,z,intensity) but makeBVFeature only uses 3 output channels (density, height, intensity)

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
        self.num_samples = len(self.lidar_files)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load LiDAR data
        lidar_path = self.lidar_files[idx]
        print("lidar path",lidar_path)
        points = self.load_lidar_data(lidar_path)
        
        # Create BEV map
        bev_map = makeBVFeature(points, cnf.DISCRETIZATION, cnf.boundary)
        
        # Load image
        img_path = self.image_files[idx] # Potential IndexError if image_files is shorter than lidar_files
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
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
        print("x,y,z,intensity shapes=",x.shape,y.shape,z.shape,intensity.shape)
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
        # Return a black image (all zeros) for the BEV map
        return np.zeros((3, H, W), dtype=np.float32)
    
    x, y, z, i = x[mask], y[mask], z[mask], i[mask]
    
    # Calculate BEV dimensions
    H = int((boundary["maxX"] - boundary["minX"]) / discretization)
    W = int((boundary["maxY"] - boundary["minY"]) / discretization)
    
    # Convert to grid indices
    # x_idx: maps world X (longitudinal) to BEV row index (height)
    # y_idx: maps world Y (lateral) to BEV column index (width)
    x_idx = np.clip(((boundary["maxX"] - x) / discretization).astype(np.int32), 0, H-1)
    y_idx = np.clip(((y - boundary["minY"]) / discretization).astype(np.int32), 0, W-1)
    
    # Initialize feature maps
    h_map = np.zeros((H, W), dtype=np.float32) # Height map
    d_map = np.zeros((H, W), dtype=np.float32) # Density map
    i_map = np.zeros((H, W), dtype=np.float32) # Intensity map
    
    # Fill maps using vectorized operations (or iterate for simplicity/debug)
    # Iterating is often clearer for small number of channels but less performant for large point clouds
    z_rel = z - boundary["minZ"] # Z relative to minZ
    for xi, yi, zi, ii in zip(x_idx, y_idx, z_rel, i):
        h_map[xi, yi] = max(h_map[xi, yi], zi) # Max height in cell
        i_map[xi, yi] = max(i_map[xi, yi], ii) # Max intensity in cell
        d_map[xi, yi] += 1 # Count points in cell
    
    # Normalize density (e.g., max 10 points per cell)
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
        self.camera_data = None
        for cam in calib_data['camera_data_']:
            if cam['key'] == f'image_raw_{camera_name}':
                self.camera_data = cam['value']
                break
        if self.camera_data is None:
            raise ValueError(f"Camera {camera_name} not found in calibration.")

        # Camera intrinsics matrix P2 (P matrix for projection from camera to image)
        fx = self.camera_data['focal_length_x_px_']
        fy = self.camera_data['focal_length_y_px_']
        cx = self.camera_data['focal_center_x_px_']
        cy = self.camera_data['focal_center_y_px_']
        self.P2 = np.array([[fx, 0, cx, 0],
                            [0, fy, cy, 0],
                            [0,  0,  1, 0]], dtype=np.float32)

        # Extrinsics: vehicle_SE3_lidar (Lidar to Vehicle transform)
        self.vehicle_SE3_lidar = calib_data['vehicle_SE3_down_lidar_']
        
        self._setup_transforms()

    def _quat_to_rot_matrix(self, q):
        """Convert quaternion [w, x, y, z] to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z,   2*x*z+2*w*y],
            [2*x*y+2*w*z,     1-2*x**2-2*z**2, 2*y*z-2*w*x],
            [2*x*z-2*w*y,     2*y*z+2*w*x,   1-2*x**2-2*y**2]
        ], dtype=np.float32)

    def _setup_transforms(self):
        """Setup transformation matrices between sensor frames."""
        # T_lidar_to_vehicle (Lidar frame to Vehicle frame)
        lidar_t = np.array(self.vehicle_SE3_lidar['translation'])
        lidar_q = np.array(self.vehicle_SE3_lidar['rotation']['coefficients'])
        R_l2v = self._quat_to_rot_matrix(lidar_q)

        T_l2v = np.eye(4, dtype=np.float32)
        T_l2v[:3, :3] = R_l2v
        T_l2v[:3, 3] = lidar_t

        # T_camera_to_vehicle (Camera frame to Vehicle frame)
        cam = self.camera_data['vehicle_SE3_camera_']
        cam_t = np.array(cam['translation'])
        cam_q = np.array(cam['rotation']['coefficients'])
        R_c2v = self._quat_to_rot_matrix(cam_q)
        
        # T_vehicle_to_camera (Vehicle frame to Camera frame) is inverse of T_camera_to_vehicle
        R_v2c = R_c2v.T
        t_v2c = -R_v2c @ cam_t

        T_v2c = np.eye(4, dtype=np.float32)
        T_v2c[:3, :3] = R_v2c
        T_v2c[:3, 3] = t_v2c

        # T_lidar_to_camera = T_vehicle_to_camera @ T_lidar_to_vehicle
        self.T_l2c = T_v2c @ T_l2v

    def project_to_image(self, points_3d_lidar):
        """Project 3D lidar points (in lidar frame) to 2D image coordinates (in pixel frame)."""
        if points_3d_lidar.shape[0] == 0:
            return np.array([])

        points_hom = np.hstack([points_3d_lidar, np.ones((points_3d_lidar.shape[0], 1))]).T
        
        # Transform points from Lidar frame to Camera frame
        points_cam = self.T_l2c @ points_hom 

        # Check if points are in front of the camera (positive Z in camera frame)
        # A small epsilon (0.1) is used to avoid issues with points exactly at Z=0
        valid_mask = points_cam[2, :] > 0.1 

        if not np.any(valid_mask):
            print("Warning: No points in front of camera after Lidar to Camera transformation.")
            return np.full((points_3d_lidar.shape[0], 2), np.nan, dtype=np.float32)

        # Project valid 3D camera points to 2D image plane
        # P2 @ [X_c, Y_c, Z_c, 1].T (then divide by Z_c for perspective projection)
        projected_points_on_image_plane = self.P2 @ points_cam[:, valid_mask]
        
        # Perform perspective division
        pts_2d_valid = projected_points_on_image_plane[:2, :] / projected_points_on_image_plane[2, :]
        
        # Create a full array and fill only valid points
        points_2d = np.full((points_3d_lidar.shape[0], 2), np.nan, dtype=np.float32)
        points_2d[valid_mask] = pts_2d_valid.T
            
        return points_2d

def create_3d_bbox_corners(dimensions, location, rotation_y):
    """
    Create 8 corners of a 3D bounding box in its local coordinate system,
    then transform to a global system based on location and rotation_y.
    
    Parameters:
        dimensions: tuple (h, w, l) — height, width, length of the box
        location: tuple (x, y, z) — center of the box in 3D space (e.g., ego-vehicle frame)
        rotation_y: float — yaw angle around the Y-axis (vertical axis in camera/object frame) in radians
    
    Returns:
        corners_3d: (8, 3) numpy array of corner points in the global coordinate system.
    """
    h, w, l = dimensions
    x, y, z = location
    print("Bbox dimensions (h,w,l):", h, w, l, "Location (x,y,z):", x, y, z, "Yaw (rad):", rotation_y)
    
    # 3D bounding box corners (relative to object center and orientation before rotation)
    # The order of corners typically assumes a standard orientation.
    # In KITTI/similar setups, the box is defined with its origin at the bottom-center of the front face.
    # Here, it seems to be defined with the origin at the center of the bottom face,
    # and rotation_y is around the local Y-axis (which is typically vertical for object).
    # x_corners: relative longitudinal offsets
    # y_corners: relative vertical offsets (y=0 is bottom, y=-h is top given the calculation)
    # z_corners: relative lateral offsets
    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [  0,    0,    0,    0,  -h,  -h,   -h,   -h] # Y-axis up, so top is negative h
    z_corners = [w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

    # Rotation matrix around Y-axis (standard for 3D object detection in camera coordinates)
    R = np.array([
        [ np.cos(rotation_y), 0, np.sin(rotation_y)],
        [ 0,                  1, 0                ],
        [-np.sin(rotation_y), 0, np.cos(rotation_y)]
    ], dtype=np.float32)

    # Rotate corners
    # The output of this (3, 8) matrix contains the rotated corners relative to the object's origin (0,0,0)
    rotated_corners = np.dot(R, corners)
    
    # Translate corners to the actual object location
    # The `location` (x, y, z) is the center of the *bottom* face of the bounding box.
    corners_3d = rotated_corners.T + np.array(location, dtype=np.float32)
    
    print("Corners 3D (shape, first 3 rows):\n", corners_3d.shape, "\n", corners_3d[:3, :])
    return corners_3d

def draw_3d_bbox_on_image(image, det, calib, boundary, discretization):
    """Draw 3D bbox on image using detection and calibration."""
    try:
        print("Entering draw_3d_bbox_on_image...")
        
        # Ensure 'det' has enough elements for a valid detection
        if not (isinstance(det, (np.ndarray, list)) and len(det) >= 8): # score, x, y, z, h, w, l, yaw
            print(f"Warning: Detection has less than 8 elements. Skipping. Detection: {det}")
            return False

        # Extract relevant fields from detection array
        # Assuming det format from post_processing: [score, x_grid, y_grid, z_coor, dim_h, dim_w, dim_l, yaw_angle_in_radians]
        score = det[0]
        x_grid, y_grid = det[1], det[2] # These are BEV grid coordinates
        z_coor = det[3] # This is the object's Z-coordinate (height) in the BEV system, typically relative to minZ

        # Dimensions (height, width, length) in metric space
        h_m = det[4]
        w_m = det[5]
        l_m = det[6]
        
        yaw = det[7] # Yaw angle in radians

        # Convert BEV grid coordinates (x_grid, y_grid) to ego-vehicle metric coordinates (cx_m, cy_m)
        # Recall from makeBVFeature:
        # x_idx (BEV row) corresponds to world X, and is calculated as `(boundary["maxX"] - x) / discretization`
        # y_idx (BEV col) corresponds to world Y, and is calculated as `(y - boundary["minY"]) / discretization`
        # So, to reverse:
        cx_m = boundary["maxX"] - x_grid * discretization # World X from BEV row (longitudinal)
        cy_m = y_grid * discretization + boundary["minY"] # World Y from BEV column (lateral)
        
        # Add elevation offset for Z-coordinate
        cz_m = z_coor + boundary["minZ"] 

        print(f"Detection (score:{score:.2f}, metric: cx={cx_m:.2f}, cy={cy_m:.2f}, cz={cz_m:.2f}, l={l_m:.2f}, w={w_m:.2f}, h={h_m:.2f}, yaw={yaw:.2f})")

        # Create 8 corner coordinates of the 3D bounding box in the ego-vehicle coordinate system.
        # `create_3d_bbox_corners` expects (h, w, l) for dimensions and (x, y, z) for location,
        # where (x, y, z) is the center of the *bottom* face of the bounding box.
        # The `yaw` from the model is assumed to be the rotation around the vertical axis (Y in camera frame, Z in ego-vehicle frame)
        # if `create_3d_bbox_corners` is set up for that. For standard KITTI-like setups, `rotation_y` is used,
        # which means rotation around the Y-axis (which is usually up/down in camera coordinates).
        # Given SFA3D context, `rotation_y` is typically the yaw.
        corners_3d_ego = create_3d_bbox_corners((h_m, w_m, l_m), (cx_m, cy_m, cz_m), yaw)
        
        # Project 3D corners (in ego-vehicle/lidar frame) to 2D image plane
        pts_2d = calib.project_to_image(corners_3d_ego)
        print(f"Projected 2D points (shape, first 3 rows):\n {pts_2d.shape}\n {pts_2d[:3, :] if pts_2d.size > 0 else 'No points'}")
        print(f"Are any projected points valid (not NaN)? {np.any(~np.isnan(pts_2d))}")

        if np.any(~np.isnan(pts_2d)):
            # Filter out NaN points before drawing
            valid_pts_2d = pts_2d[~np.isnan(pts_2d).any(axis=1)]
            
            if valid_pts_2d.shape[0] == 8: # Ensure all 8 corners projected successfully
                # Check if points are roughly within image dimensions.
                # Even if some are out, if at least one is in, we can try to draw as lines might cross
                img_h, img_w = image.shape[:2]
                valid_pts_count = np.sum((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & 
                                         (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h))
                print(f"Number of projected points within image bounds: {valid_pts_count}/8")
                
                # Draw only if at least some points are visible or the projection is coherent
                # A simple check for 8 valid points is sufficient.
                draw_3d_bbox_lines(image, pts_2d)
                return True
            else:
                print(f"Warning: Only {valid_pts_2d.shape[0]}/8 corners projected successfully. Skipping 3D bbox draw.")
                return False
        else:
            print("Warning: No valid 2D projection for this detection. Skipping 3D bbox draw.")
            return False
            
    except Exception as e:
        print(f"3D bbox draw error: {e}")
        return False

def draw_3d_bbox_lines(img, pts):
    """Draw lines between 3D bbox corners projected to 2D image."""
    print("Entering draw_3d_bbox_lines...")
    if pts.shape[0] != 8:
        print(f"Error: Expected 8 points for 3D bbox drawing, got {pts.shape[0]}.")
        return

    h_img, w_img = img.shape[:2]
    # Ensure points are integers for OpenCV drawing functions
    pts = pts.astype(int)

    # Define the 12 edges of a cuboid by their corner indices
    # This assumes a specific order of corners from create_3d_bbox_corners
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face (assuming y=0 is bottom)
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face (assuming y=-h is top)
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges connecting bottom to top
    ]

    for start, end in edges:
        pt1 = pts[start]
        pt2 = pts[end]

        # Basic check to ensure points are somewhat within or near image bounds
        # Drawing lines from completely off-screen points might lead to unexpected results.
        # A more sophisticated clipping algorithm could be used for precise drawing.
        if (0 <= pt1[0] < w_img and 0 <= pt1[1] < h_img) or \
           (0 <= pt2[0] < w_img and 0 <= pt2[1] < h_img):
            cv2.line(img, tuple(pt1), tuple(pt2), (0, 255, 0), 2) # Green color, thickness 2

def show_image_with_boxes(img, detections, calib, boundary, discretization):
    """Draw 3D bounding boxes on image."""
    img_with_boxes = img.copy()
    valid_detections_count = 0
    
    if detections is not None:
        # 'detections' is a dictionary with class_id as keys and numpy arrays of detections as values
        for class_id, class_detections in detections.items():
            if class_detections is not None and len(class_detections) > 0:
                # class_detections is a numpy array where each row is a detection
                for det_item in class_detections: # Iterate over each individual detection
                    # Ensure det_item is a valid detection with enough elements for 3D drawing
                    # Format: [score, x_grid, y_grid, z_coor, h, w, l, yaw, ...]
                    if isinstance(det_item, (np.ndarray, list)) and len(det_item) >= 8:
                        if draw_3d_bbox_on_image(img_with_boxes, det_item, calib, boundary, discretization):
                            valid_detections_count += 1
                        else:
                            # This print helps debug why a specific detection isn't drawn
                            # Print up to 8 elements for score, x, y, z, h, w, l, yaw
                            print(f"Failed to draw 3D bbox for detection (class {class_id}, score {det_item[0]:.2f}): {det_item[:8]}") 
    return img_with_boxes, valid_detections_count

if __name__ == '__main__':
    configs = parse_test_configs()

    # Create model
    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    # Force CPU for demonstration if --no_cuda is used, otherwise use specified GPU or default to CPU if no GPU.
    configs.device = torch.device('cpu' if configs.no_cuda or not torch.cuda.is_available() else f'cuda:{configs.gpu_idx}')
    print(f"Using device: {configs.device}")
    model = model.to(device=configs.device)

    # Create data loader
    dataloader = ArgoverseDataLoader(configs.lidar_dir, configs.image_dir, configs.calib_file)
    
    # Create calibration object
    with open(configs.calib_file, 'r') as f:
        calib_data = json.load(f)
    calib = ArgoverseCalibration(calib_data)

    out_cap = None # For video output
    model.eval() # Set model to evaluation mode

    with torch.no_grad(): # Disable gradient calculation for inference
        # Iterate through samples
        for batch_idx in range(len(dataloader)):
            metadata, bev_map, img_rgb = dataloader[batch_idx]
            print(f'\n\n[INFO] Testing the {batch_idx}th sample: {metadata["img_path"]}')
            
            # Convert BEV map to tensor and add batch dimension (B, C, H, W)
            bev_tensor = torch.from_numpy(bev_map).unsqueeze(0).to(configs.device).float()
            
            t1 = time_synchronized()
            outputs = model(bev_tensor)
            print(f'[INFO] Model outputs keys: {outputs.keys()}')
            
            # ============ IMPROVED HEATMAP VISUALIZATION ============
            
            # Get raw heatmap (before sigmoid) - sum across all classes for better visualization
            # Detach from graph, move to CPU, convert to numpy, get first item from batch
            raw_hm_cen = outputs['hm_cen'].detach().cpu().numpy()[0]  # Shape: (num_classes, H, W)
            
            # Sum across all classes to get combined heatmap
            raw_hm_combined = np.sum(raw_hm_cen, axis=0)  # Shape: (H, W)
            
            print(f"Raw heatmap stats - Min: {raw_hm_combined.min():.4f}, Max: {raw_hm_combined.max():.4f}, Mean: {raw_hm_combined.mean():.4f}")
            
            # Normalize raw heatmap for visualization (0-1 range)
            if raw_hm_combined.max() > raw_hm_combined.min():
                raw_hm_vis = (raw_hm_combined - raw_hm_combined.min()) / (raw_hm_combined.max() - raw_hm_combined.min())
            else: # Handle case where all values are the same (e.g., all zeros)
                raw_hm_vis = np.zeros_like(raw_hm_combined)
            
            # Convert to 8-bit (0-255) and apply colormap
            raw_hm_vis = (raw_hm_vis * 255).astype(np.uint8)
            raw_hm_colored = cv2.applyColorMap(raw_hm_vis, cv2.COLORMAP_JET)
            
            # Resize for better visibility in imshow window
            raw_hm_resized = cv2.resize(raw_hm_colored, (512, 512))
            cv2.imshow('Raw Center Heatmap (Before Sigmoid)', raw_hm_resized)
            
            # Apply sigmoid to outputs for decoding
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            
            # Get sigmoid heatmap
            sigmoid_hm_cen = outputs['hm_cen'].detach().cpu().numpy()[0]  # Shape: (num_classes, H, W)
            
            # Sum across all classes for combined view
            sigmoid_hm_combined = np.sum(sigmoid_hm_cen, axis=0)  # Shape: (H, W)
            
            print(f"Sigmoid heatmap stats - Min: {sigmoid_hm_combined.min():.4f}, Max: {sigmoid_hm_combined.max():.4f}, Mean: {sigmoid_hm_combined.mean():.4f}")
            
            # Normalize sigmoid heatmap for visualization (0-1 range)
            if sigmoid_hm_combined.max() > sigmoid_hm_combined.min():
                sigmoid_hm_vis = (sigmoid_hm_combined - sigmoid_hm_combined.min()) / (sigmoid_hm_combined.max() - sigmoid_hm_combined.min())
            else:
                sigmoid_hm_vis = np.zeros_like(sigmoid_hm_combined)
            
            # Convert to 8-bit and apply colormap
            sigmoid_hm_vis = (sigmoid_hm_vis * 255).astype(np.uint8)
            sigmoid_hm_colored = cv2.applyColorMap(sigmoid_hm_vis, cv2.COLORMAP_JET)
            
            # Resize for better visibility
            sigmoid_hm_resized = cv2.resize(sigmoid_hm_colored, (512, 512))
            cv2.imshow('Sigmoid Center Heatmap', sigmoid_hm_resized)
            
            # ============ INDIVIDUAL CLASS HEATMAPS ============
            
            # Show individual class heatmaps if there are multiple classes
            if configs.num_classes > 1:
                for class_idx in range(sigmoid_hm_cen.shape[0]):
                    class_hm = sigmoid_hm_cen[class_idx]
                    
                    if class_hm.max() > 0:  # Only show if there's activity for this class
                        print(f"Class {class_idx} heatmap stats - Min: {class_hm.min():.4f}, Max: {class_hm.max():.4f}")
                        
                        # Normalize and colorize
                        class_hm_norm = (class_hm - class_hm.min()) / (class_hm.max() - class_hm.min()) if class_hm.max() > class_hm.min() else np.zeros_like(class_hm)
                        class_hm_vis = (class_hm_norm * 255).astype(np.uint8)
                        class_hm_colored = cv2.applyColorMap(class_hm_vis, cv2.COLORMAP_HOT) # HOT colormap is good for single-channel heatmaps
                        class_hm_resized = cv2.resize(class_hm_colored, (256, 256)) # Smaller for multiple windows
                        
                        cv2.imshow(f'Class {class_idx} Heatmap', class_hm_resized)
            
            # ============ ADDITIONAL DEBUG INFO ============
            
            print(f"Model output tensor shapes:")
            for key, value in outputs.items():
                print(f"  {key}: {value.shape}")
            
            # Check for any non-zero activations from model outputs to confirm model is not dead
            has_activations = any(torch.sum(torch.abs(output)) > 0 for output in outputs.values())
            print(f"Model has non-zero activations in outputs: {has_activations}")
            
            
            # Decode detections from model outputs
            # This function uses the configured K (top K detections)
            detections_raw = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], 
                                    outputs['z_coor'], outputs['dim'], K=configs.K)
            print("Printing the shapes-----------------------")
            print(f"Decoded raw detections shape: {detections_raw.shape}")
            print(f"Decoded raw detections (first 1 rows):\n{detections_raw[:1]}")
            
            # Move decoded detections to CPU and convert to numpy for post-processing
            detections_raw_np = detections_raw.cpu().numpy().astype(np.float32)
            print(f"Decoded raw detections numpy shape: {detections_raw_np.shape}")
            print(f"Decoded raw detections numpy (first 1 rows):\n{detections_raw_np[:1]}")
            
            # Apply non-maximum suppression, filtering by peak_thresh, etc.
            # This step converts the raw decoded predictions into final bounding boxes
            # The output format here is crucial for subsequent drawing functions
            # It's typically a list of dictionaries or a dictionary mapping class IDs to numpy arrays of detections.
            
            detections_final = post_processing(detections_raw_np, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            print(f"Post-processed detections shape: {len(detections_final)}")
            print(f"Post-processed detections (first 1 items):\n{detections_final[:1] if isinstance(detections_final, list) else detections_final}")
            
            t2 = time_synchronized()

            # The `detections_for_single_sample` should be a dictionary where keys are class_ids
            # and values are numpy arrays of detections for that class.
            # `post_processing` function typically returns a list of dictionaries if batch_size > 1,
            # or a single dictionary if batch_size = 1 (as is the case here).
            detections_for_single_sample = detections_final[0] if isinstance(detections_final, list) else detections_final
            
            # Draw prediction in the BEV image
            # Convert BEV map from (C, H, W) float to (H, W, C) uint8 for OpenCV visualization
            bev_map_vis = (bev_map.transpose(1, 2, 0) * 255).astype(np.uint8)
            bev_map_vis = cv2.resize(bev_map_vis, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            # `draw_predictions` handles drawing 2D boxes on the BEV map
            bev_map_vis = draw_predictions(bev_map_vis, detections_for_single_sample.copy(), configs.num_classes)
            # Rotate BEV map for typical display orientation (e.g., ego-vehicle facing up)
            bev_map_vis = cv2.rotate(bev_map_vis, cv2.ROTATE_180) 
            bev_map_vis = cv2.flip(bev_map_vis,0 )  # Flip vertically (upside down)
            
            # Convert original image from RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw 3D boxes on the front-camera image
            img_with_boxes, num_detections = show_image_with_boxes(
                img_bgr, detections_for_single_sample, calib, cnf.boundary, cnf.DISCRETIZATION
            )
            
            # Create combined visualization of RGB image with 3D boxes and BEV map with 2D boxes
            out_img = merge_rgb_to_bev(img_with_boxes, bev_map_vis, output_width=configs.output_width)
            
            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS, detections drawn on image: {}'.format(
                batch_idx, (t2 - t1) * 1000, 1 / (t2 - t1), num_detections))
            
            # Save output if configured
            if configs.save_test_output:
                if configs.output_format == 'image':
                    # Extract original filename without extension for saving
                    img_fn = os.path.basename(metadata['img_path']).split('.')[0] 
                    output_path = os.path.join(configs.results_dir, f'{img_fn}.jpg')
                    cv2.imwrite(output_path, out_img)
                    print(f"Saved output image to: {output_path}")
                elif configs.output_format == 'video':
                    if out_cap is None:
                        # Initialize video writer once
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Codec for AVI
                        output_video_path = os.path.join(configs.results_dir, f'{configs.output_video_fn}.avi')
                        out_cap = cv2.VideoWriter(output_video_path, fourcc, 30, (out_cap_w, out_cap_h))
                        print(f"Initialized video writer for: {output_video_path}")
                    out_cap.write(out_img)
                else:
                    raise TypeError("Unsupported output_format. Must be 'image' or 'video'.")

            # Display the combined output image
            cv2.imshow('Argoverse Detection (Combined View)', out_img)
            print('\n[INFO] Press \'n\' to see the next sample >>> Press \'Esc\' to quit...\n')
            # Wait for key press: 'n' for next, 'Esc' to exit
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('n'): # 'n' key
                continue # Go to next iteration
                
    if out_cap:
        out_cap.release() # Release video writer if it was opened
    cv2.destroyAllWindows() # Close all OpenCV display windows