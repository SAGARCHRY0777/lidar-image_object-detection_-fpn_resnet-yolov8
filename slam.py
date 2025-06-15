import argparse, sys, os, time, warnings, cv2, numpy as np, torch, matplotlib.pyplot as plt, matplotlib.cm as cm
from ultralytics import YOLO
warnings.filterwarnings("ignore", category=UserWarning)
from easydict import EasyDict as edict

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import necessary modules from your project
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils_slam import Calibration

# --- Placeholder for SLAM/Calibration related functions ---
def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true')
    parser.add_argument('--output_format', type=str, default='image')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18')
    parser.add_argument('--output-width', type=int, default=608)
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Confidence threshold for fusion')

    configs = edict(vars(parser.parse_args()))

    configs.pin_memory = True
    configs.distributed = False
    configs.device = torch.device("cpu")
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
    configs.num_direction = 2

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4
    configs.root_dir = 'D:\spa\SFA3D' # Ensure this path is correct for your setup
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'fused_results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2] format
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]

    # Calculate intersection
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def apply_nms_to_fused_detections(detections, nms_threshold=0.5):
    """Apply NMS to fused detections from both models"""
    if len(detections) == 0:
        return []

    # Sort by confidence score (descending)
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    keep_detections = []

    for i, detection in enumerate(detections):
        should_keep = True

        for kept_detection in keep_detections:
            iou = calculate_iou(detection['box'], kept_detection['box'])
            if iou > nms_threshold:
                should_keep = False
                break

        if should_keep:
            keep_detections.append(detection)

    return keep_detections


def convert_sfa3d_to_2d_boxes(sfa_detections, calib_params, img_shape):
    """
    Convert SFA3D 3D detections to 2D bounding boxes using provided calibration parameters.
    Args:
        sfa_detections (np.array): SFA3D raw detections.
        calib_params (dict): Dictionary containing 'P2', 'R0', 'V2C' matrices.
        img_shape (tuple): (height, width) of the image.
    Returns:
        tuple: (list of 2D bounding boxes, list of confidences)
    """
    boxes_2d = []
    confidences = []

    if len(sfa_detections) > 0:
        kitti_dets = convert_det_to_real_values(sfa_detections)

        # Create a Calibration object from the provided parameters
        temp_calib = Calibration(None) # Initialize with None as we'll set matrices manually
        temp_calib.P2 = calib_params['P2']
        temp_calib.R0 = calib_params['R0']
        temp_calib.V2C = calib_params['V2C']

        for detection in kitti_dets:
            confidence = detection[0]
            if confidence < 0.2:  # Skip low confidence detections
                continue

            # Convert 3D box to camera coordinates
            box_3d = detection[1:]
            box_3d_cam = lidar_to_camera_box(box_3d.reshape(1, -1), temp_calib.V2C, temp_calib.R0, temp_calib.P2)[0]

            # Project 3D box to 2D
            x, y, z, h, w, l, ry = box_3d_cam

            # Create 8 corners of 3D box
            corners_3d = np.array([
                [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                [0, 0, 0, 0, -h, -h, -h, -h],
                [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
            ])

            # Rotation matrix
            R = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])

            # Rotate and translate
            corners_3d = np.dot(R, corners_3d)
            corners_3d[0, :] += x
            corners_3d[1, :] += y
            corners_3d[2, :] += z

            # Project to 2D using P2
            corners_2d = temp_calib.P2.dot(np.vstack((corners_3d, np.ones((1, 8)))))
            corners_2d = corners_2d[:2] / corners_2d[2]

            # Get 2D bounding box
            min_x, max_x = np.min(corners_2d[0]), np.max(corners_2d[0])
            min_y, max_y = np.min(corners_2d[1]), np.max(corners_2d[1])

            # Clip to image boundaries
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(img_shape[1], max_x)
            max_y = min(img_shape[0], max_y)

            if max_x > min_x and max_y > min_y:
                boxes_2d.append([int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)])
                confidences.append(confidence)

    return boxes_2d, confidences


def yolov8_detect(image_path, model):
    """YOLOv8 detection function"""
    results = model(image_path)
    detections = results[0]
    boxes = detections.boxes

    result_boxes = []
    result_confidences = []
    result_class_ids = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            w, h = x2 - x1, y2 - y1
            result_boxes.append([x1, y1, w, h])
            result_confidences.append(conf)
            result_class_ids.append(cls)

    return result_boxes, result_confidences, result_class_ids


def create_fused_detections(yolov8_data, sfa3d_data, confidence_threshold=0.3):
    """Create fused detection list from both models"""
    yolov8_boxes, yolov8_confidences, yolov8_class_ids, yolov8_class_names = yolov8_data
    sfa3d_boxes, sfa3d_confidences = sfa3d_data

    fused_detections = []

    # Add YOLOv8 detections
    for i, (box, conf, class_id) in enumerate(zip(yolov8_boxes, yolov8_confidences, yolov8_class_ids)):
        if conf >= confidence_threshold:
            fused_detections.append({
                'box': box,
                'confidence': conf,
                'class_id': class_id,
                'class_name': yolov8_class_names[class_id],
                'model': 'YOLOv8',
                'color': (0, 255, 255)  # Cyan for YOLOv8
            })

    # Add SFA3D detections
    for i, (box, conf) in enumerate(zip(sfa3d_boxes, sfa3d_confidences)):
        if conf >= confidence_threshold:
            fused_detections.append({
                'box': box,
                'confidence': conf,
                'class_id': 0,  # Assuming car class for SFA3D for simplicity based on its typical output
                'class_name': 'car',
                'model': 'SFA3D',
                'color': (255, 0, 0)  # Blue for SFA3D
            })

    return fused_detections


def draw_fused_detections(img, fused_detections):
    """Draw fused detections with different colors for each model"""
    img_with_detections = img.copy()

    for detection in fused_detections:
        box = detection['box']
        confidence = detection['confidence']
        class_name = detection['class_name']
        model = detection['model']
        color = detection['color']

        x, y, w, h = box

        # Draw bounding box
        cv2.rectangle(img_with_detections, (x, y), (x + w, y + h), color, 2)

        # Draw label with model name
        label = f"{model}: {class_name} {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw label background
        cv2.rectangle(img_with_detections, (x, y - label_size[1] - 10),
                      (x + label_size[0], y), color, -1)

        # Draw label text
        cv2.putText(img_with_detections, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_with_detections


def create_detection_summary(fused_detections_before_nms, fused_detections_after_nms):
    """Create a summary image showing detection statistics"""
    summary_img = np.zeros((300, 600, 3), dtype=np.uint8)

    # Count detections by model
    yolo_before = sum(1 for d in fused_detections_before_nms if d['model'] == 'YOLOv8')
    sfa_before = sum(1 for d in fused_detections_before_nms if d['model'] == 'SFA3D')
    yolo_after = sum(1 for d in fused_detections_after_nms if d['model'] == 'YOLOv8')
    sfa_after = sum(1 for d in fused_detections_after_nms if d['model'] == 'SFA3D')

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(summary_img, "Detection Summary", (150, 30), font, 0.8, (255, 255, 255), 2)

    cv2.putText(summary_img, "Before NMS:", (50, 80), font, 0.6, (255, 255, 255), 2)
    cv2.putText(summary_img, f"YOLOv8: {yolo_before}", (50, 110), font, 0.6, (0, 255, 255), 2) # Cyan
    cv2.putText(summary_img, f"SFA3D: {sfa_before}", (50, 140), font, 0.6, (255, 0, 0), 2) # Blue
    cv2.putText(summary_img, f"Total: {len(fused_detections_before_nms)}", (50, 170), font, 0.6, (255, 255, 255), 2)

    cv2.putText(summary_img, "After NMS:", (300, 80), font, 0.6, (255, 255, 255), 2)
    cv2.putText(summary_img, f"YOLOv8: {yolo_after}", (300, 110), font, 0.6, (0, 255, 255), 2) # Cyan
    cv2.putText(summary_img, f"SFA3D: {sfa_after}", (300, 140), font, 0.6, (255, 0, 0), 2) # Blue
    cv2.putText(summary_img, f"Total: {len(fused_detections_after_nms)}", (300, 170), font, 0.6, (255, 255, 255), 2)

    # Legend
    cv2.putText(summary_img, "Legend:", (50, 220), font, 0.6, (255, 255, 255), 2)
    cv2.rectangle(summary_img, (50, 230), (70, 250), (0, 255, 255), -1)  # Cyan
    cv2.putText(summary_img, "YOLOv8", (80, 245), font, 0.5, (255, 255, 255), 1)
    cv2.rectangle(summary_img, (200, 230), (220, 250), (255, 0, 0), -1)  # Blue
    cv2.putText(summary_img, "SFA3D", (230, 245), font, 0.5, (255, 255, 255), 1)

    return summary_img


# --- Mock RANSAC Function ---
def mock_ransac_pose_estimation(points3d, points2d, K_matrix, dist_coeffs):
    """
    A mock RANSAC-like function for camera pose estimation.
    In a real scenario, this would use cv2.solvePnP with RANSAC.
    It simulates getting a robust camera pose (R, T) from 3D-2D correspondences.
    """
    print("  Applying mock RANSAC for pose estimation...")
    # This is a highly simplified mock.
    # In reality, you'd find correspondences between 3D points (e.g., from LiDAR map)
    # and 2D image points, then use solvePnP with RANSAC to find the camera's pose.

    # Example: If your "perfect" calibration comes from a dataset, it's often already
    # a highly accurate R, T for that specific frame.
    # We will return dummy but plausible rotation and translation vectors
    # as if RANSAC found a good solution.

    # Dummy rotation vector (Rodrigues form) and translation vector
    rvec = np.array([0.01, -0.02, 0.005], dtype=np.float32) # Small rotation
    tvec = np.array([0.1, 0.2, 1.5], dtype=np.float32)    # Small translation

    # Simulate adding some noise and then RANSAC removing it
    # In a real scenario, `points3d` and `points2d` would be actual matched features
    # and `cv2.solvePnP(flags=cv2.SOLVEPNP_EPNP, rvec=rvec, tvec=tvec, useExtrinsicGuess=True)`
    # with `cv2.RANSAC` method would be used.

    # For now, we just return a pre-defined R and T.
    R_matrix, _ = cv2.Rodrigues(rvec)
    T_matrix = tvec.reshape(3, 1)

    return R_matrix, T_matrix, None # Returning None for inliers for simplicity


# --- Unified Placeholder for Calibration Retrieval from various "SLAM" methods ---
def get_calibration_from_slam_placeholder(image_path, lidar_data=None, slam_method="KITTI_DATASET_CALIB"):
    """
    Unified placeholder function to simulate getting calibration from different "SLAM" sources.
    This demonstrates where you would integrate outputs from various SLAM algorithms
    or directly read pre-existing calibration data.

    Args:
        image_path (str): Path to the current image frame. Used for frame indexing.
        lidar_data (np.array, optional): Raw LiDAR point cloud data. Placeholder for LiDAR-based SLAM.
        slam_method (str): Specifies the calibration source/method.
                          Options: "KITTI_DATASET_CALIB", "VISUAL_SLAM_SIM", "LIDAR_SLAM_SIM", "VISUAL_INERTIAL_SLAM_SIM"
    Returns:
        dict: A dictionary containing 'P2', 'R0', 'V2C' numpy arrays.
    """
    print(f"\n--- Retrieving calibration using '{slam_method}' for {os.path.basename(image_path)} ---")

    # --- Static Calibration Matrices (Base for all simulations/placeholders) ---
    # These represent a *fixed* sensor setup. In a real scenario, these would be from
    # your vehicle's intrinsic calibration.
    P2_base = np.array([
        [7.215e+02, 0.000e+00, 6.095e+02, 4.485e+01],
        [0.000e+00, 7.215e+02, 1.728e+02, 2.163e-01],
        [0.000e+00, 0.000e+00, 1.000e+00, 2.745e-03]
    ], dtype=np.float32)

    R0_base = np.array([
        [9.999e-01, 9.837e-03, -7.445e-03],
        [-9.869e-03, 9.999e-01, -4.278e-03],
        [7.402e-03, 4.351e-03, 9.999e-01]
    ], dtype=np.float32)

    V2C_base = np.array([
        [7.533e-03, -9.999e-01, -1.481e-02, -4.069e-03],
        [1.465e-02, 1.496e-02, -9.997e-01, -7.631e-02],
        [9.998e-01, 7.523e-03, 1.480e-02, -2.717e-01]
    ], dtype=np.float32)

    # These matrices would be updated based on the "SLAM" method chosen.
    P2_final = P2_base.copy()
    R0_final = R0_base.copy()
    V2C_final = V2C_base.copy()

    # --- Data Input Placeholders ---
    # In a real SLAM system, you would read frame-specific data here.
    # For KITTI, this typically means reading the corresponding
    # image, LiDAR velodyne points, and maybe IMU/GPS data.
    frame_idx = int(os.path.basename(image_path).split('.')[0])
    calib_file_path = image_path.replace('image_2', 'calib').replace('.png', '.txt') # Adjust based on your dataset structure

    if slam_method == "KITTI_DATASET_CALIB":
        # This is where you load the "perfect" calibration directly from KITTI files.
        # This is often the *ground truth* or pre-computed, highly accurate calibration.
        # This method assumes your dataloader already provides the correct path or
        # you can construct it to read the calib file.
        print(f"  Loading calibration from KITTI dataset file: {calib_file_path}")
        # In a real KITTI dataset integration, you'd parse this file:
        # Example (simplified):
        # with open(calib_file_path, 'r') as f:
        #     lines = f.readlines()
        #     P2_line = lines[2].strip().split(' ')[1:] # P2: line 3, skip 'P2:'
        #     R0_line = lines[4].strip().split(' ')[1:] # R0_rect: line 5
        #     V2C_line = lines[5].strip().split(' ')[1:] # Tr_velo_to_cam: line 6
        #     P2_final = np.array(P2_line, dtype=np.float32).reshape(3, 4)
        #     R0_final = np.array(R0_line, dtype=np.float32).reshape(3, 3)
        #     V2C_final = np.array(V2C_line, dtype=np.float32).reshape(3, 4)
        #
        # For this demonstration, we'll just use the base static values as if
        # they were loaded "perfectly" from a static calibration file.
        # If your `create_test_dataloader` already provides `metadatas['calib']`, you should use that.
        # For simplicity, we'll just use the base matrices for static "perfect" calibration.
        pass # No change needed, base matrices are used

    elif slam_method == "VISUAL_SLAM_SIM":
        print("  Simulating Visual SLAM (e.g., ORB-SLAM3, VINS-Fusion).")
        # --- INPUT FOR VISUAL SLAM ---
        # This is where you'd provide the current `img_rgb` and potentially `prev_img_rgb`
        # and extract features (e.g., ORB, SIFT, SURF) to feed into a visual SLAM backend.
        # For a truly dynamic setup, you'd need the actual `img_rgb` (or `img_path`) here
        # and feature points (e.g., from `cv2.detectAndCompute`).
        # Example:
        # current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # (Assuming you have a previous image and a feature tracker/matcher)
        # kp1, des1 = orb.detectAndCompute(prev_image, None)
        # kp2, des2 = orb.detectAndCompute(current_image, None)
        # matches = matcher.knnMatch(des1, des2, k=2)
        # good_matches = filter_matches(matches)
        #
        # In a real system, the SLAM library would estimate camera pose (R, T)
        # for `current_image` based on `prev_image` and its map.
        #
        # --- APPLY RANSAC (conceptually) ---
        # If you were computing pose using solvePnP:
        # image_points = np.array([kp2[m.trainIdx].pt for m in good_matches])
        # object_points = np.array([map_3d_points[m.queryIdx] for m in good_matches]) # 3D points from map
        # K_matrix = P2_base[:, :3] # Get intrinsic matrix from P2
        # R_est, T_est, inliers = mock_ransac_pose_estimation(object_points, image_points, K_matrix, None)
        #
        # For simulation, we apply a simulated slight rotation:
        angle_offset = frame_idx * 0.00005 # Smaller dynamic offset for visual
        rot_mat_dynamic = np.array([
            [np.cos(angle_offset), -np.sin(angle_offset), 0],
            [np.sin(angle_offset), np.cos(angle_offset), 0],
            [0, 0, 1]
        ])
        R0_final = R0_base @ rot_mat_dynamic
        # T_final = V2C_base[:, 3] + np.array([0, 0, frame_idx * 0.001]) # Simulate small translation change
        # V2C_final[:3, 3] = T_final # Update V2C_final translation
        print(f"  Simulated dynamic pose for visual SLAM applied (frame {frame_idx}).")


    elif slam_method == "LIDAR_SLAM_SIM":
        print("  Simulating LiDAR SLAM (e.g., LOAM, LIO-SAM).")
        # --- INPUT FOR LIDAR SLAM ---
        # This is where you'd provide the current `lidar_data` (point cloud)
        # and potentially `prev_lidar_data`.
        # Real LiDAR SLAM would perform point cloud registration (e.g., ICP, GICP)
        # to estimate the rigid transformation (R, T) between scans.
        # Example:
        # current_scan = load_lidar_points(lidar_file_path)
        # prev_scan = get_previous_scan_from_map()
        # R_est, T_est = icp_algorithm(prev_scan, current_scan) # This would be an external function/library
        #
        # For simulation, we apply a simulated slight rotation:
        angle_offset = frame_idx * 0.0001
        rot_mat_dynamic = np.array([
            [np.cos(angle_offset), -np.sin(angle_offset), 0],
            [np.sin(angle_offset), np.cos(angle_offset), 0],
            [0, 0, 1]
        ])
        # LiDAR to Camera (V2C) extrinsics would be updated by the estimated pose.
        # Here we'll simulate a dynamic V2C from the LiDAR SLAM's estimated pose.
        # This is a simplification. A real SLAM would give T_world_lidar, and you'd need T_world_camera.
        # For demonstration, we'll perturb V2C_final directly.
        V2C_final_rot = V2C_base[:3,:3] @ rot_mat_dynamic
        V2C_final[:3,:3] = V2C_final_rot
        V2C_final[:3, 3] = V2C_base[:3, 3] + np.array([frame_idx * 0.005, 0, 0]) # Simulate translation along X

        print(f"  Simulated dynamic V2C for LiDAR SLAM applied (frame {frame_idx}).")

    elif slam_method == "VISUAL_INERTIAL_SLAM_SIM":
        print("  Simulating Visual-Inertial SLAM (e.g., VINS-Mono, OKVIS).")
        # --- INPUT FOR VISUAL-INERTIAL SLAM ---
        # This is where you'd provide `image_path` (or `img_rgb`), `imu_data` (acceleration, angular velocity).
        # A real VIO system would fuse these to estimate robust camera pose.
        # The IMU helps with high-frequency motion and correcting drift from visual-only methods.
        # For simulation, we apply a more pronounced dynamic effect:
        angle_offset = frame_idx * 0.0002 # More dynamic
        rot_mat_dynamic = np.array([
            [np.cos(angle_offset), -np.sin(angle_offset), 0],
            [np.sin(angle_offset), np.cos(angle_offset), 0],
            [0, 0, 1]
        ])
        R0_final = R0_base @ rot_mat_dynamic
        V2C_final[:3, 3] = V2C_base[:3, 3] + np.array([0, frame_idx * 0.002, 0]) # Simulate translation along Y
        print(f"  Simulated dynamic pose for Visual-Inertial SLAM applied (frame {frame_idx}).")

    else:
        print(f"  Unknown SLAM method: {slam_method}. Using default static calibration.")

    slam_calib_params = {
        'P2': P2_final,
        'R0': R0_final,
        'V2C': V2C_final
    }
    return slam_calib_params

# --- End of SLAM/Calibration related functions ---


if __name__ == '__main__':
    # Initialize output directory
    output_dir = "D:\\spa\\SFA3D\\sfa\\fused_detection_results_slam_calib_sim"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All fused detection results will be saved to: {output_dir}")

    # Initialize YOLOv8 model
    yolov8_weights_path = "D:\\spa\\SFA3D\\sfa\\models\\yolov8n.pt"
    yolov8_model = YOLO(yolov8_weights_path)
    yolov8_class_names = yolov8_model.names

    # Initialize SFA3D model
    configs = parse_test_configs()
    sfa_model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), f"No file found at {configs.pretrained_path}"
    sfa_model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print(f'Loaded SFA3D weights from {configs.pretrained_path}\n')
    sfa_model = sfa_model.to(device=configs.device)
    sfa_model.eval()

    # Create test dataloader
    test_dataloader = create_test_dataloader(configs)

    # --- Select your desired SLAM method for demonstration ---
    # Choose one of: "KITTI_DATASET_CALIB", "VISUAL_SLAM_SIM", "LIDAR_SLAM_SIM", "VISUAL_INERTIAL_SLAM_SIM"
    chosen_slam_method = "KITTI_DATASET_CALIB" # Defaulting to KITTI dataset calibration for accuracy
    # chosen_slam_method = "VISUAL_SLAM_SIM"
    # chosen_slam_method = "LIDAR_SLAM_SIM"
    # chosen_slam_method = "VISUAL_INERTIAL_SLAM_SIM"


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device).float()

            # Get image info
            img_path = metadatas['img_path'][0]
            img_fn = os.path.basename(img_path)[:-4]

            # Read image
            img_rgb = img_rgbs[0].numpy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            print(f"\nProcessing image: {img_fn}")

            # === Get Calibration using the chosen SLAM method placeholder ===
            # The `lidar_data` parameter is a placeholder. In a real system, you'd load
            # the corresponding LiDAR scan for the current frame here.
            current_frame_calib_params = get_calibration_from_slam_placeholder(
                img_path,
                lidar_data=None, # Replace `None` with actual LiDAR data if using LiDAR SLAM
                slam_method=chosen_slam_method
            )
            print("Using calibration parameters from chosen SLAM/Calibration source for SFA3D 2D projection.")
            # === End SLAM Calibration retrieval ===

            # === YOLOv8 Detection ===
            yolov8_boxes, yolov8_confidences, yolov8_class_ids = yolov8_detect(img_path, yolov8_model)
            print(f"YOLOv8 detected {len(yolov8_boxes)} objects")

            # === SFA3D Detection ===
            t1 = time_synchronized()
            outputs = sfa_model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'],
                                outputs['z_coor'], outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()
            detections = detections[0]

            # Convert SFA3D 3D detections to 2D bounding boxes using SLAM calibration
            sfa3d_boxes_2d, sfa3d_confidences = convert_sfa3d_to_2d_boxes(detections, current_frame_calib_params, img_bgr.shape)
            print(f"SFA3D detected {len(sfa3d_boxes_2d)} objects (converted to 2D using chosen calibration method)")

            # === Fusion and NMS ===
            # Create fused detections
            yolov8_data = (yolov8_boxes, yolov8_confidences, yolov8_class_ids, yolov8_class_names)
            sfa3d_data = (sfa3d_boxes_2d, sfa3d_confidences)

            fused_detections_before_nms = create_fused_detections(
                yolov8_data, sfa3d_data, configs.confidence_threshold
            )
            print(f"Total detections before NMS: {len(fused_detections_before_nms)}")

            # Apply NMS
            fused_detections_after_nms = apply_nms_to_fused_detections(
                fused_detections_before_nms, configs.nms_threshold
            )
            print(f"Total detections after NMS: {len(fused_detections_after_nms)}")

            # === Visualization ===
            # Create different visualization images
            img_before_nms = draw_fused_detections(img_bgr.copy(), fused_detections_before_nms)
            img_after_nms = draw_fused_detections(img_bgr.copy(), fused_detections_after_nms)

            # Create summary
            summary_img = create_detection_summary(fused_detections_before_nms, fused_detections_after_nms)

            # Combine images for display
            combined_top = np.hstack([
                cv2.resize(img_before_nms, (640, 480)),
                cv2.resize(img_after_nms, (640, 480))
            ])

            # Add labels
            cv2.putText(combined_top, "Before NMS", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_top, "After NMS", (690, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Resize summary to match width
            summary_resized = cv2.resize(summary_img, (1280, 200))

            # Final combined image
            final_combined = np.vstack([combined_top, summary_resized])

            # === Save Results ===
            output_path = os.path.join(output_dir, f"{img_fn}_fused_detection_{chosen_slam_method}.jpg")
            cv2.imwrite(output_path, final_combined)
            print(f"Saved fused detection result: {output_path}")

            # === Display ===
            cv2.namedWindow("Fused Object Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Fused Object Detection", final_combined)

            # Auto-resize window (using tkinter for best screen resolution detection)
            screen_res = 1920, 1080
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_res = root.winfo_screenwidth(), root.winfo_screenheight()
                root.destroy()
            except ImportError:
                print("tkinter not found, cannot auto-resize window optimally.")
                pass # Continue without tkinter for non-GUI environments

            win_width = min(screen_res[0] - 100, 1280)
            win_height = min(screen_res[1] - 100, 800)
            cv2.resizeWindow("Fused Object Detection", win_width, win_height)

            print(f'\nProcessing time for SFA3D (inference only): {(t2 - t1) * 1000:.1f}ms')
            print(f'Detection Summary:')
            print(f'  - YOLOv8: {len(yolov8_boxes)} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "YOLOv8")}')
            print(f'  - SFA3D: {len(sfa3d_boxes_2d)} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "SFA3D")}')
            print(f'  - Total: {len(fused_detections_before_nms)} -> {len(fused_detections_after_nms)}')
            print('\n[INFO] Press n for next sample | Press Esc to quit\n')

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Esc key
                break
            elif key == ord('c'):
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print(f"\nAll results saved to: {output_dir}")