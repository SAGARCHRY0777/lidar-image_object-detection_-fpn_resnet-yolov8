import argparse, sys, os, time, warnings, cv2, numpy as np, torch, matplotlib.pyplot as plt, matplotlib.cm as cm
from ultralytics import YOLO
warnings.filterwarnings("ignore", category=UserWarning)
from easydict import EasyDict as edict

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.sys.path.append(src_dir)

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


# --- Mock RANSAC Function - Now uses cv2.solvePnP with RANSAC ---
def mock_ransac_pose_estimation(object_points, image_points, K_matrix, dist_coeffs):
    """
    RANSAC-based camera pose estimation using cv2.solvePnP.
    Args:
        object_points (np.array): 3D points in the world coordinate system.
        image_points (np.array): 2D projections of the 3D points in the image.
        K_matrix (np.array): Intrinsic camera matrix (3x3).
        dist_coeffs (np.array): Distortion coefficients (e.g., k1, k2, p1, p2, k3).
    Returns:
        tuple: (rotation vector, translation vector, inliers)
    """
    # Ensure inputs are float32 and correct shape
    object_points = np.ascontiguousarray(object_points, dtype=np.float32)
    image_points = np.ascontiguousarray(image_points, dtype=np.float32)
    K_matrix = np.ascontiguousarray(K_matrix, dtype=np.float32)
    dist_coeffs = np.ascontiguousarray(dist_coeffs, dtype=np.float32)


    # Minimum points for solvePnP
    if len(object_points) < 4:
        print("    Not enough points for RANSAC pose estimation. Returning identity pose.")
        return np.zeros((3,1)), np.zeros((3,1)), None

    print("    Applying RANSAC (cv2.SOLVEPNP_EPNP + RANSAC) for pose estimation...")

    # Initial guess for rvec and tvec (can be zeros or from previous frame for smoother tracking)
    # Using None for initial guess to let RANSAC find the solution from scratch
    # If tracking, you'd pass a previous frame's R, T
    rvec_initial = None
    tvec_initial = None

    # Using SOLVEPNP_EPNP with RANSAC
    # reprojectionError: Maximum allowed reprojection error to treat a point pair as an inlier.
    # iterationsCount: The number of RANSAC iterations.
    # confidence: The confidence level for the RANSAC algorithm.
    try:
        success, rvec, tvec, inliers = cv2.solvePnP(
            object_points,
            image_points,
            K_matrix,
            dist_coeffs,
            rvec=rvec_initial,
            tvec=tvec_initial,
            useExtrinsicGuess=False, # Set to True if using initial guess
            flags=cv2.SOLVEPNP_EPNP, # Efficient PnP method
            reprojectionError=8.0, # A typical value, adjust based on noise
            iterationsCount=1000,
            confidence=0.99
        )
    except cv2.error as e:
        print(f"    cv2.solvePnP encountered an error: {e}")
        success = False # Explicitly set success to False

    if not success:
        print("    cv2.solvePnP with RANSAC failed to find a solution. Returning identity pose.")
        return np.zeros((3,1)), np.zeros((3,1)), None

    print(f"    RANSAC found {len(inliers) if inliers is not None else 0} inliers.")
    return rvec, tvec, inliers


# --- Unified Placeholder for Calibration Retrieval from various "SLAM" methods ---
def get_calibration_from_slam_placeholder(image_path, img_rgb, lidar_data=None, slam_method="KITTI_DATASET_CALIB"):
    """
    Unified placeholder function to simulate getting calibration from different "SLAM" sources.
    This demonstrates where you would integrate outputs from various SLAM algorithms
    or directly read pre-existing calibration data.

    Args:
        image_path (str): Path to the current image frame. Used for frame indexing.
        img_rgb (np.array): The current RGB image data. Needed for feature extraction in Visual SLAM.
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
    frame_idx = int(os.path.basename(image_path).split('.')[0])
    calib_file_path = image_path.replace('image_2', 'calib').replace('.png', '.txt') # Adjust based on your dataset structure

    K_matrix = P2_base[:, :3] # Intrinsic camera matrix from P2
    dist_coeffs = np.zeros((4, 1), dtype=np.float32) # Assuming no distortion for simplicity, or load from calibration

    if slam_method == "KITTI_DATASET_CALIB":
        print(f"    Loading calibration from KITTI dataset file: {calib_file_path}")
        # In a real KITTI dataset integration, you'd parse this file:
        # For this demonstration, we'll just use the base static values as if
        # they were loaded "perfectly" from a static calibration file.
        pass

    elif slam_method == "VISUAL_SLAM_SIM":
        print("    Simulating Visual SLAM (e.g., ORB-SLAM3, VINS-Fusion) with RANSAC-based pose estimation.")

        # --- SIMULATE 3D-2D CORRESPONDENCES ---
        # For a more robust simulation, generate a grid of 3D points
        # in front of the camera and project them. This ensures non-coplanar points.
        x_coords = np.linspace(-5, 5, 5)
        y_coords = np.linspace(-2, 2, 3)
        z_depth = np.linspace(8, 15, 3) # Points at different depths

        # Create a meshgrid of 3D points
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_depth)
        simulated_object_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T.astype(np.float32)

        # Simulate a dynamic ground truth pose (rvec_gt, tvec_gt) for the current frame
        sim_rvec_gt = np.array([0.005 * np.sin(frame_idx * 0.01),
                                0.003 * np.cos(frame_idx * 0.01),
                                0.001 * frame_idx], dtype=np.float32)
        sim_tvec_gt = np.array([0.01 * np.sin(frame_idx * 0.005),
                                0.005 * np.cos(frame_idx * 0.005),
                                0.05 * frame_idx + 10.0], dtype=np.float32) # Moving away

        # Project 3D points to 2D using the simulated ground truth pose
        simulated_image_points_perfect, _ = cv2.projectPoints(
            simulated_object_points, sim_rvec_gt, sim_tvec_gt, K_matrix, dist_coeffs
        )
        simulated_image_points_perfect = simulated_image_points_perfect.reshape(-1, 2)

        # Add some noise to simulate imperfect feature detection/matching
        noise = np.random.normal(0, 1.5, simulated_image_points_perfect.shape).astype(np.float32)
        simulated_image_points_noisy = simulated_image_points_perfect + noise

        # Filter out points that fall outside the image boundaries after adding noise
        h, w, _ = img_rgb.shape
        valid_indices = (simulated_image_points_noisy[:, 0] >= 0) & \
                        (simulated_image_points_noisy[:, 0] < w) & \
                        (simulated_image_points_noisy[:, 1] >= 0) & \
                        (simulated_image_points_noisy[:, 1] < h)

        simulated_object_points_filtered = simulated_object_points[valid_indices]
        simulated_image_points_noisy_filtered = simulated_image_points_noisy[valid_indices]

        # Call the RANSAC pose estimation
        rvec_est, tvec_est, inliers = mock_ransac_pose_estimation(
            simulated_object_points_filtered, simulated_image_points_noisy_filtered, K_matrix, dist_coeffs
        )

        # Convert rvec to rotation matrix
        R_est, _ = cv2.Rodrigues(rvec_est)

        # In a real SLAM, (R_est, tvec_est) would be the camera's pose relative to the world frame.
        # You then combine this with your static sensor extrinsics (V2C, R0)
        # to get the current frame's full calibration.
        # This part is highly dependent on your SLAM's output convention.

        # Here, we will apply the estimated camera pose directly to the *camera-related* matrices.
        # If (R_est, tvec_est) is the camera's pose from world to camera (P_cam = R_est * P_world + t_est):
        # Then, the world-to-camera extrinsic matrix is [R_est | t_est].
        # For simplicity, we assume R_est directly updates R0 and tvec_est updates V2C's translation.
        # This is an approximation for simulation.

        # Update R0_final: The estimated camera orientation influences the rectified rotation.
        # A more precise way would be to get the full T_world_camera matrix from SLAM
        # and then decompose it or compute T_world_rectified_camera = T_rect_camera * T_camera_world.
        R0_final = R_est @ R0_base # Apply the estimated rotation to the baseline rectified rotation

        # Update V2C_final: This is the LiDAR to Camera transform.
        # If (R_est, tvec_est) represents the camera's pose, and you know the static
        # T_lidar_camera (which is V2C_base), you would compose transformations.
        # Here, as a simulation, we perturb V2C_base's translation directly by tvec_est
        # and keep its rotation relatively similar, or use the estimated rotation from R_est.
        # This is a highly simplified integration.
        # Real VIO would give T_world_camera. To get V2C, you'd need T_world_lidar and T_camera_lidar (inverse of V2C_base).
        # T_world_lidar = T_world_camera @ T_camera_lidar
        # T_camera_lidar = np.linalg.inv(V2C_base_full_matrix)

        # For this simulation, we'll try to update V2C's translation.
        # Assuming tvec_est gives a camera translation in world coordinates, we use it to perturb V2C_base's translation.
        # This is a conceptual update.
        # More correctly, if (R_est, tvec_est) represents T_camera_world (transforming world points to camera frame),
        # then the inverse is T_world_camera.
        # Let's say R_slam_world_to_cam, T_slam_world_to_cam are the outputs of mock_ransac_pose_estimation
        # We want to adjust V2C_base (T_lidar_to_camera_static) to reflect the new camera position relative to LiDAR.
        # This is usually done by tracking the vehicle's pose in a global frame.
        # For demonstration, we simply apply a translation offset based on the estimated `tvec_est`.
        V2C_final_translation_offset = tvec_est.flatten() # Directly use the estimated translation as an offset
        # Ensure it's applied correctly; this might need to be negated or transformed depending on convention.
        V2C_final[:3, 3] = V2C_base[:3, 3] + V2C_final_translation_offset

        print(f"    Estimated pose (rvec, tvec) updated for visual SLAM (frame {frame_idx}).")

    elif slam_method == "LIDAR_SLAM_SIM":
        print("    Simulating LiDAR SLAM (e.g., LOAM, LIO-SAM).")
        angle_offset = frame_idx * 0.0001
        rot_mat_dynamic = np.array([
            [np.cos(angle_offset), -np.sin(angle_offset), 0],
            [np.sin(angle_offset), np.cos(angle_offset), 0],
            [0, 0, 1]
        ])
        V2C_final_rot = V2C_base[:3,:3] @ rot_mat_dynamic
        V2C_final[:3,:3] = V2C_final_rot
        V2C_final[:3, 3] = V2C_base[:3, 3] + np.array([frame_idx * 0.005, 0, 0]) # Simulate translation along X
        print(f"    Simulated dynamic V2C for LiDAR SLAM applied (frame {frame_idx}).")

    elif slam_method == "VISUAL_INERTIAL_SLAM_SIM":
        print("    Simulating Visual-Inertial SLAM (e.g., VINS-Mono, OKVIS).")
        angle_offset = frame_idx * 0.0002 # More dynamic
        rot_mat_dynamic = np.array([
            [np.cos(angle_offset), -np.sin(angle_offset), 0],
            [np.sin(angle_offset), np.cos(angle_offset), 0],
            [0, 0, 1]
        ])
        R0_final = R0_base @ rot_mat_dynamic
        V2C_final[:3, 3] = V2C_base[:3, 3] + np.array([0, frame_idx * 0.002, 0]) # Simulate translation along Y
        print(f"    Simulated dynamic pose for Visual-Inertial SLAM applied (frame {frame_idx}).")

    else:
        print(f"    Unknown SLAM method: {slam_method}. Using default static calibration.")

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
    chosen_slam_method = "VISUAL_SLAM_SIM" # Changed to demonstrate RANSAC
    # chosen_slam_method = "KITTI_DATASET_CALIB"
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
            current_frame_calib_params = get_calibration_from_slam_placeholder(
                img_path,
                img_rgb=img_rgb, # Pass the actual image data
                lidar_data=None, # Replace `None` with actual LiDAR data if using LiDAR SLAM
                slam_method=chosen_slam_method
            )
            print(f"Calibration parameters for {chosen_slam_method}:")
            print(f"    P2: {current_frame_calib_params['P2']}")    
            print(f"    R0: {current_frame_calib_params['R0']}")
            print(f"    V2C: {current_frame_calib_params['V2C']}")
            # Use these calibration parameters for SFA3D 2D projection
            
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
            print(f'   - YOLOv8: {len(yolov8_boxes)} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "YOLOv8")}')
            print(f'   - SFA3D: {len(sfa3d_boxes_2d)} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "SFA3D")}')
            print(f'   - Total: {len(fused_detections_before_nms)} -> {len(fused_detections_after_nms)}')
            print('\n[INFO] Press n for next sample | Press Esc to quit\n')

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Esc key
                break
            elif key == ord('c'):
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print(f"\nAll results saved to: {output_dir}")