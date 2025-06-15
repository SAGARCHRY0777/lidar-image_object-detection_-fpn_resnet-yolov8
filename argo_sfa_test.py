# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------------
# Modified for Argoverse dataset testing with SFA model
# Based on original SFA testing script
-----------------------------------------------------------------------------------
"""

import argparse
import sys
import os
import time
import warnings
import json
import math

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import open3d as o3d # Keep this if you use it directly, though argoverse_data_utils handles it
from tqdm import tqdm

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions # Removed convert_det_to_real_values as it's KITTI specific
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf # This might be renamed to argo_config or something similar if you have one
# from data_process.transformation import lidar_to_camera_box # Likely not needed if Argoverse specific transform is used
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes # show_rgb_image_with_boxes might not be used here


# Import the new Argoverse data utilities
from data_process.argoverse_data_utils_copy import (
    ArgoverseCalibration,
    makeBVFeature,
    load_lidar_data,
    get_argoverse_frame_paths,
    create_3d_bbox_corners,
    draw_3d_bbox_lines
)

# You commented this out in the previous step; if you intend to use a DataLoader, uncomment it
# from data_process.argoverse_dataloader import create_test_dataloader


def convert_detections_to_kitti_format(detections, boundary, discretization):
    """
    Convert SFA detections to KITTI-like format for visualization purposes.
    The output format matches: [class, x, y, z, h, w, l, yaw, score]
    where x,y,z,h,w,l,yaw are in the **LiDAR coordinate system (BEV space)**
    as returned by SFA's output, then converted to real-world meters.
    """
    if len(detections) == 0:
        return np.array([])
        
    kitti_dets = []
    for det in detections:
        # SFA detection format: [x_grid, y_grid, z_height_rel, dim_x, dim_y, dim_z, yaw_sin, yaw_cos, score, class]
        # Make sure the `det` array has enough elements based on your SFA output.
        # Assuming SFA outputs (x_grid, y_grid, z_rel, dim_x, dim_y, dim_z, yaw, score, class) after decoding
        if len(det) >= 9: # Check for expected number of elements
            # Convert from grid coordinates to metric coordinates
            # x_grid is along the BEV map's height (configs.maxX - x_real) / discretization
            # y_grid is along the BEV map's width (y_real - configs.minY) / discretization
            
            # SFA's x_grid is related to real_X (length direction along ego-car)
            # SFA's y_grid is related to real_Y (width direction perpendicular to ego-car)
            
            # Convert grid x_idx to real_X
            # x_grid = (boundary["maxX"] - real_X) / discretization
            # real_X = boundary["maxX"] - (x_grid * discretization)
            real_x = boundary["maxX"] - (det[0] * discretization)

            # Convert grid y_idx to real_Y
            # y_grid = (real_Y - boundary["minY"]) / discretization
            # real_Y = y_grid * discretization + boundary["minY"]
            real_y = det[1] * discretization + boundary["minY"]

            # Z coordinate (height). SFA's z_coor is relative to minZ.
            real_z = det[2] + boundary["minZ"]
            
            # Dimensions (usually L, W, H or H, W, L from model)
            # Let's assume det[3], det[4], det[5] are length, width, height (L, W, H)
            # based on common KITTI format for visualization.
            # You must verify your model's actual output order for dimensions.
            # SFA typically outputs (dim_x, dim_y, dim_z) which might map to (L, W, H) or (H, W, L).
            # For KITTI-like, it's (h, w, l). Let's assume dim_x=L, dim_y=W, dim_z=H for now.
            # So, to match KITTI order (h, w, l):
            h_det, w_det, l_det = det[5], det[4], det[3] # Assuming SFA output order was (L,W,H) or needs reordering
                                                         # Adjust this mapping (det[3],det[4],det[5]) as per your SFA model's output
                                                         # H is often det[5] (z dimension), W is det[4] (y dimension), L is det[3] (x dimension)
            
            # Yaw angle
            # SFA outputs sin and cos of yaw.
            yaw_rad = np.arctan2(det[6], det[7]) # arctan2(sin, cos)
            
            score = det[8]
            cls = int(det[9]) if len(det) > 9 else 0 # Class ID

            kitti_dets.append([cls, real_x, real_y, real_z, h_det, w_det, l_det, yaw_rad, score])
        else:
            print(f"Warning: Detection has unexpected length: {len(det)}. Expected at least 9. Skipping: {det}")
    
    return np.array(kitti_dets) if kitti_dets else np.array([])


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Argoverse Testing config for SFA')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str, required=True, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    
    # === ADD THESE ARGUMENTS ===
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to Argoverse dataset sequence directory (e.g., .../sample/c6911883-...)')
    parser.add_argument('--camera_name', type=str, default='ring_front_center',
                        help='Camera name for Argoverse dataset (e.g., ring_front_center)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting frame index in the sequence')
    parser.add_argument('--num_frames', type=int, default=50,
                        help='Number of frames to process from start_idx')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='GPU index to use (e.g., 0, 1)')
    # ===========================

    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K objects to detect')
    parser.add_argument('--batch_size', type=int, default=1, # Often 1 for testing
                        help='mini-batch size (default: 1)')
    parser.add_argument('--peak_thresh', type=float, default=0.2,
                        help='Threshold for peak detection in heatmap')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18', metavar='PATH',
                        help='the video filename if the output format is video')
    parser.add_argument('--output-width', type=int, default=1280, # Increased for better visualization
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False # Assuming single GPU testing
    configs.device = torch.device(f'cuda:{configs.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    # Model input/output sizes
    configs.input_size = (608, 608) # Input BEV map size for the model
    configs.hm_size = (152, 152) # Heatmap size
    configs.down_ratio = 4 # Downsampling ratio from input_size to hm_size
    configs.max_objects = 50 # Max objects per frame

    configs.imagenet_pretrained = False # Not relevant for inference, but usually set during training
    configs.head_conv = 64 # Number of convolutions in the detection head
    configs.num_classes = 3 # Example: Car, Pedestrian, Cyclist. Adjust for your Argoverse classes.
                           # If using original SFA KITTI model, this is 3.
    configs.num_center_offset = 2 # x, y offset in heatmap
    configs.num_z = 1 # Z-coordinate of object center
    configs.num_dim = 3 # Dimensions (length, width, height)
    configs.num_direction = 2 # sin, cos of yaw angle

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }
    configs.num_input_features = 4 # BEV channels: height, density, intensity, occupancy

    # BEV parameters for Argoverse - these must match your training setup
    # These values are crucial for transforming LiDAR points to BEV grid and back
    configs.boundary = {"minX": 0, "maxX": 50, "minY": -25, "maxY": 25, "minZ": -2.73, "maxZ": 1.27}
    configs.BEV_HEIGHT = 608 # The target height of the BEV map
    configs.BEV_WIDTH = 608  # The target width of the BEV map
    # Discretization is (range_X / pixels_X) or (range_Y / pixels_Y)
    configs.discretization = (configs.boundary["maxX"] - configs.boundary["minX"]) / configs.BEV_HEIGHT # Assuming square pixels and input_size

    if configs.save_test_output:
        # Save results in a subdirectory specific to this run
        configs.results_dir = os.path.join(os.path.dirname(__file__), 'results_argo', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), f"No pretrained model file found at: {configs.pretrained_path}"
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=configs.device))
    print(f'Loaded weights from {configs.pretrained_path}\n')

    model = model.to(device=configs.device)
    out_cap = None # For video output
    model.eval() # Set model to evaluation mode

    # Initialize ArgoverseCalibration once for the entire sequence
    # This class handles coordinate transformations (LiDAR to Camera)
    calib_filepath = os.path.join(configs.dataset_dir, 'vehicle_calibration_info.json')
    calibration_obj = ArgoverseCalibration(calib_filepath, configs.camera_name)

    print(f"Processing {configs.num_frames} frames from '{configs.dataset_dir}' starting at index {configs.start_idx}...")
    
    with torch.no_grad(): # Disable gradient computation for inference
        for frame_idx in tqdm(range(configs.start_idx, configs.start_idx + configs.num_frames)):
            try:
                # Retrieve paths for the current frame
                frame_data = get_argoverse_frame_paths(configs.dataset_dir, frame_idx, configs.camera_name)
                if frame_data is None:
                    print(f"Skipping frame {frame_idx}: unable to find necessary files.")
                    continue

                # Load image (for visualization)
                image = cv2.imread(frame_data['image_path'])
                if image is None:
                    print(f"Failed to load image: {frame_data['image_path']}")
                    continue

                # Load LiDAR data
                points = load_lidar_data(frame_data['lidar_path'])
                if points.shape[0] == 0:
                    print(f"No LiDAR points loaded for frame {frame_idx}.")
                    continue

                # Generate BEV feature map from LiDAR points
                bev_feature = makeBVFeature(points, configs.discretization, configs.boundary)
                
                if np.all(bev_feature == 0): # Check if BEV map is completely empty
                    print(f"Empty BEV feature map for frame {frame_idx}.")
                    continue

                # Prepare BEV feature for model input (add batch dimension)
                input_bev_maps = torch.from_numpy(bev_feature).unsqueeze(0).float().to(configs.device)
                
                t1 = time_synchronized() # Start inference time tracking
                outputs = model(input_bev_maps) # Forward pass
                outputs['hm_cen'] = _sigmoid(outputs['hm_cen']) # Apply sigmoid to heatmap
                outputs['cen_offset'] = _sigmoid(outputs['cen_offset']) # Apply sigmoid to center offset

                # Decode raw model outputs into detectable objects
                detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], 
                                    outputs['z_coor'], outputs['dim'], K=configs.K)
                detections = detections.cpu().numpy().astype(np.float32)
                # Post-process detections (NMS, filtering by score threshold)
                detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
                t2 = time_synchronized() # End inference time tracking

                # Detections are returned as a list of arrays (one per batch item)
                # Since batch_size is 1, take the first element
                current_detections = detections[0] if detections[0] is not None else []
                
                # --- Visualization ---
                # Prepare BEV map for display
                # Select first 3 channels (H, D, I) and transpose to HWC for OpenCV
                bev_map_display = (bev_feature[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
                bev_map_display = cv2.resize(bev_map_display, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) # Resize to display dimensions
                bev_map_display = draw_predictions(bev_map_display, current_detections.copy(), configs.num_classes)
                bev_map_display = cv2.rotate(bev_map_display, cv2.ROTATE_180) # Rotate for correct orientation

                # Original image for display (copy to draw on)
                img_bgr = image.copy()

                # Convert SFA's internal detection format to a KITTI-like format
                # This makes it easier to use existing 3D box drawing functions
                kitti_style_detections = convert_detections_to_kitti_format(
                    current_detections, configs.boundary, configs.discretization
                )

                # Project 3D bounding boxes from LiDAR to camera image
                if len(kitti_style_detections) > 0:
                    for det_kitti in kitti_style_detections:
                        # det_kitti: [cls, x_lidar, y_lidar, z_lidar, h_lidar, w_lidar, l_lidar, yaw_lidar_rad, score]
                        cls_id, x_l, y_l, z_l, h_l, w_l, l_l, yaw_l, score = det_kitti
                        
                        # create_3d_bbox_corners function expects: dimensions [L, W, H], location [x, y, z], rotation_y
                        # The location [x,y,z] should be the bottom-center of the box in the frame of reference
                        # where the rotation_y is applied. If yaw_l is a global LiDAR yaw,
                        # and create_3d_bbox_corners assumes camera-aligned rotation_y,
                        # this step requires careful transformation.
                        # For simplicity, if your SFA model outputs are aligned with KITTI's camera coord system
                        # this might work directly, but usually LiDAR coords need to be transformed.
                        
                        # Assuming create_3d_bbox_corners generates corners in LiDAR coordinates,
                        # then we project them using calibration_obj.project_lidar_to_image
                        
                        # Argoverse defines dimensions as (length, width, height) in meters.
                        # SFA often outputs (dim_x, dim_y, dim_z) corresponding to (L, W, H) or (H, W, L).
                        # Ensure the order (l_l, w_l, h_l) matches your model's output interpretation.
                        
                        # NOTE: The create_3d_bbox_corners function as written assumes a KITTI-like camera coordinate system
                        # where x is right, y is down, z is forward, and rotation_y is around camera's Y axis.
                        # Your SFA outputs are typically in a BEV/LiDAR coordinate system.
                        # A direct call `create_3d_bbox_corners(dimensions=[l_l, w_l, h_l], location=[x_l, y_l, z_l], rotation_y=yaw_l)`
                        # implies these are already in camera coordinates, which is likely incorrect.
                        # You would typically transform the 3D center and dimensions from LiDAR to camera first,
                        # then calculate rotation_y in the camera frame.
                        # For now, let's assume `create_3d_bbox_corners` works with LiDAR inputs and produces
                        # corners that can be directly transformed by `project_lidar_to_image`.
                        
                        # Important: The yaw_l from SFA is typically in LiDAR/BEV space.
                        # For 3D box projection in the image, we often need the 3D box's
                        # orientation relative to the CAMERA frame's Y-axis (like KITTI's alpha/ry).
                        # This transformation is non-trivial and depends on your Argoverse calibration setup.
                        
                        # For a robust solution, you'd convert the full 3D LiDAR box (center + dimensions + yaw)
                        # into a 3D camera box (center + dimensions + rotation_y in camera frame)
                        # using `calibration_obj.get_T_l2c()`.
                        # Since `create_3d_bbox_corners` expects `rotation_y`,
                        # a proper conversion function `lidar_box_to_camera_box` would be needed.
                        # As a simplification for demonstration, we will assume `create_3d_bbox_corners`
                        # can handle lidar-like inputs for generating raw 3D corners which are then transformed.

                        # Generate 8 corners of the 3D box in LiDAR coordinates
                        # The order of l,w,h depends on your model's definition of dim_x, dim_y, dim_z.
                        # Common: dim_x=length, dim_y=width, dim_z=height.
                        # So, dimensions=[l_l, w_l, h_l] seems reasonable if SFA's dim order is LWH.
                        # If your SFA output is (H,W,L) or (W,L,H), adjust the indices here.
                        # For KITTI visualization it's (h, w, l) order for dimensions.
                        # create_3d_bbox_corners expects [length, width, height]
                        box_dims_for_corners = [l_l, w_l, h_l]
                        box_loc_for_corners = [x_l, y_l, z_l]
                        box_yaw_for_corners = yaw_l # This is the LiDAR yaw.

                        corners_3d_lidar = create_3d_bbox_corners(
                            dimensions=box_dims_for_corners,
                            location=box_loc_for_corners,
                            rotation_y=box_yaw_for_corners # This function ideally expects camera's rotation_y.
                                                          # If this is LiDAR yaw, the output corners will be in LiDAR space.
                        )
                        
                        # Project the 3D corners (currently in LiDAR coordinates) to 2D image plane
                        projected_corners_2d = calibration_obj.project_lidar_to_image(corners_3d_lidar)
                        draw_3d_bbox_lines(img_bgr, projected_corners_2d) # Draw on the BGR image

                # Merge RGB image with BEV map for final output
                out_img = merge_rgb_to_bev(img_bgr, bev_map_display, output_width=configs.output_width)

                print(f'\tFrame {frame_idx}: {len(current_detections)} detections, '
                      f'inference time: {(t2 - t1)*1000:.1f}ms, '
                      f'speed: {1/(t2-t1):.2f} FPS')
                
                # Save output if configured
                if configs.save_test_output:
                    if configs.output_format == 'image':
                        img_fn = f'frame_{frame_idx:06d}'
                        cv2.imwrite(os.path.join(configs.results_dir, f'{img_fn}.jpg'), out_img)
                    elif configs.output_format == 'video':
                        if out_cap is None:
                            out_cap_h, out_cap_w = out_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Codec for video
                            out_cap = cv2.VideoWriter(
                                os.path.join(configs.results_dir, f'{configs.output_video_fn}.avi'),
                                fourcc, 30, (out_cap_w, out_cap_h)) # 30 FPS
                        out_cap.write(out_img)

                # Display the image
                cv2.imshow('Argoverse SFA Detection', out_img)
                print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
                key = cv2.waitKey(1) & 0xFF # Wait for 1ms for key press
                if key == 27:  # Esc key
                    break # Exit loop
                elif key == ord('n'):
                    continue # Continue to next frame

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                continue

    # Release video writer if it was initialized
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")