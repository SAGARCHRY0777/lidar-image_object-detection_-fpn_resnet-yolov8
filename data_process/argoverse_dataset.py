# argoverse_dataset.py
import sys
import os
import math
from builtins import int
import glob
from typing import NamedTuple, Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import json

# For quaternion to Euler conversion, typically from scipy.spatial.transform
from scipy.spatial.transform import Rotation as R

# Assume this is relative to your SFA root. Adjust if needed.
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.argoverse_config as cnf
from data_process.argoverse_data_utils_copy import ArgoverseCalibration, get_filtered_lidar, makeBEVMap, compute_radius, gen_hm_radius
# from data_process import transformation # You might need custom transformations for Argoverse if they differ significantly


class ArgoverseDataset(Dataset):
    def __init__(self, configs, mode='train', lidar_aug=None, hflip_prob=None, num_samples=None,
                 # Specify a specific camera for visualization if needed, e.g., 'ring_front_center'
                 target_camera='ring_front_center'): # target_camera is still used for image folder name
        self.dataset_dir = configs.dataset_dir # Base directory for the dataset
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')

        self.lidar_aug = lidar_aug
        self.hflip_prob = hflip_prob
        self.target_camera = target_camera # Camera to load for RGB image if multiple exist

        # Define paths to your specific data folders
        self.lidar_data_dir = os.path.join(self.dataset_dir, "samplefile", "lidar")
        self.image_data_dir = os.path.join(self.dataset_dir, "samplefile", self.target_camera)
        # Assuming annotations are in a structure similar to Argoverse, e.g., 'annotations/track_label.json'
        # If your annotations are structured differently, you'll need to adjust get_labels.
        # For simplicity, assuming a single annotations file per sequence or a direct mapping.
        # For sample dataset, it's often a single JSON file.
        self.annotation_file = os.path.join(self.dataset_dir, "annotations", "track_label.json") # Adjust if different

        # Get sorted lists of file paths
        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_data_dir, "*.bin"))) # Assuming .bin for lidar
        self.image_files = sorted(glob.glob(os.path.join(self.image_data_dir, "*.jpg"))) # Assuming .jpg for images

        # Ensure the number of LiDAR and image files match
        if len(self.lidar_files) != len(self.image_files):
            print(f"Warning: Number of LiDAR files ({len(self.lidar_files)}) does not match number of image files ({len(self.image_files)}).")
            # You stated "no need to check the length of file", but for paired loading, they usually should match.
            # We'll proceed with the minimum length to avoid index errors.
            self.num_samples = min(len(self.lidar_files), len(self.image_files))
        else:
            self.num_samples = len(self.lidar_files)
        
        # Load all annotations once if they are in a single file
        self.all_annotations = self._load_all_annotations()

        print(f"Loaded {self.num_samples} Argoverse samples from local directories in {self.mode} mode.")

    def __len__(self):
        return self.num_samples

    def _load_all_annotations(self):
        """
        Loads all annotations from the track_label.json file.
        This assumes a single JSON file containing annotations for all frames, indexed by timestamp.
        """
        if not os.path.exists(self.annotation_file):
            print(f"Warning: Annotation file not found at {self.annotation_file}. Labels will be empty.")
            return {}
        with open(self.annotation_file, 'r') as f:
            # Assuming the JSON is a dictionary where keys are timestamps
            all_annotations = json.load(f)
        return all_annotations

    def get_labels(self, timestamp: str) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Loads object labels for a specific timestamp from the pre-loaded annotations.
        """
        annotations_for_frame = self.all_annotations.get(timestamp)
        if annotations_for_frame:
            return annotations_for_frame['track_label_list'], True
        return [], False


    def __getitem__(self, index):
        if index >= self.num_samples:
            raise IndexError(f"Index {index} out of bounds for dataset of size {self.num_samples}")

        lidar_filepath = self.lidar_files[index]
        image_filepath = self.image_files[index]
        
        # Extract timestamp from filename (e.g., 315974052820626000.jpg -> 315974052820626000)
        timestamp = os.path.splitext(os.path.basename(image_filepath))[0]

        # Load LiDAR data (x, y, z, intensity)
        lidar_data = self.get_lidar_data(lidar_filepath)

        # Load image data
        img = self.get_image(image_filepath) # This returns RGB image

        # Initialize calibration for this specific frame if needed, or load once globally if static
        # For simplicity, assuming a single calibration file relevant to the dataset.
        # You'd typically have log-specific calibration.
        # This path might need to be dynamic based on the log ID.
        # For the sample dataset, often there's one vehicle_calibration_info.json in the top-level folder.
        calibration_file_path = os.path.join(self.dataset_dir, "vehicle_calibration_info.json")
        if not os.path.exists(calibration_file_path):
            raise FileNotFoundError(f"Calibration file not found at {calibration_file_path}. "
                                    "Please ensure 'vehicle_calibration_info.json' exists in your dataset root.")
        calib = ArgoverseCalibration(calibration_file_path, target_camera=self.target_camera)


        # Process labels
        labels_raw, has_labels = self.get_labels(timestamp)
        labels = []
        if has_labels:
            for obj_dict in labels_raw:
                obj_type = obj_dict['object_type']
                if obj_type not in cnf.CLASS_NAME_TO_ID:
                    continue
                cls_id = cnf.CLASS_NAME_TO_ID[obj_type]

                x, y, z = obj_dict['translation']
                h, w, l = obj_dict['height'], obj_dict['width'], obj_dict['length']
                
                # Argoverse rotation coefficients are (qx, qy, qz, qw)
                # scipy.spatial.transform.Rotation.from_quat expects (x, y, z, w)
                rot_quat = np.array([obj_dict['rotation'][0], obj_dict['rotation'][1], obj_dict['rotation'][2], obj_dict['rotation'][3]])
                
                r = R.from_quat(rot_quat)
                euler_angles = r.as_euler('xyz', degrees=False)
                yaw = euler_angles[2] # Assuming yaw is around Z-axis

                labels.append([cls_id, x, y, z, h, w, l, yaw])
            labels = np.array(labels, dtype=np.float32)
        else:
            labels = np.zeros((0, 8), dtype=np.float32) # Return empty array if no labels

        # Filter LiDAR data by boundary defined in config
        lidar_data_filtered = get_filtered_lidar(lidar_data, cnf.boundary)
        
        # Prepare BEV map
        bev_map = makeBEVMap(lidar_data_filtered, cnf.boundary, cnf.DISCRETIZATION)

        # For training, you'd generate heatmaps and regression targets here.
        # For now, let's just return the raw data and calibration for visualization.
        # If no labels, `labels` will be (0,8) and that's fine.

        ret = {
            'img': img,
            'lidar_data': lidar_data_filtered,
            'bev_map': bev_map,
            'labels': labels,
            'calib': calib, # Pass the calibration object
            'metadata': {'timestamp': timestamp, 'img_path': image_filepath, 'lidar_path': lidar_filepath}
        }
        return ret

    def get_lidar_data(self, lidar_filepath: str) -> np.ndarray:
        """Loads LiDAR data from .bin file and returns Nx4 array (x,y,z,intensity)."""
        # Argoverse .bin files are typically float32.
        # Format: (x, y, z, intensity)
        lidar_pts = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 4)
        # Argoverse LiDAR might have additional channels depending on the sensor.
        # Ensure it's (x, y, z, intensity) for consistency if more exist.
        
        # Argoverse LiDAR points are often in the ego-vehicle frame.
        # No transformation needed here, as filtering and BEV map creation
        # expect points in ego-vehicle frame.
        return lidar_pts

    def get_image(self, image_filepath: str) -> np.ndarray:
        """Loads image directly from a file (e.g., .jpg)."""
        img = cv2.imread(image_filepath)
        if img is None:
            raise FileNotFoundError(f"Image not found or could not be loaded: {image_filepath}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
        return img_rgb