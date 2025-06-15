import argparse
import sys
import os
import time
import warnings
from ultralytics import YOLO
import cv2
import os

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration


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
    parser.add_argument('--enable_kfpn_viz', action='store_true', help='Enable KFPN visualization')
    configs = edict(vars(parser.parse_args()))

    configs.pin_memory = True
    configs.distributed = False
    configs.device = torch.device("cpu")  # Force CPU
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
    configs.root_dir = 'D:\spa\SFA3D'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'complex_results', configs.saved_fn)
        make_folder(configs.results_dir)
        make_folder(os.path.join(configs.results_dir, "visualizations"))
        make_folder(os.path.join(configs.results_dir, "kfpn_visualizations"))
        make_folder(os.path.join(configs.results_dir, "backbone_visualizations"))
        make_folder(os.path.join(configs.results_dir, "fpn_visualizations"))
        make_folder(os.path.join(configs.results_dir, "attention_visualizations"))

    return configs


def save_or_show_image(window_name, image, save_path=None, resizable=True):
    """Display image in a resizable window with proper sizing"""
    if resizable:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    cv2.imshow(window_name, image)
    
    if save_path:
        cv2.imwrite(save_path, image)
    
    # Auto-resize window to fit screen if too large
    if resizable:
        screen_res = 1920, 1080  # Default screen resolution
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_res = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
        except:
            pass
        
        img_h, img_w = image.shape[:2]
        if img_w > screen_res[0] or img_h > screen_res[1]:
            scale = min(screen_res[0] / img_w, screen_res[1] / img_h) * 0.8
            cv2.resizeWindow(window_name, int(img_w * scale), int(img_h * scale))


def normalize_feature_map(feat_map):
    """Normalize feature map to 0-255 range"""
    if feat_map.max() == feat_map.min():
        return np.zeros_like(feat_map, dtype=np.uint8)
    
    feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
    return (feat_map * 255).astype(np.uint8)


def create_feature_grid(feature_maps, max_channels=16, target_size=(800, 600)):
    """Create a grid visualization of multiple feature maps"""
    if len(feature_maps) == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Take only first few channels for visualization
    n_channels = min(max_channels, feature_maps.shape[0])
    
    # Calculate grid dimensions
    grid_h = int(np.ceil(np.sqrt(n_channels)))
    grid_w = int(np.ceil(n_channels / grid_h))
    
    # Get individual feature map size
    feat_h, feat_w = feature_maps.shape[1], feature_maps.shape[2]
    
    # Create grid image
    grid_img = np.zeros((feat_h * grid_h, feat_w * grid_w), dtype=np.uint8)
    
    for i in range(n_channels):
        row = i // grid_w
        col = i % grid_w
        if row >= grid_h:
            break
            
        feat_map = normalize_feature_map(feature_maps[i])
        start_row, end_row = row * feat_h, (row + 1) * feat_h
        start_col, end_col = col * feat_w, (col + 1) * feat_w
        grid_img[start_row:end_row, start_col:end_col] = feat_map
    
    # Apply colormap and resize
    grid_img_color = cv2.applyColorMap(grid_img, cv2.COLORMAP_JET)
    grid_img_color = cv2.resize(grid_img_color, target_size)
    
    return grid_img_color


def visualize_backbone_features(backbone_features, batch_idx, configs):
    """Visualize backbone features from different layers"""
    print("Visualizing backbone features...")
    
    for layer_name, features in backbone_features.items():
        if isinstance(features, torch.Tensor):
            features = features[0].cpu().numpy()  # Take first batch
            
        # Create grid visualization
        grid_img = create_feature_grid(features, max_channels=16, target_size=(600, 400))
        
        # Create averaged feature map
        avg_feat = features.mean(axis=0)
        avg_feat = normalize_feature_map(avg_feat)
        avg_feat_color = cv2.applyColorMap(avg_feat, cv2.COLORMAP_JET)
        avg_feat_color = cv2.resize(avg_feat_color, (300, 300))
        
        # Combine grid and average views
        combined = np.hstack([avg_feat_color, cv2.resize(grid_img, (600, 300))])
        
        # Save and show
        save_path = None
        if configs.save_test_output:
            save_dir = os.path.join(configs.results_dir, "backbone_visualizations", f"{batch_idx:04d}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{layer_name}.jpg")
        
        save_or_show_image(f"Backbone {layer_name} (Avg + Grid)", combined, save_path, resizable=True)

def visualize_kfpn_features(kfpn_features, batch_idx, configs):
    """Visualize KFPN features from different levels"""
    print("Visualizing KFPN features...")
    
    for level, features in enumerate(kfpn_features):
        if isinstance(features, torch.Tensor):
            features = features[0].cpu().numpy()  # Take first batch
            
        # Create averaged feature map
        avg_feat = features.mean(axis=0)
        avg_feat = normalize_feature_map(avg_feat)
        feat_color = cv2.applyColorMap(avg_feat, cv2.COLORMAP_JET)
        feat_color = cv2.resize(feat_color, (400, 400))
        
        # Create grid visualization for individual channels
        grid_img = create_feature_grid(features, max_channels=9, target_size=(600, 400))
        
        # Create statistics visualization
        stats_img = create_feature_statistics(features)
        stats_img = cv2.resize(stats_img, (600, 200))  # Resize to match top row width
        
        # Create zero padding to match widths
        zero_pad = np.zeros((200, 400, 3), dtype=np.uint8)  # Adjusted to match
        
        # Combine all views with consistent widths
        top_row = np.hstack([feat_color, grid_img])  # Total width: 400 + 600 = 1000
        bottom_row = np.hstack([stats_img, zero_pad])  # Now also width 1000 (600 + 400)
        
        # Combine top and bottom
        combined = np.vstack([top_row, bottom_row])
        
        # Save and show in resizable window
        save_path = None
        if configs.save_test_output:
            save_dir = os.path.join(configs.results_dir, "kfpn_visualizations", f"{batch_idx:04d}")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"kfpn_level_{level}.jpg")
        
        save_or_show_image(f"KFPN Level {level} Features", combined, save_path, resizable=True)
def create_feature_statistics(features):
    """Create a visualization showing feature statistics"""
    # Calculate statistics
    mean_vals = features.mean(axis=(1, 2))
    std_vals = features.std(axis=(1, 2))
    max_vals = features.max(axis=(1, 2))[0] if isinstance(features, torch.Tensor) else features.max(axis=(1, 2))
    min_vals = features.min(axis=(1, 2))[0] if isinstance(features, torch.Tensor) else features.min(axis=(1, 2))
    
    # Create a simple bar chart visualization
    n_channels = min(features.shape[0], 32)  # Limit to first 32 channels
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    
    channels = np.arange(n_channels)
    ax1.bar(channels, mean_vals[:n_channels])
    ax1.set_title('Channel Means')
    ax1.set_xlabel('Channel')
    
    ax2.bar(channels, std_vals[:n_channels])
    ax2.set_title('Channel Std Dev')
    ax2.set_xlabel('Channel')
    
    ax3.bar(channels, max_vals[:n_channels])
    ax3.set_title('Channel Max Values')
    ax3.set_xlabel('Channel')
    
    ax4.bar(channels, min_vals[:n_channels])
    ax4.set_title('Channel Min Values')
    ax4.set_xlabel('Channel')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to opencv image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return cv2.resize(img_bgr, (400, 200))


def visualize_fpn_outputs(fpn_outputs, batch_idx, configs):
    """Visualize FPN outputs for each head and level"""
    print("Visualizing FPN outputs...")
    
    for head, level_outputs in fpn_outputs.items():
        # Create a combined visualization for all levels of this head
        level_imgs = []
        
        for level, features in enumerate(level_outputs):
            if isinstance(features, torch.Tensor):
                features = features[0].cpu().numpy()  # Take first batch
                
            # Create averaged feature map
            avg_feat = features.mean(axis=0) if features.shape[0] > 1 else features[0]
            avg_feat = normalize_feature_map(avg_feat)
            
            # Use different colormaps for different heads
            colormap = cv2.COLORMAP_VIRIDIS if 'hm' in head else cv2.COLORMAP_PLASMA
            feat_color = cv2.applyColorMap(avg_feat, colormap)
            feat_color = cv2.resize(feat_color, (300, 300))
            
            # Add level label
            cv2.putText(feat_color, f'Level {level}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            level_imgs.append(feat_color)
        
        # Combine all levels horizontally
        if len(level_imgs) > 0:
            combined_levels = np.hstack(level_imgs)
            
            # Save and show
            save_path = None
            if configs.save_test_output:
                save_dir = os.path.join(configs.results_dir, "fpn_visualizations", f"{batch_idx:04d}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"fpn_{head}_all_levels.jpg")
            
            save_or_show_image(f"FPN {head} - All Levels", combined_levels, save_path, resizable=True)


def visualize_attention_weights(attention_weights, batch_idx, configs):
    print("Visualizing weights...")
    
    for head, weights in attention_weights.items():
        if isinstance(weights, torch.Tensor):
            weights = weights[0].cpu().numpy()  # Take first batch
            
        # weights shape: [C, H, W, num_levels]
        if len(weights.shape) < 4:
            print(f"Warning: Unexpected attention weights shape for {head}: {weights.shape}")
            continue
            
        num_levels = weights.shape[-1]
        
        # Create visualization for each level's attention
        level_attentions = []
        for level in range(num_levels):
            level_weight = weights[..., level].mean(axis=0)  # Average across channels
            level_weight = normalize_feature_map(level_weight)
            level_weight_color = cv2.applyColorMap(level_weight, cv2.COLORMAP_HOT)
            level_weight_color = cv2.resize(level_weight_color, (250, 250))
            
            # Add level label
            cv2.putText(level_weight_color, f'Level {level}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            level_attentions.append(level_weight_color)
        
        # Combine all levels horizontally
        if len(level_attentions) > 0:
            combined_attention = np.hstack(level_attentions)
            
            # Create attention distribution plot
            avg_attention = weights.mean(axis=(0, 1, 2))  # Average across spatial and channel dims
            attention_plot = create_attention_distribution_plot(avg_attention, head)
            
            # Combine heatmaps and distribution plot
            final_viz = np.vstack([
                combined_attention,
                cv2.resize(attention_plot, (combined_attention.shape[1], 200))
            ])
            
            # Save and show
            save_path = None
            if configs.save_test_output:
                save_dir = os.path.join(configs.results_dir, "attention_visualizations", f"{batch_idx:04d}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"attention_{head}.jpg")
            
            save_or_show_image(f"Attention Weights {head}", final_viz, save_path, resizable=True)


def create_attention_distribution_plot(attention_values, head_name):
    """Create a plot showing attention distribution across levels"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    
    levels = np.arange(len(attention_values))
    bars = ax.bar(levels, attention_values, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, attention_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel('FPN Level')
    ax.set_ylabel('Average Weight')
    ax.set_title(f'Distribution - {head_name}')
    ax.set_xticks(levels)
    ax.set_xticklabels([f'L{i}' for i in levels])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to opencv image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return img_bgr


def comprehensive_kfpn_visualization(model, batch_idx, configs):
    """Comprehensive KFPN visualization using stored features"""
    if not hasattr(model, 'get_visualization_data'):
        print("Model doesn't support visualization data extraction")
        return
    
    viz_data = model.get_visualization_data()
    
    print(f"\n=== KFPN Visualization for Sample {batch_idx} ===")
    
    # 1. Visualize backbone features
    if viz_data['backbone_features']:
        visualize_backbone_features(viz_data['backbone_features'], batch_idx, configs)
    
    # 2. Visualize KFPN features
    if viz_data['kfpn_features']:
        visualize_kfpn_features(viz_data['kfpn_features'], batch_idx, configs)
    
    # 3. Visualize FPN outputs
    if viz_data['fpn_outputs']:
        visualize_fpn_outputs(viz_data['fpn_outputs'], batch_idx, configs)
    
    # 4. Visualize attention weights
    if viz_data['kfpn_weights']:
        visualize_attention_weights(viz_data['kfpn_weights'], batch_idx, configs)
    
    print("=== KFPN Visualization Complete ===\n")


def yolov8_detect(image_path, model):
    results = model(image_path)
    detections = results[0]  # Get first image's results
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

    print(f"Detected {len(result_boxes)} boxes.")
    return result_boxes, result_confidences, result_class_ids

def draw_yolov8_boxes(img, boxes, confidences, class_ids, class_names):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img
# [Previous imports and functions remain the same until the main loop]

if __name__ == '__main__':
    # Initialize output directory
    output_dir = "D:\\spa\\SFA3D\\sfa\\test3_saves"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All detection results will be saved to: {output_dir}")
    
    # Initialize YOLOv8 model
    yolov8_weights_path = "D:\\spa\\SFA3D\\sfa\\models\\yolov8n.pt"
    yolov8_model = YOLO(yolov8_weights_path)
    yolov8_class_names = yolov8_model.names
    
    # Initialize SFA3D (FPN-ResNet) model
    configs = parse_test_configs()
    sfa_model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    sfa_model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))
    sfa_model = sfa_model.to(device=configs.device)
    sfa_model.eval()

    # Create test dataloader
    test_dataloader = create_test_dataloader(configs)
    out_cap = None

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device).float()
            
            # Get image path for YOLOv8
            img_path = metadatas['img_path'][0]
            img_fn = os.path.basename(img_path)[:-4]  # Get filename without extension
            
            # === YOLOv8 Detection ===
            # Read image for YOLOv8 (using the same image as SFA3D)
            img_rgb = img_rgbs[0].numpy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Perform YOLOv8 detection
            yolov8_boxes, yolov8_confidences, yolov8_class_ids = yolov8_detect(img_path, yolov8_model)
            yolov8_img = draw_yolov8_boxes(img_bgr.copy(), yolov8_boxes, yolov8_confidences, yolov8_class_ids, yolov8_class_names)
            
            # === SFA3D (FPN-ResNet) Processing ===
            # Raw BEV visualization
            raw_bev_np = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            raw_bev_resized = cv2.resize(raw_bev_np, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            
            t1 = time_synchronized()
            outputs = sfa_model(input_bev_maps)
            
            # KFPN visualization if enabled
            if configs.enable_kfpn_viz or configs.save_test_output:
                comprehensive_kfpn_visualization(sfa_model, batch_idx, configs)
            
            # Process heatmaps
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            
            # Decode detections
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                              outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()
            detections = detections[0]
            
            # Draw SFA3D predictions
            bev_map = draw_predictions(raw_bev_resized.copy(), detections.copy(), configs.num_classes)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            
            # Convert SFA3D detections to camera coordinates
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            kitti_dets = convert_det_to_real_values(detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
            
            # Create combined SFA3D view
            sfa_combined = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
            
            # === Save Results ===
            # Save YOLOv8 results
            yolov8_output_path = os.path.join(output_dir, f"{img_fn}_yolov8.jpg")
            cv2.imwrite(yolov8_output_path, yolov8_img)
            
            # Save SFA3D results
            sfa_output_path = os.path.join(output_dir, f"{img_fn}_sfa3d.jpg")
            cv2.imwrite(sfa_output_path, sfa_combined)
            
            print(f"Saved results for {img_fn} to:")
            print(f"- YOLOv8: {yolov8_output_path}")
            print(f"- SFA3D: {sfa_output_path}")
            
            # === Display Results ===
            # Create resizable windows for both models
            cv2.namedWindow("YOLOv8 2D Detection", cv2.WINDOW_NORMAL)
            cv2.namedWindow("SFA3D 3D Detection", cv2.WINDOW_NORMAL)
            
            # Show YOLOv8 results
            cv2.imshow("YOLOv8 2D Detection", yolov8_img)
            
            # Show SFA3D results
            cv2.imshow("SFA3D 3D Detection", sfa_combined)
            
            # Auto-resize windows to fit screen
            screen_res = 1920, 1080  # Default screen resolution
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_res = root.winfo_screenwidth(), root.winfo_screenheight()
                root.destroy()
            except:
                pass
            
            # Calculate window positions and sizes
            win_width = min(screen_res[0] // 2 - 50, 1000)
            win_height = min(screen_res[1] - 100, 800)
            
            # Set window properties
            cv2.resizeWindow("YOLOv8 2D Detection", win_width, win_height)
            cv2.resizeWindow("SFA3D 3D Detection", win_width, win_height)
            
            # Position windows side by side
            cv2.moveWindow("YOLOv8 2D Detection", 20, 20)
            cv2.moveWindow("SFA3D 3D Detection", win_width + 40, 20)
            
            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(
                batch_idx, (t2 - t1) * 1000, 1 / (t2 - t1)))
            
            print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...')
            print('[INFO] Left Window: YOLOv8 2D Detection | Right Window: SFA3D 3D Detection\n')
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Esc key
                break
            elif key == ord('c'):  # Close all windows
                cv2.destroyAllWindows()

    cv2.destroyAllWindows()