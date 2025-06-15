import argparse, sys, os, time, warnings, cv2, numpy as np, torch, matplotlib.pyplot as plt, matplotlib.cm as cm
from ultralytics import YOLO
warnings.filterwarnings("ignore", category=UserWarning)
from easydict import EasyDict as edict

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
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Confidence threshold for fusion')
    parser.add_argument('--fusion_iou_threshold', type=float, default=0.7, help='IoU threshold for associating detections for fusion')

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
    configs.root_dir = 'D:\spa\SFA3D'
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


def convert_sfa3d_to_2d_boxes(sfa_detections, calib, img_shape):
    """Convert SFA3D 3D detections to 2D bounding boxes"""
    boxes_2d = []
    confidences = []
    
    if len(sfa_detections) > 0:
        kitti_dets = convert_det_to_real_values(sfa_detections)
        
        for detection in kitti_dets:
            confidence = detection[0]
            if confidence < 0.3:  # Skip low confidence detections
                continue
                
            # Convert 3D box to camera coordinates
            box_3d = detection[1:]
            box_3d_cam = lidar_to_camera_box(box_3d.reshape(1, -1), calib.V2C, calib.R0, calib.P2)[0]
            
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
            
            # Project to 2D
            corners_2d = calib.P2.dot(np.vstack((corners_3d, np.ones((1, 8)))))
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

# Helper to convert confidence to a "variance"
def confidence_to_variance(confidence, max_variance_pixels=100.0, min_confidence_threshold=0.1):
    if confidence < min_confidence_threshold:
        return max_variance_pixels * 100.0 
    return max_variance_pixels * ((1.0 - confidence) / (confidence + 0.01))

# Function to fuse two Gaussian parameters (mean and variance)
def fuse_gaussian_parameters(mean1, var1, mean2, var2):
    epsilon = 1e-6
    var1 = max(var1, epsilon)
    var2 = max(var2, epsilon)

    inv_var1 = 1.0 / var1
    inv_var2 = 1.0 / var2

    fused_mean = (mean1 * inv_var1 + mean2 * inv_var2) / (inv_var1 + inv_var2)
    fused_variance = 1.0 / (inv_var1 + inv_var2)
    
    return fused_mean, fused_variance

def bayesian_inspired_fuse_overlapping_detections(yolov8_detections, sfa3d_detections, fusion_iou_threshold):
    """
    Fuses detections from YOLOv8 and SFA3D using a Bayesian-inspired approach.
    For overlapping detections, it combines their bounding box parameters
    and confidence scores based on their uncertainties (derived from confidence).
    Non-overlapping detections are passed through as is.
    """
    fused_detections = []
    sfa3d_matched = [False] * len(sfa3d_detections)

    for yolo_det in yolov8_detections:
        yolo_box = yolo_det['box']  # [x, y, w, h]
        yolo_conf = yolo_det['confidence']
        yolo_class_id = yolo_det['class_id']
        yolo_class_name = yolo_det['class_name']

        best_match_sfa3d_idx = -1
        max_iou = 0.0
        
        # Find the best SFA3D match for the current YOLOv8 detection
        for i, sfa3d_det in enumerate(sfa3d_detections):
            if sfa3d_matched[i]:
                continue
            
            sfa3d_box = sfa3d_det['box']
            iou = calculate_iou(yolo_box, sfa3d_box)
            
            if iou > max_iou and iou >= fusion_iou_threshold:
                max_iou = iou
                best_match_sfa3d_idx = i
        
        if best_match_sfa3d_idx != -1:
            # Found a good overlap, perform fusion
            sfa3d_det = sfa3d_detections[best_match_sfa3d_idx]
            sfa3d_box = sfa3d_det['box']
            sfa3d_conf = sfa3d_det['confidence']

            max_var_pos = 100.0   # Max variance for x, y coordinates
            max_var_dim = 50.0   # Max variance for width, height dimensions

            # Calculate variances for each coordinate based on confidence
            yolo_var_x = confidence_to_variance(yolo_conf, max_variance_pixels=max_var_pos)
            yolo_var_y = confidence_to_variance(yolo_conf, max_variance_pixels=max_var_pos)
            yolo_var_w = confidence_to_variance(yolo_conf, max_variance_pixels=max_var_dim)
            yolo_var_h = confidence_to_variance(yolo_conf, max_variance_pixels=max_var_dim)

            sfa3d_var_x = confidence_to_variance(sfa3d_conf, max_variance_pixels=max_var_pos)
            sfa3d_var_y = confidence_to_variance(sfa3d_conf, max_variance_pixels=max_var_pos)
            sfa3d_var_w = confidence_to_variance(sfa3d_conf, max_variance_pixels=max_var_dim)
            sfa3d_var_h = confidence_to_variance(sfa3d_conf, max_variance_pixels=max_var_dim)
            
            # Fuse each bounding box parameter independently using Gaussian fusion
            fused_x, _ = fuse_gaussian_parameters(yolo_box[0], yolo_var_x, sfa3d_box[0], sfa3d_var_x)
            fused_y, _ = fuse_gaussian_parameters(yolo_box[1], yolo_var_y, sfa3d_box[1], sfa3d_var_y)
            fused_w, _ = fuse_gaussian_parameters(yolo_box[2], yolo_var_w, sfa3d_box[2], sfa3d_var_w)
            fused_h, _ = fuse_gaussian_parameters(yolo_box[3], yolo_var_h, sfa3d_box[3], sfa3d_var_h)
            
            # The fused confidence can be derived from the fused variance or simply averaged/max.
            fused_confidence = max(yolo_conf, sfa3d_conf) 
            
            fused_detections.append({
                'box': [int(fused_x), int(fused_y), int(fused_w), int(fused_h)],
                'confidence': fused_confidence,
                'class_id': yolo_class_id, # Keep YOLOv8's class ID, assuming it's more general
                'class_name': yolo_class_name,
                'model': 'Fused (Bayesian-Inspired)', 
                'color': (0, 255, 0) # Green for fused detections
            })
            sfa3d_matched[best_match_sfa3d_idx] = True # Mark SFA3D detection as matched
        else:
            # If no good SFA3D match, add YOLOv8 detection as is
            fused_detections.append(yolo_det)
    
    # Add any remaining unmatched SFA3D detections
    for i, sfa3d_det in enumerate(sfa3d_detections):
        if not sfa3d_matched[i]:
            fused_detections.append(sfa3d_det)
            
    return fused_detections
def create_fused_detections_wrapper(yolov8_data, sfa3d_data, confidence_threshold, fusion_iou_threshold):
    """
    Wrapper function to prepare data for Bayesian-inspired fusion and call it.
    This also handles initial filtering by confidence threshold before fusion.
    """
    yolov8_boxes, yolov8_confidences, yolov8_class_ids, yolov8_class_names = yolov8_data
    sfa3d_boxes, sfa3d_confidences = sfa3d_data

    # Filter detections by confidence before attempting fusion
    filtered_yolov8_dets = []
    for i, (box, conf, class_id) in enumerate(zip(yolov8_boxes, yolov8_confidences, yolov8_class_ids)):
        if conf >= confidence_threshold:
            filtered_yolov8_dets.append({
                'box': box,
                'confidence': conf,
                'class_id': class_id,
                'class_name': yolov8_class_names[class_id],
                'model': 'YOLOv8',
                'color': (0, 255, 255)  # Yellow for YOLOv8
            })
    
    filtered_sfa3d_dets = []
    for i, (box, conf) in enumerate(zip(sfa3d_boxes, sfa3d_confidences)):
        if conf >= confidence_threshold:
            filtered_sfa3d_dets.append({
                'box': box,
                'confidence': conf,
                'class_id': 0,  # Assuming 'car' for SFA3D for simplicity
                'class_name': 'car',
                'model': 'SFA3D',
                'color': (255, 0, 0)  # Blue for SFA3D
            })
    
    # Perform the Bayesian-inspired fusion
    fused_detections = bayesian_inspired_fuse_overlapping_detections(
        filtered_yolov8_dets, filtered_sfa3d_dets, fusion_iou_threshold
    )
    
    return fused_detections


def draw_fused_detections(img, fused_detections):
    """Draw fused detections with different colors for each model and enhanced visuals."""
    img_with_detections = img.copy()
    
    for detection in fused_detections:
        box = detection['box']
        confidence = detection['confidence']
        class_name = detection['class_name']
        model = detection['model']
        color = detection['color']
        
        x, y, w, h = box
        
        # Bounding box line thickness based on model type
        thickness = 2
        if model == 'Fused (Bayesian-Inspired)':
            thickness = 3 # Slightly thicker for fused detections

        # Draw bounding box
        cv2.rectangle(img_with_detections, (x, y), (x + w, y + h), color, thickness)
        
        # Create label text
        label = f"{model}: {class_name} ({confidence:.2f})"
        
        # Determine font scale and thickness dynamically for better readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, min(1.0, w / 200.0)) # Adjust font scale based on box width
        font_thickness = max(1, int(font_scale * 2))

        label_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Smart label placement: above the box if space, otherwise inside top-left
        label_y = y - 10 if y - 10 - label_size[1] > 0 else y + label_size[1] + 5

        # Draw label background
        cv2.rectangle(img_with_detections, (x, label_y - label_size[1] - baseline), 
                      (x + label_size[0], label_y + baseline), color, -1)
        
        # Draw label text
        cv2.putText(img_with_detections, label, (x, label_y), 
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # Use LINE_AA for anti-aliasing
    
    return img_with_detections


def create_detection_summary(fused_detections_before_nms, fused_detections_after_nms):
    """Create a summary image showing detection statistics with improved visuals and reduced height."""
    # Reduced height for summary
    summary_height = 180 # Adjusted from 300
    summary_img = np.zeros((summary_height, 1280, 3), dtype=np.uint8) 
    
    # Define colors
    YOLO_COLOR = (0, 255, 255) # Yellow
    SFA3D_COLOR = (255, 0, 0) # Blue
    FUSED_COLOR = (0, 255, 0) # Green
    TEXT_COLOR = (255, 255, 255) # White
    HEADER_COLOR = (0, 200, 255) # Orange

    # Count detections by model
    yolo_before = sum(1 for d in fused_detections_before_nms if d['model'] == 'YOLOv8')
    sfa_before = sum(1 for d in fused_detections_before_nms if d['model'] == 'SFA3D')
    fused_before = sum(1 for d in fused_detections_before_nms if d['model'] == 'Fused (Bayesian-Inspired)')

    yolo_after = sum(1 for d in fused_detections_after_nms if d['model'] == 'YOLOv8')
    sfa_after = sum(1 for d in fused_detections_after_nms if d['model'] == 'SFA3D')
    fused_after = sum(1 for d in fused_detections_after_nms if d['model'] == 'Fused (Bayesian-Inspired)')
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Adjust font sizes and line spacing for compactness
    header_font_scale = 0.7
    main_font_scale = 0.5
    legend_font_scale = 0.45
    text_thickness = 1 # Reduced thickness for smaller fonts
    
    line_spacing = 22 # Reduced line spacing

    # Header
    cv2.putText(summary_img, "Detection Summary", (480, 25), font, header_font_scale, HEADER_COLOR, text_thickness, cv2.LINE_AA)
    
    # Columns for "Before NMS" and "After NMS"
    col1_x = 50
    col2_x = 400
    col3_x = 750 # Start legend further right

    y_start = 60 # Starting Y coordinate for text
    cv2.putText(summary_img, "Before NMS:", (col1_x, y_start), font, main_font_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"YOLOv8: {yolo_before}", (col1_x, y_start + line_spacing), font, main_font_scale, YOLO_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"SFA3D: {sfa_before}", (col1_x, y_start + 2*line_spacing), font, main_font_scale, SFA3D_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"Fused: {fused_before}", (col1_x, y_start + 3*line_spacing), font, main_font_scale, FUSED_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"Total: {len(fused_detections_before_nms)}", (col1_x, y_start + 4*line_spacing), font, main_font_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)
    
    cv2.putText(summary_img, "After NMS:", (col2_x, y_start), font, main_font_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"YOLOv8: {yolo_after}", (col2_x, y_start + line_spacing), font, main_font_scale, YOLO_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"SFA3D: {sfa_after}", (col2_x, y_start + 2*line_spacing), font, main_font_scale, SFA3D_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"Fused: {fused_after}", (col2_x, y_start + 3*line_spacing), font, main_font_scale, FUSED_COLOR, text_thickness, cv2.LINE_AA)
    cv2.putText(summary_img, f"Total: {len(fused_detections_after_nms)}", (col2_x, y_start + 4*line_spacing), font, main_font_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)
    
    # Legend
    legend_y_start = y_start + 5 # Slightly adjust legend start
    cv2.putText(summary_img, "Legend:", (col3_x, legend_y_start), font, main_font_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)
    rect_height = 15 # Smaller rectangles
    text_offset_y = 12 # Offset for text in legend
    
    cv2.rectangle(summary_img, (col3_x, legend_y_start + line_spacing - rect_height // 2), 
                  (col3_x + 20, legend_y_start + line_spacing + rect_height // 2), YOLO_COLOR, -1)
    cv2.putText(summary_img, "YOLOv8 (Unfused)", (col3_x + 30, legend_y_start + line_spacing + text_offset_y // 2), 
                font, legend_font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
    
    cv2.rectangle(summary_img, (col3_x, legend_y_start + 2*line_spacing - rect_height // 2), 
                  (col3_x + 20, legend_y_start + 2*line_spacing + rect_height // 2), SFA3D_COLOR, -1)
    cv2.putText(summary_img, "SFA3D (Unfused)", (col3_x + 30, legend_y_start + 2*line_spacing + text_offset_y // 2), 
                font, legend_font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
    
    cv2.rectangle(summary_img, (col3_x, legend_y_start + 3*line_spacing - rect_height // 2), 
                  (col3_x + 20, legend_y_start + 3*line_spacing + rect_height // 2), FUSED_COLOR, -1)
    cv2.putText(summary_img, "Fused (Bayesian)", (col3_x + 30, legend_y_start + 3*line_spacing + text_offset_y // 2), 
                font, legend_font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
    
    return summary_img


if __name__ == '__main__':
    # Initialize output directory
    output_dir = "D:\\spa\\SFA3D\\sfa\\fused_detection_results_bayesian" 
    os.makedirs(output_dir, exist_ok=True)
    print(f"All fused detection results will be saved to: {output_dir}")
    
    # Initialize YOLOv8 model
    yolov8_weights_path = "D:\\spa\\SFA3D\\sfa\\models\\yolov8n.pt"
    yolov8_model = YOLO(yolov8_weights_path)
    yolov8_class_names = yolov8_model.names
    
    # Initialize SFA3D model
    configs = parse_test_configs()
    sfa_model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    sfa_model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))
    sfa_model = sfa_model.to(device=configs.device)
    sfa_model.eval()

    # Create test dataloader
    test_dataloader = create_test_dataloader(configs)

    cv2.namedWindow("Fused Object Detection", cv2.WINDOW_NORMAL)
    
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_res = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except:
        screen_res = 1920, 1080 # Default if tkinter is not available

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
            
            # Convert SFA3D 3D detections to 2D bounding boxes
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            sfa3d_boxes_2d, sfa3d_confidences = convert_sfa3d_to_2d_boxes(detections, calib, img_bgr.shape)
            print(f"SFA3D detected {len(sfa3d_boxes_2d)} objects (converted to 2D)")
            
            # === Bayesian-Inspired Fusion and NMS ===
            # Prepare data for fusion
            yolov8_data = (yolov8_boxes, yolov8_confidences, yolov8_class_ids, yolov8_class_names)
            sfa3d_data = (sfa3d_boxes_2d, sfa3d_confidences)
            
            # Perform fusion using the new Bayesian-inspired logic
            fused_detections_before_nms = create_fused_detections_wrapper(
                yolov8_data, sfa3d_data, configs.confidence_threshold, configs.fusion_iou_threshold
            )
            print(f"Total detections after Bayesian-inspired fusion (before NMS): {len(fused_detections_before_nms)}")
            
            # Apply NMS
            fused_detections_after_nms = apply_nms_to_fused_detections(
                fused_detections_before_nms, configs.nms_threshold
            )
            print(f"Total detections after NMS: {len(fused_detections_after_nms)}")
            
            # === Visualization ===
            # Create different visualization images
            img_before_nms = draw_fused_detections(img_bgr.copy(), fused_detections_before_nms)
            img_after_nms = draw_fused_detections(img_bgr.copy(), fused_detections_after_nms)
            
            # Create summary (will be smaller)
            summary_img = create_detection_summary(fused_detections_before_nms, fused_detections_after_nms)
            
            # Calculate aspect ratios for resizing of main images
            original_h, original_w, _ = img_bgr.shape
            
            # Determine ideal width for combined top images to maximize their size
            # based on screen resolution and available space after summary
            
            # Maximum allowed height for the total combined image (screen_height - some margin)
            max_total_height = screen_res[1] - 100 
            
            # Height available for the top image section
            available_top_height = max_total_height - summary_img.shape[0] - 20 # 20 for some padding
            
            # Calculate target width based on available height and original aspect ratio
            # Each image will have this height, so the combined_top will be `target_height`
            target_height_for_top_images = int(original_h * (available_top_height / original_h))
            target_width_for_top_images = int(original_w * (target_height_for_top_images / original_h))
            
            # Ensure target_width is reasonable for two images side-by-side
            # Max width for combined_top is screen_res[0] - 100
            max_combined_top_width = screen_res[0] - 100
            
            if (target_width_for_top_images * 2) > max_combined_top_width:
                # If images are too wide, scale down based on max_combined_top_width
                target_width_for_top_images = max_combined_top_width // 2
                target_height_for_top_images = int(original_h * (target_width_for_top_images / original_w))
            
            # Ensure a minimum reasonable size if calculations result in too small images
            min_img_dim = 400
            if target_width_for_top_images < min_img_dim or target_height_for_top_images < min_img_dim:
                target_width_for_top_images = max(min_img_dim, target_width_for_top_images)
                target_height_for_top_images = max(min_img_dim, target_height_for_top_images)
                # Re-calculate the other dimension to maintain aspect ratio
                if (original_w/original_h) > 1: # Wider than tall
                    target_height_for_top_images = int(target_width_for_top_images / (original_w / original_h))
                else: # Taller than wide or square
                    target_width_for_top_images = int(target_height_for_top_images * (original_w / original_h))

            
            img_before_nms_resized = cv2.resize(img_before_nms, (target_width_for_top_images, target_height_for_top_images))
            img_after_nms_resized = cv2.resize(img_after_nms, (target_width_for_top_images, target_height_for_top_images))

            # Combine images for display
            combined_top = np.hstack([img_before_nms_resized, img_after_nms_resized])
            
            # Add labels
            text_x_offset = 20
            text_y_offset = 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_scale = 1.0 # Keep this relatively large for titles
            text_thickness = 2
            text_color = (100, 100, 100) # White

            cv2.putText(combined_top, "Before NMS", (text_x_offset, text_y_offset), font, text_scale, text_color, text_thickness, cv2.LINE_AA)
            cv2.putText(combined_top, "After NMS", (target_width_for_top_images + text_x_offset, text_y_offset), font, text_scale, text_color, text_thickness, cv2.LINE_AA)
            
            # Resize summary to match the width of combined_top
            summary_resized = cv2.resize(summary_img, (combined_top.shape[1], summary_img.shape[0]))
            
            # Final combined image
            final_combined = np.vstack([combined_top, summary_resized])
            
            # === Save Results ===
            output_path = os.path.join(output_dir, f"{img_fn}_fused_bayesian_detection.jpg")
            cv2.imwrite(output_path, final_combined)
            print(f"Saved fused detection result: {output_path}")
            
            # === Display ===
            # The window will automatically resize to fit final_combined
            cv2.imshow("Fused Object Detection", final_combined)
            
            print(f'\nProcessing time: {(t2 - t1) * 1000:.1f}ms')
            print(f'Detection Summary:')
            print(f'   - YOLOv8: {len(yolov8_boxes)} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "YOLOv8")}')
            print(f'   - SFA3D: {len(sfa3d_boxes_2d)} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "SFA3D")}')
            print(f'   - Fused (Bayesian-Inspired): {sum(1 for d in fused_detections_before_nms if d["model"] == "Fused (Bayesian-Inspired)")} -> {sum(1 for d in fused_detections_after_nms if d["model"] == "Fused (Bayesian-Inspired)")}')
            print(f'   - Total (before NMS): {len(fused_detections_before_nms)}')
            print(f'   - Total (after NMS): {len(fused_detections_after_nms)}')
            print('\n[INFO] Press N for next sample | Press ESC to quit\n')
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # Esc key
                break
            elif key == ord('N') or key == ord('n'): # Added 'N' or 'n' key for next sample
                continue 

    cv2.destroyAllWindows()
    print(f"\nAll results saved to: {output_dir}")