import argparse,sys,os,time,warnings,cv2,numpy as np,torch,matplotlib.pyplot as plt,matplotlib.cm as cm
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
            if confidence < 0.2:  # Skip low confidence detections
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
                'color': (255, 255, 0)	# Orange for YOLOv8
            })
    
    # Add SFA3D detections
    for i, (box, conf) in enumerate(zip(sfa3d_boxes, sfa3d_confidences)):
        if conf >= confidence_threshold:
            fused_detections.append({
                'box': box,
                'confidence': conf,
                'class_id': 0,  # Assuming car class
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
    cv2.putText(summary_img, f"YOLOv8: {yolo_before}", (50, 110), font, 0.6, (255, 255, 0)	, 2)
    cv2.putText(summary_img, f"SFA3D: {sfa_before}", (50, 140), font, 0.6, (255, 0, 0), 2)
    cv2.putText(summary_img, f"Total: {len(fused_detections_before_nms)}", (50, 170), font, 0.6, (255, 255, 255), 2)
    
    cv2.putText(summary_img, "After NMS:", (300, 80), font, 0.6, (255, 255, 255), 2)
    cv2.putText(summary_img, f"YOLOv8: {yolo_after}", (300, 110), font, 0.6, (255, 255, 0)	, 2)
    cv2.putText(summary_img, f"SFA3D: {sfa_after}", (300, 140), font, 0.6, (255, 0, 0), 2)
    cv2.putText(summary_img, f"Total: {len(fused_detections_after_nms)}", (300, 170), font, 0.6, (255, 255, 255), 2)
    
    # Legend
    cv2.putText(summary_img, "Legend:", (50, 220), font, 0.6, (255, 255, 255), 2)
    cv2.rectangle(summary_img, (50, 230), (70, 250), (255, 140, 0), -1)
    cv2.putText(summary_img, "YOLOv8", (80, 245), font, 0.5, (255, 255, 255), 1)
    cv2.rectangle(summary_img, (200, 230), (220, 250), (255, 255, 0)	, -1)
    cv2.putText(summary_img, "SFA3D", (230, 245), font, 0.5, (255, 255, 255), 1)
    
    return summary_img


if __name__ == '__main__':
    # Initialize output directory
    output_dir = "D:\\spa\\SFA3D\\sfa\\fused_detection_results_nms"
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
            output_path = os.path.join(output_dir, f"{img_fn}_fused_detection.jpg")
            cv2.imwrite(output_path, final_combined)
            print(f"Saved fused detection result: {output_path}")
            
            # === Display ===
            cv2.namedWindow("Fused Object Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Fused Object Detection", final_combined)
            
            # Auto-resize window
            screen_res = 1920, 1080
            try:
                import tkinter as tk
                root = tk.Tk()
                screen_res = root.winfo_screenwidth(), root.winfo_screenheight()
                root.destroy()
            except:
                pass
            
            win_width = min(screen_res[0] - 100, 1280)
            win_height = min(screen_res[1] - 100, 800)
            cv2.resizeWindow("Fused Object Detection", win_width, win_height)
            
            print(f'\nProcessing time: {(t2 - t1) * 1000:.1f}ms')
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