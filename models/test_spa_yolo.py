import argparse
import sys
import os
import time
import warnings
import cv2
import torch
import numpy as np

from easydict import EasyDict as edict

warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Local imports from SFA3D
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing, draw_predictions, convert_det_to_real_values
from utils.torch_utils import _sigmoid
import config.kitti_config as cnf
from data_process.transformation import lidar_to_camera_box
from utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from data_process.kitti_data_utils import Calibration

# ------------------------------ #
# Load YOLOv4
# ------------------------------ #
def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    output_layers = [net.getLayerNames()[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_with_yolo(net, output_layers, classes, image, conf_threshold=0.5, nms_threshold=0.4):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# ------------------------------ #
# Config Parser
# ------------------------------ #
def parse_test_configs():
    parser = argparse.ArgumentParser(description='Testing config for the Implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN')
    parser.add_argument('--arch', type=str, default='fpn_resnet_18', metavar='ARCH')
    parser.add_argument('--pretrained_path', type=str, default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth')
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_idx', default=0, type=int)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--save_test_output', action='store_true')
    parser.add_argument('--output_format', type=str, default='image')
    parser.add_argument('--output_video_fn', type=str, default='out_fpn_resnet_18')
    parser.add_argument('--output-width', type=int, default=608)

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False
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
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs

# ------------------------------ #
# Main Execution
# ------------------------------ #
if __name__ == '__main__':
    configs = parse_test_configs()

    # Load YOLOv4
    yolo_cfg = r"D:\spa\SFA3D\sfa\models\yolov4.cfg"
    yolo_weights = r"D:\spa\SFA3D\sfa\models\yolov4.weights"
    yolo_names = r"D:\spa\SFA3D\sfa\models\coco.names"
    yolo_net, yolo_classes, yolo_output_layers = load_yolo_model(yolo_cfg, yolo_weights, yolo_names)

    # Load SFA3D model
    model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), f"No file at {configs.pretrained_path}"
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    configs.device = torch.device('cpu')  # Force CPU usage
    model = model.to(device=configs.device)
    model.eval()

    test_dataloader = create_test_dataloader(configs)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device).float()

            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]
            bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            bev_map = draw_predictions(bev_map, detections.copy(), configs.num_classes)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

            img_path = metadatas['img_path'][0]
            img_rgb = img_rgbs[0].numpy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Run YOLO on RGB image
            img_bgr_yolo = detect_with_yolo(yolo_net, yolo_output_layers, yolo_classes, img_bgr.copy())

            # Project 3D boxes to image
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            kitti_dets = convert_det_to_real_values(detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr_yolo = show_rgb_image_with_boxes(img_bgr_yolo, kitti_dets, calib)

            # Final merge
            out_img = merge_rgb_to_bev(img_bgr_yolo, bev_map, output_width=configs.output_width)

            print(f'[INFO] Sample {batch_idx} | Time: {(t2 - t1) * 1000:.1f} ms | FPS: {1 / (t2 - t1):.2f}')
            if configs.save_test_output:
                img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                cv2.imwrite(os.path.join(configs.results_dir, f'{img_fn}.jpg'), out_img)

            cv2.imshow("YOLO + SFA3D Object Detection", out_img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('n'):  # 'n' for next image
                continue

    cv2.destroyAllWindows()
