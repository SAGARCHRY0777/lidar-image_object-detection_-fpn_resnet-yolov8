import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

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
        make_folder(os.path.join(configs.results_dir, "visualizations"))  # === Added

    return configs


def save_or_show_image(window_name, image, save_path=None):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    if save_path:
        cv2.imwrite(save_path, image)
    # Adjust window size to fit screen if too large
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
        scale = min(screen_res[0] / img_w, screen_res[1] / img_h) * 0.9
        cv2.resizeWindow(window_name, int(img_w * scale), int(img_h * scale))


if __name__ == '__main__':
    configs = parse_test_configs()

    model = create_model(configs)## this contains the model architecture and the additional configurations aas well as the outptut heads
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    model = model.to(device=configs.device)
    out_cap = None
    model.eval()

    # Create visualization directory if needed
    if configs.save_test_output:
        os.makedirs(os.path.join(configs.results_dir, "visualizations"), exist_ok=True)

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            metadatas, bev_maps, img_rgbs = batch_data
            input_bev_maps = bev_maps.to(configs.device).float()

            # === 1. Raw BEV visualization ===
            raw_bev_np = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            raw_bev_resized = cv2.resize(raw_bev_np, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            save_or_show_image("1. Raw BEV", raw_bev_resized,
                             os.path.join(configs.results_dir, "visualizations", f"{batch_idx:04d}_raw_bev.jpg") if configs.save_test_output else None)

            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            print("outputs keys:", outputs.keys())
            # === 2. Raw Center Heatmap (Pre-sigmoid) - Combined ===
            hm_raw_combined = outputs['hm_cen'][0].cpu().numpy().max(axis=0)
            heat_raw_combined = (hm_raw_combined * 255).astype(np.uint8)
            heat_raw_combined_color = cv2.applyColorMap(heat_raw_combined, cv2.COLORMAP_JET)
            save_or_show_image("2. Raw Center Heatmap (Pre-sigmoid)", heat_raw_combined_color,
                              None)  # Only display, not save

            # === 3. Raw Heatmaps (Per-class, Pre-sigmoid) ===
            hm_raw = outputs['hm_cen'][0].cpu().numpy()
            for cls_id in range(hm_raw.shape[0]):
                heat_raw = (hm_raw[cls_id] * 255).astype(np.uint8)
                heat_raw_color = cv2.applyColorMap(heat_raw, cv2.COLORMAP_JET)
                save_or_show_image(f"3. Raw Heatmap Class {cls_id}", heat_raw_color,
                                  os.path.join(configs.results_dir, "visualizations", f"{batch_idx:04d}_raw_heatmap_cls{cls_id}.jpg") if configs.save_test_output else None)

            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])

            # === 4. Sigmoid Center Heatmap - Combined ===
            hm_sigmoid_combined = outputs['hm_cen'][0].cpu().numpy().max(axis=0)
            heat_sigmoid_combined = (hm_sigmoid_combined * 255).astype(np.uint8)
            heat_sigmoid_combined_color = cv2.applyColorMap(heat_sigmoid_combined, cv2.COLORMAP_JET)
            save_or_show_image("4. Sigmoid Center Heatmap", heat_sigmoid_combined_color,
                              None)  # Only display, not save

            # === 5. Sigmoid Heatmaps (Per-class) ===
            hm_sigmoid = outputs['hm_cen'][0].cpu().numpy()
            for cls_id in range(hm_sigmoid.shape[0]):
                heat = (hm_sigmoid[cls_id] * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                save_or_show_image(f"5. Sigmoid Heatmap Class {cls_id}", heat_color,
                                  os.path.join(configs.results_dir, "visualizations", f"{batch_idx:04d}_sigmoid_heatmap_cls{cls_id}.jpg") if configs.save_test_output else None)

            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

            # === Detection decoding ===
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                              outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
            t2 = time_synchronized()

            detections = detections[0]
            
            # === 6. BEV Map with Boxes ===
            bev_map = draw_predictions(raw_bev_resized.copy(), detections.copy(), configs.num_classes)
            bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
            
            # === 7. RGB Image with 3D Boxes ===
            img_path = metadatas['img_path'][0]
            img_rgb = img_rgbs[0].numpy()
            img_rgb = cv2.resize(img_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
            kitti_dets = convert_det_to_real_values(detections)

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

            # === 8. Combined View (RGB + BEV) ===
            out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=configs.output_width)
            save_or_show_image("8. Combined View", out_img, None)  # Always display combined view

            print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(
                batch_idx, (t2 - t1) * 1000, 1 / (t2 - t1)))

            # === 9. Save Outputs ===
            if configs.save_test_output:
                # Save final image output
                if configs.output_format == 'image':
                    img_fn = os.path.basename(metadatas['img_path'][0])[:-4]
                    cv2.imwrite(os.path.join(configs.results_dir, f'{img_fn}.jpg'), out_img)
                # Save video output
                elif configs.output_format == 'video':
                    if out_cap is None:
                        out_cap_h, out_cap_w = out_img.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_cap = cv2.VideoWriter(
                            os.path.join(configs.results_dir, f'{configs.output_video_fn}.avi'),
                            fourcc, 30, (out_cap_w, out_cap_h))
                    out_cap.write(out_img)
                else:
                    raise TypeError

            print('\n[INFO] Press n to see the next sample >>> Press Esc to quit...\n')
            if cv2.waitKey(0) & 0xFF == 27:
                break

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()