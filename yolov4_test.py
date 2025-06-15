import cv2
import numpy as np
import os

def load_yolov4_model(weights_path, config_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(f"Loaded YOLOv4 model with {len(classes)} classes.")
    print(f"Output layers: {output_layers}")
    return net, classes, output_layers

def yolov4_detect(image, net, output_layers, conf_threshold=0.74, nms_threshold=0.2):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print(f"Detected {len(boxes)} boxes before NMS, {len(indexes)} after NMS.")
    print("indexes=",indexes)
    result_boxes = []
    result_confidences = []
    result_class_ids = []

    if len(indexes) > 0:
        for i in indexes.flatten():
            result_boxes.append(boxes[i])
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
    print(f"Returning {len(result_boxes)} boxes after filtering.")
    print("result_boxes=",result_boxes) 
    print("result_confidences=",result_confidences)
    print("result_class_ids=",result_class_ids)
    print("class_ids=",class_ids)
    print("confidences=",confidences)
    print("boxes=",boxes)   
    print("len(result_boxes)=",len(result_boxes))

    return result_boxes, result_confidences, result_class_ids

def draw_yolov4_boxes(img, boxes, confidences, class_ids, classes):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

if __name__ == "__main__":
    yolov4_weights_path = "D:\\spa\\SFA3D\\sfa\\models\\yolov4.weights"
    yolov4_cfg_path = "D:\\spa\\SFA3D\\sfa\\models\\yolov4.cfg"
    yolov4_names_path = "D:\\spa\\SFA3D\\sfa\\models\\coco.names"

    net, classes, output_layers = load_yolov4_model(yolov4_weights_path, yolov4_cfg_path, yolov4_names_path)

    image_folder = r"D:\spa\SFA3D\dataset\kitti\training\image_2"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    idx = 0
    while idx < len(image_files):
        img_path = os.path.join(image_folder, image_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read image {img_path}, skipping...")
            idx += 1
            continue

        boxes, confidences, class_ids = yolov4_detect(image, net, output_layers)
        image_with_boxes = draw_yolov4_boxes(image, boxes, confidences, class_ids, classes)

        cv2.imshow("YOLOv4 Detection", image_with_boxes)
        print(f"Showing image {idx + 1}/{len(image_files)}: {image_files[idx]}")

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('n'):  # 'n' key for next image
            idx += 1
        else:
            print("Press 'n' for next image, ESC to quit.")

    cv2.destroyAllWindows()
