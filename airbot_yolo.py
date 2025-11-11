import cv2
import yaml
from ultralytics import YOLO

class AirbotYolo:
    def __init__(self):
        with open("configs/config_file.yaml", "r") as file:
            config_path = yaml.safe_load(file)["Path"]
        config = yaml.safe_load(open(config_path, "r"))
        
        ckpt_path = config["AirbotYolo"]["checkpoint"]        
        self.verbose = config["AirbotYolo"]["verbose"]
        
        self.model = YOLO(model=ckpt_path)

        self.result = None
    
    def inference(self, image):
        self.result = self.model(image, verbose=self.verbose)
        return self.result
    
    def get_max_conf_bbox_and_label(self, image, label_filter: list[str] = []):
        results = self.inference(image)
        max_conf = 0
        bbox = None
        label = None
        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            for box in boxes:
                bbox_raw = box.xyxy[0].flatten().tolist()
                if bbox_raw[0] < 0.05 * image.shape[1] or bbox_raw[2] > 0.95 * image.shape[1] or bbox_raw[1] < 0.05 * image.shape[1] or bbox_raw[3] > 0.95 * image.shape[0]:
                    continue
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_label = self.model.names[cls_id]
                if conf > max_conf and cls_label not in label_filter:
                    max_conf = conf
                    bbox = [int(x) for x in bbox_raw]
                    label = cls_label

        return bbox, label, max_conf

def draw_bbox(img, bbox, label, conf):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def main():
    cam = cv2.VideoCapture(2)

    if not cam.isOpened():
        print("无法打开摄像头")
        return

    airbot_yolo = AirbotYolo()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("无法读取图像")
            break

        # 直接使用 YOLO 原始推理结果
        results = airbot_yolo.inference(frame)
        result = results[0]  # 默认只取第一帧结果

        if result.boxes is not None:
            for box in result.boxes:
                bbox_raw = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, bbox_raw)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = airbot_yolo.model.names[cls_id]

                draw_bbox(frame, [x1, y1, x2, y2], label, conf)

        cv2.imshow("Airbot YOLO - All Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
