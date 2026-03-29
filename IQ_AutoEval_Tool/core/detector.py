from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model(image_path, verbose=False)

        boxes = []
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])

                boxes.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "cls": cls_id
                })

        return boxes