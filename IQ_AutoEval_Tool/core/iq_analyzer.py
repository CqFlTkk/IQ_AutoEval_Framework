import cv2
import numpy as np


# 接收 numpy数组(img) 和 框坐标(bbox)
def bright_analyzer(img, bbox):
    if img is None:
        return None

    x1, y1, x2, y2 = bbox

    # 核心：根据 YOLO 给的坐标裁剪 ROI (注意防止越界)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x1 >= x2 or y1 >= y2:
        return None  # 框不合法

    roi = img[y1:y2, x1:x2]  # 切片拿到 ROI

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright = np.mean(gray)
    return round(bright, 2)  # 保留两位小数，CSV好看点


def judge_expose(brightness):
    # ... 你的原逻辑不变
    if brightness > 220:
        return "过曝"
    elif brightness < 60:
        return "欠曝"
    else:
        return "正常"