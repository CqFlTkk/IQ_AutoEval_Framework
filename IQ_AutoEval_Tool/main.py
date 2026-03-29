import argparse
import os
import cv2
# 把你的模块都导入进来
from core.iq_analyzer import bright_analyzer, judge_expose
from core.detector import ObjectDetector
from utils.report_gen import save2csv

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--output", default="report.csv")
args = parser.parse_args()

# 1. 初始化检测器 (只在循环外初始化一次，否则每张图都加载一遍模型会卡死)
detector = ObjectDetector("yolov8n.pt")
results = []

for file in os.listdir(args.input_dir):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        full_path = os.path.abspath(os.path.join(args.input_dir, file))

        # 2. 在主流程中统一读取一次图片
        img = cv2.imread(full_path)
        if img is None:
            print("读取图片失败:", full_path)
            continue

        # 3. 传给 YOLO 找框
        # 注意：你需要稍微改一下你的 detector.py，让它能接收 img 而不是 path
        # 也就是把 results = self.model(image_path) 改成 results = self.model(img)
        detected_boxes = detector.detect(img)

        # 4. 遍历这张图里找到的所有目标
        for obj in detected_boxes:
            bbox = obj["bbox"]
            cls_id = obj["cls"]

            # 5. 计算该框(ROI)区域的亮度
            brightness = bright_analyzer(img, bbox)
            if brightness is None:
                continue

            status = judge_expose(brightness)
            print(f"{file} | 类别:{cls_id} | 亮度:{brightness} | {status}")

            # 6. 把类别和坐标也记录下来，否则日后复盘你不知道测的是图里的哪个东西
            results.append([file, cls_id, str(bbox), brightness, status])

if results:
    # 注意：你的 save2csv 的表头要跟着更新
    # writer.writerow(["image", "class_id", "bbox", "bright", "status"])
    save2csv(results, args.output)
    print("报告成功生成", args.output)
else:
    print("没有处理任何有效数据")