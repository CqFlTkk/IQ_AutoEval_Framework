import argparse
import os
import cv2
import yaml
# 把你的模块都导入进来
from core.iq_analyzer import bright_analyzer, judge_expose
from core.detector import ObjectDetector
from utils.report_gen import save2csv

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="YAML 配置文件路径")
    parser.add_argument("--input_dir", help="覆盖 YAML 中 io.input_dir")
    parser.add_argument("--output", help="覆盖 YAML 中 io.output_csv")
    parser.add_argument("--model_path", help="覆盖 YAML 中 model.path")
    return parser.parse_args()

def resolve_settings(args, cfg):
    model_path = args.model_path or cfg.get("model", {}).get("path", "yolov8n.pt")
    input_dir = args.input_dir or cfg.get("io", {}).get("input_dir")
    output_csv = args.output or cfg.get("io", {}).get("output_csv", "report.csv")
    image_extensions = tuple(
        ext.lower() for ext in cfg.get("runtime", {}).get("image_extensions", [".jpg", ".jpeg", ".png"])
    )

    # 从 config.yaml 的 exposure 节读取直方图截断率阈值
    exposure_cfg = cfg.get("exposure", {})
    exposure_params = {
        "overexpose_thresh": exposure_cfg.get("overexpose_thresh", 245),
        "underexpose_thresh": exposure_cfg.get("underexpose_thresh", 10),
        "over_ratio_limit": exposure_cfg.get("over_ratio_limit", 0.05),
        "under_ratio_limit": exposure_cfg.get("under_ratio_limit", 0.05),
    }

    if not input_dir:
        raise ValueError("缺少输入目录，请在 YAML 配置 io.input_dir 或通过 --input_dir 传入。")
    if not os.path.isdir(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")

    return model_path, input_dir, output_csv, image_extensions, exposure_params

def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_path, input_dir, output_csv, image_extensions, exposure_params = resolve_settings(args, cfg)

    # 1. 初始化检测器 (只在循环外初始化一次，否则每张图都加载一遍模型会卡死)
    detector = ObjectDetector(model_path)
    results = []

    for file in os.listdir(input_dir):
        if not file.lower().endswith(image_extensions):
            continue

        full_path = os.path.abspath(os.path.join(input_dir, file))

        # 2. 在主流程中统一读取一次图片
        img = cv2.imread(full_path)
        if img is None:
            print("读取图片失败:", full_path)
            continue

        # 3. 传给 YOLO 找框
        detected_boxes = detector.detect(img)

        # 4. 遍历这张图里找到的所有目标
        for obj in detected_boxes:
            bbox = obj["bbox"]
            cls_id = obj["cls"]

            # 5. 基于直方图截断率分析该 ROI 区域的曝光状态
            analysis = bright_analyzer(
                img, bbox,
                overexpose_thresh=exposure_params["overexpose_thresh"],
                underexpose_thresh=exposure_params["underexpose_thresh"]
            )
            if analysis is None:
                continue

            status = judge_expose(
                analysis,
                over_ratio_limit=exposure_params["over_ratio_limit"],
                under_ratio_limit=exposure_params["under_ratio_limit"]
            )
            over_pct = f'{analysis["over_ratio"] * 100:.2f}%'
            under_pct = f'{analysis["under_ratio"] * 100:.2f}%'
            print(f"{file} | 类别:{cls_id} | 过曝率:{over_pct} | 死黑率:{under_pct} | {status}")

            # 6. 把类别和坐标也记录下来，否则日后复盘你不知道测的是图里的哪个东西
            results.append([file, cls_id, str(bbox),
                            analysis["over_ratio"], analysis["under_ratio"],
                            analysis["mean_brightness"], status])

    if results:
        save2csv(results, output_csv)
        print("报告成功生成", output_csv)
    else:
        print("没有处理任何有效数据")

if __name__ == "__main__":
    main()