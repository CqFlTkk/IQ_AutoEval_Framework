import cv2
import numpy as np


def bright_analyzer(img, bbox,
                    overexpose_thresh=245,
                    underexpose_thresh=10):
    """
    基于直方图截断率(Histogram Clipping)的 ROI 曝光分析。

    Args:
        img: BGR 原图 (numpy array)
        bbox: YOLO 检测框坐标 (x1, y1, x2, y2)
        overexpose_thresh: 过曝灰度阈值，>= 该值视为溢出像素 (默认245)
        underexpose_thresh: 死黑灰度阈值，<= 该值视为死黑像素 (默认10)

    Returns:
        dict: {
            "over_ratio":  过曝像素占比 (0~1),
            "under_ratio": 死黑像素占比 (0~1),
            "mean_brightness": 平均亮度 (仅供参考记录)
        }
        若输入非法则返回 None
    """
    if img is None:
        return None

    x1, y1, x2, y2 = bbox

    # ROI 防越界极值钳位 (Clamping)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x1 >= x2 or y1 >= y2:
        return None  # 框不合法

    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    total_pixels = gray.shape[0] * gray.shape[1]
    if total_pixels == 0:
        return None

    # 利用 cv2.calcHist 提取 256 阶灰度直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # 过曝像素集合：灰度值 >= overexpose_thresh 的像素数量
    over_count = int(np.sum(hist[overexpose_thresh:]))
    # 死黑像素集合：灰度值 <= underexpose_thresh 的像素数量
    under_count = int(np.sum(hist[:underexpose_thresh + 1]))

    over_ratio = round(over_count / total_pixels, 4)
    under_ratio = round(under_count / total_pixels, 4)
    mean_brightness = round(float(np.mean(gray)), 2)  # 仅作辅助参考

    return {
        "over_ratio": over_ratio,
        "under_ratio": under_ratio,
        "mean_brightness": mean_brightness
    }


def judge_expose(analysis_result,
                 over_ratio_limit=0.05,
                 under_ratio_limit=0.05):
    """
    基于直方图截断率判定曝光状态。

    Args:
        analysis_result: bright_analyzer 返回的 dict
        over_ratio_limit:  过曝像素占比容忍水位线 (默认5%)
        under_ratio_limit: 死黑像素占比容忍水位线 (默认5%)

    Returns:
        str: "过曝" / "欠曝" / "过曝+欠曝" / "正常"
    """
    if analysis_result is None:
        return "无效"

    over = analysis_result["over_ratio"] >= over_ratio_limit
    under = analysis_result["under_ratio"] >= under_ratio_limit

    if over and under:
        return "过曝+欠曝"
    elif over:
        return "过曝"
    elif under:
        return "欠曝"
    else:
        return "正常"
