# IQ AutoEval Tool

基于 YOLO 检测目标区域并进行亮度评估（过曝/欠曝/正常），最终输出 CSV 报告。

## 功能说明

- 遍历输入目录中的图片（支持后缀可配置）
- 使用 YOLO 检测图像中的目标框
- 对每个目标框（ROI）计算亮度
- 按阈值判断曝光状态
- 导出检测结果到 CSV

## 项目结构

```text
IQ_AutoEval_Tool/
├─ main.py
├─ config.yaml
├─ core/
│  ├─ detector.py
│  └─ iq_analyzer.py
└─ utils/
   └─ report_gen.py
```

## 依赖安装

建议使用 Python 3.10+。

```bash
pip install ultralytics opencv-python numpy pyyaml
```

如果你使用 `requirements.txt`，请先补全依赖后再执行：

```bash
pip install -r requirements.txt
```

## 配置文件（YAML）

默认读取项目根目录下的 `config.yaml`。

```yaml
model:
  path: "yolov8n.pt"

io:
  input_dir: "./images"
  output_csv: "report.csv"

runtime:
  image_extensions:
    - ".jpg"
    - ".jpeg"
    - ".png"
```

### 配置项说明

- `model.path`：YOLO 模型路径（如 `yolov8n.pt`）
- `io.input_dir`：待评估图片目录
- `io.output_csv`：输出报告路径
- `runtime.image_extensions`：允许处理的图片后缀（小写）

## 运行方式

### 1）使用默认配置运行

```bash
python main.py
```

### 2）指定配置文件

```bash
python main.py --config config.yaml
```

### 3）命令行覆盖 YAML 配置

```bash
python main.py --input_dir ./images_test --output report_test.csv --model_path yolov8n.pt
```

参数优先级：命令行参数 > YAML 配置。

## 输出结果

生成 CSV，表头如下：

- `image_name`：图片文件名
- `class_id`：检测类别 ID
- `roi_box`：目标框坐标 `[x1, y1, x2, y2]`
- `brightness`：ROI 平均亮度
- `status`：曝光状态（`过曝` / `欠曝` / `正常`）

## 曝光判定规则

当前默认逻辑位于 `core/iq_analyzer.py`：

- `brightness > 220` -> `过曝`
- `brightness < 60` -> `欠曝`
- 其他 -> `正常`

如需调整阈值，可修改 `judge_expose()`。

## 常见问题

- 输入目录不存在：请检查 `io.input_dir` 或 `--input_dir`
- 无法读取图片：请确认图片格式受支持且文件未损坏
- 模型加载失败：请确认 `model.path` 指向有效模型文件
