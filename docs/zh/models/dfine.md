---
title: D-FINE：细粒度分布精炼检测 Transformer
comments: true
description: 在 Ultralytics 中使用 D-FINE——基于细粒度分布精炼（FDR）的实时 DETR 目标检测器。
keywords: D-FINE, DFINE, DETR, FDR, GO-LSD, Ultralytics, 目标检测, Transformer, HGNetv2, 实时检测
---

# D-FINE：细粒度分布精炼检测 Transformer

> **English docs**  
> English version: [docs/en/models/dfine.md](../../en/models/dfine.md)

## 概述

[D-FINE](https://github.com/Peterande/D-FINE)（论文：[_D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement_](https://arxiv.org/abs/2410.13842)）是一种实时 DETR 风格[目标检测](https://www.ultralytics.com/glossary/object-detection)器，将边界框回归重定义为**细粒度分布精炼（FDR）**，并引入**全局最优定位自蒸馏（GO-LSD）**。Ultralytics 以类似 [RT-DETR](../../en/models/rtdetr.md) 的方式原生集成 D-FINE：YAML 构图、`DFINEDecoder`、YOLO 格式数据，以及标准的[训练](../../en/modes/train.md) / [验证](../../en/modes/val.md) / [预测](../../en/modes/predict.md) / [导出](../../en/modes/export.md)流程。

官方预训练权重采用 HGNetv2 骨干 + HybridEncoder（`RepNCSPELAN4` / `SCDown`）与 D-FINE Transformer 解码器。本仓库提供的 Ultralytics `.pt` 在相同预处理下（640×640 方形缩放 + `/255`，无 ImageNet normalize）与官方 fp32 输出数值对齐。

### 主要特性

- **细粒度分布精炼（FDR）：** 框边界以离散分布（`reg_max`）精炼，而非单一 L1/GIoU 偏移，有利于模糊边缘与密集场景定位。
- **GO-LSD：** 解码层间自蒸馏提升定位精度，且不增加推理开销。
- **无 NMS 的 DETR 头：** 端到端集合预测；Ultralytics 后处理使用置信度过滤（与 RT-DETR 相同）。
- **原生 Ultralytics 管线：** `DFINE` 门面、`DFINEDetectionModel`、YOLO `data=*.yaml` 训练，以及便于 ONNX/TensorRT 的导出。Objects365 权重也可作为 [WeDetect](wedetect.md) 伪标签教师。
- **多尺度覆盖：** Nano → XLarge（`dfine-n/s/m/l/x`），并提供 COCO、Objects365、Objects365→COCO 预训练权重。

## 模型库

指标与延迟来自[官方 D-FINE Model Zoo](https://github.com/Peterande/D-FINE)（COCO val2017；T4、`batch_size=1`、fp16、TensorRT）。Ultralytics `.pt` 权重托管于 [Release v2.0.0](https://github.com/wellybreeze/ultralytics/releases/tag/v2.0.0)。

### COCO

|   模型   | 数据集 | AP<sup>val</sup> | 参数量 | 时延 (ms) | GFLOPs |                            配置                            |                                          权重                                          |
| :------: | :----: | :--------------: | :----: | :-------: | :----: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| D-FINE-N |  COCO  |     **42.8**     |   4M   |  2.12ms   |   7    | [yaml](../../../ultralytics/cfg/models/dfine/dfine-n.yaml) | [42.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-n.pt) |
| D-FINE-S |  COCO  |     **48.5**     |  10M   |  3.49ms   |   25   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [48.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s.pt) |
| D-FINE-M |  COCO  |     **52.3**     |  19M   |  5.62ms   |   57   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [52.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m.pt) |
| D-FINE-L |  COCO  |     **54.0**     |  31M   |  8.07ms   |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [54.0](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt) |
| D-FINE-X |  COCO  |     **55.8**     |  62M   |  12.89ms  |  202   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [55.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x.pt) |

### Objects365+COCO

|   模型   |     数据集      | AP<sup>val</sup> | 参数量 | 时延 (ms) | GFLOPs |                            配置                            |                                                权重                                                 |
| :------: | :-------------: | :--------------: | :----: | :-------: | :----: | :--------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| D-FINE-S | Objects365+COCO |     **50.7**     |  10M   |  3.49ms   |   25   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) |   [50.7](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj2coco.pt)   |
| D-FINE-M | Objects365+COCO |     **55.1**     |  19M   |  5.62ms   |   57   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) |   [55.1](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj2coco.pt)   |
| D-FINE-L | Objects365+COCO |     **57.3**     |  31M   |  8.07ms   |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [57.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj2coco-e25.pt) |
| D-FINE-X | Objects365+COCO |     **59.3**     |  62M   |  12.89ms  |  202   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) |   [59.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj2coco.pt)   |

**我们强烈推荐您使用 Objects365 预训练模型进行微调：**

⚠️ 重要提醒：通常这种预训练模型对复杂场景的理解非常有用。如果您的类别非常简单，请注意，这可能会导致过拟合和次优性能。

<details>
<summary><strong>🔥 Objects365 预训练模型（泛化性最好）</strong></summary>

|      模型      |   数据集   | AP<sup>val</sup> | AP<sup>5000</sup> | 参数量 | 时延 (ms) | GFLOPs |                            配置                            |                                               权重                                                |
| :------------: | :--------: | :--------------: | :---------------: | :----: | :-------: | :----: | :--------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
|    D-FINE-S    | Objects365 |     **31.0**     |     **30.5**      |  10M   |  3.49ms   |   25   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) |   [30.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj365.pt)   |
|    D-FINE-M    | Objects365 |     **38.6**     |     **37.4**      |  19M   |  5.62ms   |   57   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) |   [37.4](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj365.pt)   |
|    D-FINE-L    | Objects365 |        -         |     **40.6**      |  31M   |  8.07ms   |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) |   [40.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365.pt)   |
| D-FINE-L (E25) | Objects365 |     **44.7**     |     **42.6**      |  31M   |  8.07ms   |   91   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [42.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365-e25.pt) |
|    D-FINE-X    | Objects365 |     **49.5**     |     **46.5**      |  62M   |  12.89ms  |  202   | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) |   [46.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj365.pt)   |

- **E25**：官方将训练延长至 25 个 epoch 的重训版本。
- **AP<sup>val</sup>** 是在 _Objects365_ 完整验证集上评估的。
- **AP<sup>5000</sup>** 是在 _Objects365_ 验证集前 5000 个样本上评估的。

</details>

**注意：**

- **AP<sup>val</sup>**（COCO / Objects365+COCO 表）是在 _MSCOCO val2017_ 上评估的。
- **时延** 是在单张 T4 GPU 上以 $batch\_size = 1$、fp16、TensorRT 评估的（见官方说明）。
- **Objects365+COCO** 表示使用在 _Objects365_ 上预训练的权重再在 _COCO_ 上微调的模型。
- YAML 说明：**N/S/M** 开启 Learnable Affine Block（`use_lab=True`）；**L/X** 设置 `freeze_norm: true` 以匹配官方 `FrozenBatchNorm2d`。

## 使用示例

### Python

```python
from ultralytics import DFINE

# 从 Release v2.0.0 加载 Ultralytics 预训练权重（也可传本地路径或 YAML）
model = DFINE("https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt")
# model = DFINE("dfine-l.yaml")  # 随机初始化

# 显示模型信息（可选）
model.info()

# 在 YOLO 格式数据集上训练
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# 验证
metrics = model.val(data="coco8.yaml")

# 预测（方形 scale-fill letterbox，无 NMS）
results = model("path/to/bus.jpg")

# 导出
path = model.export(format="onnx")
```

### CLI

```bash
# 下载权重后训练 / 预测（文件名含 "dfine" 时自动选择 DFINE）
wget https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt
yolo detect train model=dfine-l.pt data=coco8.yaml epochs=100 imgsz=640
yolo predict model=dfine-l.pt source=path/to/bus.jpg
yolo val model=dfine-l.pt data=coco8.yaml
yolo export model=dfine-l.pt format=onnx
```

> **预处理**  
> **预测 / 伪标签教师** 对齐官方 D-FINE：`LetterBox(scale_fill=True)` 拉伸到方形 `imgsz`，再 `/255`，**不要**做 ImageNet mean/std。  
> **验证** 用带 padding 的 `LetterBox(scale_fill=False, scaleup=False)`。**训练** 用 mosaic / `RandomPerspective` 仿射到方形，不是官方 scale-fill。与官方 zoo 对比 val mAP 时注意此差异。

## 支持的任务与模式

| 模型类型 | 配置 / 权重                   | 支持任务 | 推理 | 验证 | 训练 | 导出 |
| -------- | ----------------------------- | -------- | ---- | ---- | ---- | ---- |
| D-FINE-N | `dfine-n.yaml` / `dfine-n.pt` | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-S | `dfine-s.yaml` / `dfine-s.pt` | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-M | `dfine-m.yaml` / `dfine-m.pt` | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-L | `dfine-l.yaml` / `dfine-l.pt` | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-X | `dfine-x.yaml` / `dfine-x.pt` | 目标检测 | ✅   | ✅   | ✅   | ✅   |

架构 YAML 位于 `ultralytics/cfg/models/dfine/`。

## 全流程

```text
YOLO 数据 YAML  →  DFINE.train()  →  best.pt
        ↓
DFINE.val() / DFINE.predict() / DFINE.export()
        ↓
可选：作为 WeDetect 伪标签教师（Objects365 权重）
```

`YOLO("dfine-*.pt")` 在检测头为 `DFINEDecoder` 时会变成 `DFINE`。推荐 `from ultralytics import DFINE`。需要 `torch>=1.11`。

### 训练

`DFINETrainer` 复用 RT-DETR 循环（`rect=False`，DETR 风格 batch 供 CDN）。日志里的 loss 名为 `giou_loss` / `cls_loss` / `l1_loss`（FGL/DDF 计入总 loss 但不显示）；实际准则为 `DFINEDetectionLoss`：

| 项     | 默认权重        | 作用           |
| ------ | --------------- | -------------- |
| VFL    | `loss_vfl=1.0`  | 分类           |
| L1 box | `loss_bbox=5.0` | 框回归         |
| GIoU   | `loss_giou=2.0` | 重叠           |
| FGL    | `loss_fgl=0.15` | FDR 细粒度定位 |
| DDF    | `loss_ddf=1.5`  | GO-LSD 蒸馏    |

```python
from ultralytics import DFINE

model = DFINE("pretrained_weights/dfine-x-obj365.pt")  # 自定义/密集场景建议 Obj365
model.train(data="coco8.yaml", epochs=50, imgsz=640, batch=4, device=0)
```

YAML：

- **N/S/M：** `use_lab=True`（Learnable Affine Block）
- **L/X：** `freeze_norm: true` → 骨干 FrozenBatchNorm2d
- 解码器默认 `nq=300`、`reg_max=32`。改 `nc` 会重初始化分类头；Objects365 权重为 `nc=366`

`fuse()` 对齐官方 `DFINE.deploy()`，不是整网 YOLO Conv+BN fuse。AMP 可能导致 NaN / 匹配失败（与 RT-DETR 相同）；`F.grid_sample` 不支持 `deterministic=True`。

### 验证与预测

预处理 **按模式不同**：

| 模式                 | 缩放                                                                |
| -------------------- | ------------------------------------------------------------------- |
| 预测 / WeDetect 教师 | `LetterBox(..., scale_fill=True)` — 拉伸成方（官方推理）            |
| 验证                 | `LetterBox(..., scale_fill=False, scaleup=False)` — 保比例、padding |
| 训练                 | Mosaic + `RandomPerspective` 到方形 `imgsz`                         |

均 `/255`、无 ImageNet mean/std。后处理只做置信度过滤（无 NMS）。`max_det` 只限制返回条数，query 数仍是 `nq`（默认 300）。导出时 `max_det` 会钳到 `num_queries`，并 **强制 `nms=False`**。

**非 640 `imgsz`：** `DFINEDecoder` 在特征图 `spatial_shapes` 变化时重建 log-anchor（`self.shapes` 缓存；N 版 YAML 可能预置 `eval_spatial_size=[640,640]`，首次 forward 会覆盖）。不要假设 anchors 永远按 640 固化。

### 类别名

| 权重                           | 头 `nc` | `names`                                          |
| ------------------------------ | ------- | ------------------------------------------------ |
| COCO `dfine-*.pt`              | 80      | COCO 80，id `0..79`                              |
| Objects365 `dfine-*-obj365.pt` | 366     | 下标 `0` = `background`；数据集类 `i` → 头 `i+1` |

加载后 `ensure_dfine_class_names` 会替换占位 `{i: str(i)}`。作为 **WeDetect 教师** 时，按英文 `pseudo_label_classes` 对齐到上述 head 名，而不是 YOLO 本地 id。

### 导出

与 RT-DETR 同一导出路径（`DFINEDecoder` 在 `_DETR_DECODERS`）。ONNX 需要 **opset ≥ 16**。TensorFlow 导出需要 opset 16–19。DETR 头 **强制 `nms=False`**。该系列 CoreML 不支持 `dynamic=True`。

```python
from ultralytics import DFINE

DFINE("dfine-l.pt").export(format="onnx", imgsz=640, opset=17)
```

### 作为 WeDetect 伪标签教师

写在 WeDetect **train 子集** YAML（见 [WeDetect](wedetect.md)）：

```yaml
pseudo_label: true
pseudo_label_model: ./pretrained_weights/dfine-x-obj365.pt
pseudo_label_classes: Objects365.yaml
pseudo_label_class_texts: texts/objects365_zh_class_texts.json
pseudo_label_conf: 0.2
pseudo_label_imgsz: 640 # 与 WeDetect 训练 imgsz 无关
```

教师走 `DFINE.predict`（scale-fill、无 NMS）。即使 WeDetect 以 1280 训练，教师默认仍是 640，除非改 `pseudo_label_imgsz`。若教师在 **已写入部分 cache 之后** OOM，不会自动减半 batch——请调低 `pseudo_label_batch` 后重跑。

## 架构说明

- **骨干：** HGNetv2，以 Ultralytics `HGStem` / `HGBlock` / `DWConv` 拆层实现。
- **编码器：** 官方 HybridEncoder 对应 `DFINERepNCSPELAN4` / `DFINESCDown`（不要与 YOLO 的 `RepNCSPELAN4` / `SCDown` 混用）。
- **解码器：** `DFINEDecoder` 实现 FDR（`Integral`、LQE、Gate）与 CDN 训练查询。
- **损失：** `DFINEDetectionLoss`（VFL + L1/GIoU + FGL/DDF + GO-LSD）。
- **训练/验证/预测：** 与 RT-DETR 共享 DETR 风格数据管线（`rect=False`、无 NMS）。**预测** 用 scale-fill letterbox；**验证** 用 padding letterbox；**训练** 用 mosaic / affine。`imgsz` 变化时重建 anchors。
- **BN：** 将 `BatchNorm2d` 重置为官方 `eps=1e-5`、`momentum=0.1`（不是 YOLO 的 `1e-3` / `0.03`）。

## 引用与致谢

若在研究或产品中使用 D-FINE，请引用[原论文](https://arxiv.org/abs/2410.13842)：

**BibTeX**

```bibtex
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

感谢 [D-FINE 作者](https://github.com/Peterande/D-FINE) 开源实现与预训练权重。

## 常见问题

### D-FINE 与 Ultralytics 中的 RT-DETR 有何不同？

二者都是无 NMS 的 DETR 检测器，并采用类似的 Ultralytics 门面。D-FINE 用 FDR 分布回归替代普通框回归，并使用 GO-LSD / FGL-DDF 损失。为实现权重对齐，编码器必须使用官方语义的 `DFINERepNCSPELAN4` / `DFINESCDown`。

### `.pt` 权重从哪里获取？

从 [Release v2.0.0](https://github.com/wellybreeze/ultralytics/releases/tag/v2.0.0) 下载，或见上方[模型库](#模型库)表格中的权重链接。指标与官方一致；权重为已对齐的 Ultralytics 格式。

### 能否在自定义 YOLO 数据集上训练？

可以。向 `model.train(data=...)` 传入任意 Ultralytics 检测数据 YAML 即可。标签为标准 YOLO 格式；训练器使用与 RT-DETR 相同的 DETR 风格数据路径（`rect=False`）。

### 为什么 L/X 配置要设置 `freeze_norm: true`？

官方 L/X 的 HGNetv2 使用 `FrozenBatchNorm2d`（`freeze_norm=True`）。Ultralytics 同步该融合 `rsqrt` 路径，使转换权重与官方前向数值一致。

### `max_det` 会增加 D-FINE 的 query 数量吗？

不会。与 RT-DETR 相同，`max_det` 只限制返回检测数。解码器仍输出固定数量的 query（默认 300）。若需要更多 query，请在 YAML 中提高 `nq` 并重新训练。

### 640 训练后能否用 1280 推理？

可以。解码器按当前特征图尺寸重建 anchors。WeDetect 把 D-FINE 当教师并设 `pseudo_label_imgsz=1280` 时走同一路径。

### Objects365 的类别下标怎么对应？

官方 Obj365 权重有 366 维：0 为 background，其后 365 类整体 +1。`DFINE("dfine-x-obj365.pt").names[0]` 是 `"background"`。用 `classes=` 过滤时要用这些 head 下标，不要用 Objects365 yaml 的 0 起始 id。

### 能否把 D-FINE 当作 WeDetect 伪标签教师？

可以。在子集 YAML 设 `pseudo_label_model` 为 D-FINE `.pt`，`pseudo_label_classes` 用 `Objects365.yaml`（COCO 权重则用 COCO names）。见 [WeDetect](wedetect.md) 与 [OV 微调教程](../guides/wedetect-ov-finetune.md)。
