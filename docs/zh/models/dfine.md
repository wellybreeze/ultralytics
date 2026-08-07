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

[D-FINE](https://github.com/Peterande/D-FINE)（论文：[*D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement*](https://arxiv.org/abs/2410.13842)）是一种实时 DETR 风格[目标检测](https://www.ultralytics.com/glossary/object-detection)器，将边界框回归重定义为**细粒度分布精炼（FDR）**，并引入**全局最优定位自蒸馏（GO-LSD）**。Ultralytics 以类似 [RT-DETR](../../en/models/rtdetr.md) 的方式原生集成 D-FINE：YAML 构图、`DFINEDecoder`、YOLO 格式数据，以及标准的[训练](../../en/modes/train.md) / [验证](../../en/modes/val.md) / [预测](../../en/modes/predict.md) / [导出](../../en/modes/export.md)流程。

官方预训练权重采用 HGNetv2 骨干 + HybridEncoder（`RepNCSPELAN4` / `SCDown`）与 D-FINE Transformer 解码器。本仓库提供的 Ultralytics `.pt` 在相同预处理下（640×640 方形缩放 + `/255`，无 ImageNet normalize）与官方 fp32 输出数值对齐。

### 主要特性

- **细粒度分布精炼（FDR）：** 框边界以离散分布（`reg_max`）精炼，而非单一 L1/GIoU 偏移，有利于模糊边缘与密集场景定位。
- **GO-LSD：** 解码层间自蒸馏提升定位精度，且不增加推理开销。
- **无 NMS 的 DETR 头：** 端到端集合预测；Ultralytics 后处理使用置信度过滤（与 RT-DETR 相同）。
- **原生 Ultralytics 管线：** `DFINE` 门面、`DFINEDetectionModel`、YOLO `data=*.yaml` 训练，以及便于 ONNX/TensorRT 的导出。
- **多尺度覆盖：** Nano → XLarge（`dfine-n/s/m/l/x`），并提供 COCO、Objects365、Objects365→COCO 预训练权重。

## 模型库

指标与延迟来自[官方 D-FINE Model Zoo](https://github.com/Peterande/D-FINE)（COCO val2017；T4、`batch_size=1`、fp16、TensorRT）。Ultralytics `.pt` 权重托管于 [Release v2.0.0](https://github.com/wellybreeze/ultralytics/releases/tag/v2.0.0)。

### COCO

| 模型 | 数据集 | AP<sup>val</sup> | 参数量 | 时延 (ms) | GFLOPs | 配置 | 权重 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-N | COCO | **42.8** | 4M | 2.12ms | 7 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-n.yaml) | [42.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-n.pt) |
| D-FINE-S | COCO | **48.5** | 10M | 3.49ms | 25 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [48.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s.pt) |
| D-FINE-M | COCO | **52.3** | 19M | 5.62ms | 57 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [52.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m.pt) |
| D-FINE-L | COCO | **54.0** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [54.0](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l.pt) |
| D-FINE-X | COCO | **55.8** | 62M | 12.89ms | 202 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [55.8](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x.pt) |

### Objects365+COCO

| 模型 | 数据集 | AP<sup>val</sup> | 参数量 | 时延 (ms) | GFLOPs | 配置 | 权重 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-S | Objects365+COCO | **50.7** | 10M | 3.49ms | 25 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [50.7](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj2coco.pt) |
| D-FINE-M | Objects365+COCO | **55.1** | 19M | 5.62ms | 57 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [55.1](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj2coco.pt) |
| D-FINE-L | Objects365+COCO | **57.3** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [57.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj2coco-e25.pt) |
| D-FINE-X | Objects365+COCO | **59.3** | 62M | 12.89ms | 202 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [59.3](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj2coco.pt) |

**我们强烈推荐您使用 Objects365 预训练模型进行微调：**

⚠️ 重要提醒：通常这种预训练模型对复杂场景的理解非常有用。如果您的类别非常简单，请注意，这可能会导致过拟合和次优性能。

<details>
<summary><strong>🔥 Objects365 预训练模型（泛化性最好）</strong></summary>

| 模型 | 数据集 | AP<sup>val</sup> | AP<sup>5000</sup> | 参数量 | 时延 (ms) | GFLOPs | 配置 | 权重 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-S | Objects365 | **31.0** | **30.5** | 10M | 3.49ms | 25 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-s.yaml) | [30.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-s-obj365.pt) |
| D-FINE-M | Objects365 | **38.6** | **37.4** | 19M | 5.62ms | 57 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-m.yaml) | [37.4](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-m-obj365.pt) |
| D-FINE-L | Objects365 | - | **40.6** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [40.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365.pt) |
| D-FINE-L (E25) | Objects365 | **44.7** | **42.6** | 31M | 8.07ms | 91 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-l.yaml) | [42.6](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-l-obj365-e25.pt) |
| D-FINE-X | Objects365 | **49.5** | **46.5** | 62M | 12.89ms | 202 | [yaml](../../../ultralytics/cfg/models/dfine/dfine-x.yaml) | [46.5](https://github.com/wellybreeze/ultralytics/releases/download/v2.0.0/dfine-x-obj365.pt) |

- **E25**：官方将训练延长至 25 个 epoch 的重训版本。
- **AP<sup>val</sup>** 是在 *Objects365* 完整验证集上评估的。
- **AP<sup>5000</sup>** 是在 *Objects365* 验证集前 5000 个样本上评估的。

</details>

**注意：**

- **AP<sup>val</sup>**（COCO / Objects365+COCO 表）是在 *MSCOCO val2017* 上评估的。
- **时延** 是在单张 T4 GPU 上以 $batch\_size = 1$、fp16、TensorRT 评估的（见官方说明）。
- **Objects365+COCO** 表示使用在 *Objects365* 上预训练的权重再在 *COCO* 上微调的模型。
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
> 与官方 D-FINE 推理一致：将图像方形缩放到 `imgsz`（默认 640），使用 `LetterBox(scale_fill=True)`，再除以 255。Ultralytics 路径下**不要**做 ImageNet mean/std 归一化。

## 支持的任务与模式

| 模型类型 | 配置 / 权重                       | 支持任务 | 推理 | 验证 | 训练 | 导出 |
| -------- | --------------------------------- | -------- | ---- | ---- | ---- | ---- |
| D-FINE-N | `dfine-n.yaml` / `dfine-n.pt`     | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-S | `dfine-s.yaml` / `dfine-s.pt`     | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-M | `dfine-m.yaml` / `dfine-m.pt`     | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-L | `dfine-l.yaml` / `dfine-l.pt`     | 目标检测 | ✅   | ✅   | ✅   | ✅   |
| D-FINE-X | `dfine-x.yaml` / `dfine-x.pt`     | 目标检测 | ✅   | ✅   | ✅   | ✅   |

架构 YAML 位于 `ultralytics/cfg/models/dfine/`。

## 架构说明

- **骨干：** HGNetv2，以 Ultralytics `HGStem` / `HGBlock` / `DWConv` 拆层实现。
- **编码器：** 官方 HybridEncoder 对应 `DFINERepNCSPELAN4` / `DFINESCDown`（不要与 YOLO 的 `RepNCSPELAN4` / `SCDown` 混用）。
- **解码器：** `DFINEDecoder` 实现 FDR（`Integral`、LQE、Gate）与 CDN 训练查询。
- **损失：** `DFINEDetectionLoss`（VFL + L1/GIoU + FGL/DDF + GO-LSD）。
- **训练/验证/预测：** 与 RT-DETR 共享 DETR 风格数据管线（`rect=False`、scale-fill letterbox、无 NMS）。

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
