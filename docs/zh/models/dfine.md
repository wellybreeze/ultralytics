---
title: D-FINE：细粒度分布精炼检测 Transformer
comments: true
description: 在 Ultralytics 中使用 D-FINE——基于细粒度分布精炼（FDR）的实时 DETR 目标检测器。
keywords: D-FINE, DFINE, DETR, FDR, GO-LSD, Ultralytics, 目标检测, Transformer, HGNetv2, 实时检测
---

# D-FINE：细粒度分布精炼检测 Transformer

!!! tip "English docs"

    English version: [docs/en/models/dfine.md](../../en/models/dfine.md)

## 概述

[D-FINE](https://github.com/Peterande/D-FINE)（论文：[*D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement*](https://arxiv.org/abs/2410.13842)）是一种实时 DETR 风格[目标检测](https://www.ultralytics.com/glossary/object-detection)器，将边界框回归重定义为**细粒度分布精炼（FDR）**，并引入**全局最优定位自蒸馏（GO-LSD）**。Ultralytics 以类似 [RT-DETR](../../en/models/rtdetr.md) 的方式原生集成 D-FINE：YAML 构图、`DFINEDecoder`、YOLO 格式数据，以及标准的[训练](../../en/modes/train.md) / [验证](../../en/modes/val.md) / [预测](../../en/modes/predict.md) / [导出](../../en/modes/export.md)流程。

官方预训练权重采用 HGNetv2 骨干 + HybridEncoder（`RepNCSPELAN4` / `SCDown`）与 D-FINE Transformer 解码器。转换后的 Ultralytics `.pt` 在相同预处理下（640×640 方形缩放 + `/255`，无 ImageNet normalize）与官方 fp32 输出数值对齐。

### 主要特性

- **细粒度分布精炼（FDR）：** 框边界以离散分布（`reg_max`）精炼，而非单一 L1/GIoU 偏移，有利于模糊边缘与密集场景定位。
- **GO-LSD：** 解码层间自蒸馏提升定位精度，且不增加推理开销。
- **无 NMS 的 DETR 头：** 端到端集合预测；Ultralytics 后处理使用置信度过滤（与 RT-DETR 相同）。
- **原生 Ultralytics 管线：** `DFINE` 门面、`DFINEDetectionModel`、YOLO `data=*.yaml` 训练，以及便于 ONNX/TensorRT 的导出。
- **多尺度覆盖：** Nano → XLarge（`dfine-n/s/m/l/x`），并可通过权重转换使用 Objects365 / Objects365→COCO 权重。

## 官方 COCO 性能

下表数据来自[官方 D-FINE Model Zoo](https://github.com/Peterande/D-FINE)（COCO val2017，T4 上 TensorRT 延迟）。Ultralytics 转换后保持架构与权重对齐；实际 FPS 取决于导出后端。

| 模型     | AP<sup>val</sup> | 参数量 | 延迟 (T4) | GFLOPs |
| -------- | ---------------- | ------ | --------- | ------ |
| D-FINE-N | 42.8             | 4M     | 2.12 ms   | 7      |
| D-FINE-S | 48.5             | 10M    | 3.49 ms   | 25     |
| D-FINE-M | 52.3             | 19M    | 5.62 ms   | 57     |
| D-FINE-L | 54.0             | 31M    | 8.07 ms   | 91     |
| D-FINE-X | 55.8             | 62M    | 12.89 ms  | 202    |

Objects365 预训练及 Objects365→COCO 微调权重（通常具有更高 COCO AP）也可在转换后使用，见[权重转换](#权重转换)。

## 使用示例

!!! example

    === "Python"

        ```python
        from ultralytics import DFINE

        # 加载已转换的 Ultralytics 权重（或 YAML 随机初始化）
        model = DFINE("dfine-l.pt")  # 或 "dfine-l.yaml"

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

    === "CLI"

        ```bash
        # 训练
        yolo detect train model=dfine-l.pt data=coco8.yaml epochs=100 imgsz=640

        # 预测（文件名含 "dfine" 时自动选择 DFINE）
        yolo predict model=dfine-l.pt source=path/to/bus.jpg

        # 验证 / 导出
        yolo val model=dfine-l.pt data=coco8.yaml
        yolo export model=dfine-l.pt format=onnx
        ```

!!! tip "预处理"

    与官方 D-FINE 推理一致：将图像方形缩放到 `imgsz`（默认 640），使用 `LetterBox(scale_fill=True)`，再除以 255。Ultralytics 路径下**不要**做 ImageNet mean/std 归一化。

## 权重转换

官方发布格式为 `.pth`（权重通常在 `model` / `ema` 下）。使用仓库自带脚本转为 Ultralytics `.pt`（默认 fp32，便于数值对齐）：

```bash
# 在 ultralytics 仓库根目录执行
python tools/convert_dfine_weights.py \
  --weights path/to/dfine_l_coco.pth \
  --cfg dfine-l.yaml \
  --out dfine-l.pt

# Objects365 权重（类别数默认从官方 head 推断为 366）
python tools/convert_dfine_weights.py \
  --weights path/to/dfine_l_obj365.pth \
  --cfg dfine-l.yaml \
  --out dfine-l-obj365.pt
```

| 参数        | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| `--weights` | 官方 `.pth` 路径                                             |
| `--cfg`     | Ultralytics YAML（`dfine-n/s/m/l/x.yaml`）                   |
| `--out`     | 输出 `.pt` 路径                                              |
| `--nc`      | 可选类别数（省略时从 `enc_score_head` 自动推断）             |
| `--fp16`    | 保存半精度（做数值对齐时不推荐）                             |
| `--dry-run` | 仅打印键映射                                                 |

YAML 说明：

- **N/S/M** 开启 Learnable Affine Block（`use_lab=True`）以匹配官方 HGNetv2。
- **L/X** 设置 `freeze_norm: true`，使骨干 BN 与官方 `FrozenBatchNorm2d` 一致。

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

!!! quote ""

    === "BibTeX"

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

使用 `tools/convert_dfine_weights.py` 转换官方 `.pth`（见[权重转换](#权重转换)）。与官方模型做数值对齐时请保持 fp32。

### 能否在自定义 YOLO 数据集上训练？

可以。向 `model.train(data=...)` 传入任意 Ultralytics 检测数据 YAML 即可。标签为标准 YOLO 格式；训练器使用与 RT-DETR 相同的 DETR 风格数据路径（`rect=False`）。

### 为什么 L/X 配置要设置 `freeze_norm: true`？

官方 L/X 的 HGNetv2 使用 `FrozenBatchNorm2d`（`freeze_norm=True`）。Ultralytics 同步该融合 `rsqrt` 路径，使转换权重与官方前向数值一致。

### `max_det` 会增加 D-FINE 的 query 数量吗？

不会。与 RT-DETR 相同，`max_det` 只限制返回检测数。解码器仍输出固定数量的 query（默认 300）。若需要更多 query，请在 YAML 中提高 `nq` 并重新训练。
