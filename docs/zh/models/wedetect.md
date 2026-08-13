---
title: WeDetect 开放词汇检测
comments: true
description: WeDetect（ConvNeXt + XLM-RoBERTa）在 Ultralytics 中的架构、训练 / 验证 / 预测 / 导出全流程，以及混数、伪标签与 dual 导出。
keywords: WeDetect, 开放词汇, XLM-RoBERTa, ConvNeXt, Ultralytics, 目标检测, class_texts, 伪标签, D-FINE
---

# WeDetect：开放词汇检测

> **English docs**  
> 模型页：[docs/en/models/wedetect.md](../../en/models/wedetect.md)  
> 开放词汇微调逐步教程（本仓库最完整中文流程）：[guides/wedetect-ov-finetune.md](../guides/wedetect-ov-finetune.md)

[WeDetect](https://github.com/WeDetect/WeDetectPT) 是开放词汇检测器：视觉塔为 **ConvNeXt**，文本塔为 **XLM-RoBERTa**。类别身份是提示词字符串，而不是固定 `class_id`，因此混合 YOLO 子集可以保留各自的本地编号。Ultralytics 以 `WeDetect` 接入标准[训练](../../en/modes/train.md) / [验证](../../en/modes/val.md) / [预测](../../en/modes/predict.md) / [导出](../../en/modes/export.md)。

文件名含 `wedetect` 的 `.pt` / `.yaml` 可用 `WeDetect(...)` 或 `YOLO(...)` 加载；后者会自动变成 `WeDetect`。**WeDetect-Uni** 请用 `WeDetectUni(...)`；`YOLO("wedetect-uni-*.pt")` 仍会变成 `WeDetect`（stem 含 `wedetect`）。

## 架构

| 部件 | 实现 |
| ---- | ---- |
| 视觉骨干 | ConvNeXt（`tiny` / `base` / `large` / `xlarge`） |
| Neck | CSPRepBiFPAN |
| 检测头 | `WeDetectDetect`（区域–文本对比，embed=768） |
| 文本塔 | XLM-RoBERTa base（`xlm-roberta:base`） |
| 模型类 | `WeDetectModel`（程序组装，不用 YOLO 式 `parse_model` 层表） |
| 训练 / 验证 / 预测 | `WeDetectTrainer`、`WeDetectValidator`、`WeDetectPredictor` |

结构 YAML 在 `ultralytics/cfg/models/wedetect/`。无在线 LM 的 **WeDetect-Uni** 见 `wedetect-uni-*.yaml`。

两种微调：

- **开放词汇（OV）：** `freeze_text_encoder=False`，使用 `ultralytics/cfg/wedetect_finetune.yaml`。LM 在线更新，学习率为 `lr0 * text_lr_mult`。
- **闭集：** `freeze_text_encoder=True`（`default.yaml`）或 `close_set=True`。缓存 embeddings，不更新 LM。

## 全流程

```text
YOLO 标注 + class_texts JSON
        ↓
可选教师伪标签（SAM3 / YOLO / WeDetect / D-FINE）
        ↓
WeDetect.train(cfg=wedetect_finetune.yaml)
        ↓
best.pt（视觉 + text_model_weights）
        ↓
export_mode=dual → *_vision.* + *_language.*
        ↓
set_classes([...]).predict(...)
```

最小训练示例：

```python
from ultralytics import WeDetect

model = WeDetect("pretrained_weights/wedetect_base.pt")
model.train(
    data="ultralytics/cfg/datasets/wedetect_coco.yaml",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    freeze_text_encoder=False,
    device=0,
)
```

### 验证与 fitness

混数 `val.yolo_data` 每个 epoch 按子集切换 `nc` / `names` / `class_texts` 并重建 dataloader。

| 列 | 含义 |
| -- | ---- |
| 无前缀 `metrics/mAP50-95(B)` | **第一个** val 集的拷贝（给默认曲线图） |
| `<数据集>/metrics/...` | 该子集自己的指标 |
| 无前缀 `fitness` | 各集 mAP50-95 **加权平均**，决定 `best.pt` |

`val_fitness_dynamic=true` 时：epoch 1 用 YAML `val_fitness_weights`；之后按上一轮 mAP 调权。含 LVIS 时，LVIS 目标 = `val_fitness_lvis_target_mult ×` 客户子集均值（默认 2.0）。

### 伪标签要点

- 教师分辨率是 **`pseudo_label_imgsz`（默认 640）**，与训练 `imgsz` 无关。
- 增量 flush 默认每 **200** 张；预取 `pseudo_label_prefetch=2`。
- Cache 版本与数据集统一为 `DATASET_CACHE_VERSION`（当前 `1.0.4`）。版本不一致时 `apply_pseudo_labels` 会 **重跑教师**；`get_labels` 在 hash 匹配时对 merged cache 更宽松。
- D-FINE Objects365 教师：`nc=366`、下标 0 为 background，按英文名对齐；预处理为 `LetterBox(scale_fill=True)`。详见 [D-FINE](dfine.md)。

逐步数据准备、导出 dual ONNX/TensorRT、排错见 [开放词汇微调教程](../guides/wedetect-ov-finetune.md)。

## 配置文件

| 文件 | 用途 |
| ---- | ---- |
| `ultralytics/cfg/wedetect_finetune.yaml` | OV 微调（必须用它，避免落到 `default.yaml` 的冻结文本塔） |
| `ultralytics/cfg/wedetect_finetune_mask_refine.yaml` | OV + mask refine |
| `ultralytics/cfg/wedetect_scratch.yaml` | 从零混数 |
| `ultralytics/cfg/datasets/wedetect_mixed.yaml` | 混数模板 |
| `ultralytics/cfg/datasets/wedetect_mixed_customer.yaml` | 客户多任务 + LVIS 示例 |

## 支持的任务与模式

| 模型 | 配置 / 权重 | 任务 | 训练 | 验证 | 预测 | 导出 |
| ---- | ----------- | ---- | ---- | ---- | ---- | ---- |
| WeDetect | `wedetect-*.yaml` / `wedetect_*.pt` | 开放词汇检测 | ✅ | ✅ | ✅ | ✅ dual/whole |
| WeDetect-Uni | `wedetect-uni-*.yaml` | 可学习 prompt 检测 | ✅ | ✅ | ✅ | ✅ |
