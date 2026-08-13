---
title: WeDetect：开放词汇检测
comments: true
description: 在 Ultralytics 中使用 WeDetect——ConvNeXt + XLM-RoBERTa 开放词汇目标检测器（训练 / 验证 / 预测 / 导出 / 预训练权重）。
keywords: WeDetect, 开放词汇, XLM-RoBERTa, ConvNeXt, Ultralytics, 目标检测, class_texts, 伪标签, dual 导出
---

# WeDetect：开放词汇检测

> **English docs**  
> English version: [docs/en/models/wedetect.md](../../en/models/wedetect.md)  
> 开放词汇微调逐步教程：[guides/wedetect-ov-finetune.md](../guides/wedetect-ov-finetune.md)

## 概述

[WeDetect](https://github.com/WeChatCV/WeDetect)（论文：[_WeDetect: Fast Open-Vocabulary Object Detection as Retrieval_](https://arxiv.org/abs/2512.12309)）是一种开放词汇[目标检测](https://www.ultralytics.com/glossary/object-detection)器：视觉塔为 **ConvNeXt**，文本塔为 **XLM-RoBERTa**。类别身份是提示词字符串，而不是固定 `class_id`，因此混合 YOLO 子集可以保留各自的本地编号。Ultralytics 以 `WeDetect` 接入标准[训练](../../en/modes/train.md) / [验证](../../en/modes/val.md) / [预测](../../en/modes/predict.md) / [导出](../../en/modes/export.md)流程。

文件名含 `wedetect` 的 `.pt` / `.yaml` 可用 `WeDetect(...)` 或 `YOLO(...)` 加载；后者会自动变成 `WeDetect`。**WeDetect-Uni** 请用 `WeDetectUni(...)`；`YOLO("wedetect-uni-*.pt")` 仍会变成 `WeDetect`（stem 含 `wedetect`）。

### 主要特性

- **文本当 ID：** 用 `class_texts` / `set_classes` 提示词对齐区域特征，不要求各数据集本地 `class_id` 一致。
- **多语文本塔：** Tiny / Base / Uni 配 [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base)；Large / XLarge 配 [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large)。推荐中文同义组。详见[语言塔](#语言塔)。
- **开放词汇微调：** `freeze_text_encoder=False` 时在线编码并更新 LM（`lr0 * text_lr_mult`）；闭集可冻结 LM 或 `close_set=True`。
- **原生 Ultralytics 管线：** `WeDetect` 门面、`WeDetectModel`、YOLO 格式数据、混数多 val、训练前伪标签，以及 dual ONNX/TensorRT 导出。
- **多尺度覆盖：** Tiny / Base / Large（以及 YAML 中的 XLarge、WeDetect-Uni）。

## 模型库

零样本指标来自论文 Table 1（LVIS val / minival 为 fixed AP）。Ultralytics `.pt` 权重托管于 [ModelScope changsu/wedetect-ultralytics](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/files)。

### Zero-shot

| 模型 | 骨干 | 语言塔 | 分辨率 | AP<sup>minival</sup> | COCO AP | 参数量 | FPS | 配置 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [WeDetect-Tiny](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_tiny.pt) | ConvNeXt-T | [XLM-R-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 640×640 | **37.4** | 44.9 | 33M | 62.5 | [yaml](../../../ultralytics/cfg/models/wedetect/wedetect-tiny.yaml) |
| [WeDetect-Base](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_base.pt) | ConvNeXt-B | [XLM-R-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 640×640 | **47.3** | 52.1 | 176M | 35.1 | [yaml](../../../ultralytics/cfg/models/wedetect/wedetect-base.yaml) |
| [WeDetect-Large](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_large.pt) | ConvNeXt-L | [XLM-R-large](https://huggingface.co/FacebookAI/xlm-roberta-large) | 1280×1280 | **55.0** | 54.5 | 490M | 6.0 | [yaml](../../../ultralytics/cfg/models/wedetect/wedetect-large.yaml) |

<details>
<summary><strong>分项指标（LVIS / COCO / ODInW）</strong></summary>

|      模型      | LVIS minival AP / AP<sub>r</sub> / AP<sub>c</sub> / AP<sub>f</sub> | LVIS AP / AP<sub>r</sub> / AP<sub>c</sub> / AP<sub>f</sub> | COCO AP | COCO-O AP | ODInW13 AP | ODInW35 AP |
| :------------: | :----------------------------------------------------------------: | :--------------------------------------------------------: | :-----: | :-------: | :--------: | :--------: |
| WeDetect-Tiny  |                     37.4 / 33.3 / 36.8 / 38.8                      |                 31.4 / 24.7 / 29.2 / 36.8                  |  44.9   |   38.6    |    46.4    |    21.1    |
| WeDetect-Base  |                     47.3 / 43.5 / 45.9 / 49.3                      |                 41.4 / 35.2 / 39.5 / 46.2                  |  52.1   |   44.1    |    53.1    |    24.6    |
| WeDetect-Large |                     55.0 / 51.1 / 54.5 / 56.1                      |                 49.4 / 43.3 / 48.2 / 53.5                  |  54.5   |   47.0    |    53.4    |    25.8    |

</details>

### WeDetect-Uni

推理不走在线 LM，使用可学习 prompt embedding（更快，不能按 `set_classes` 任意换开放词汇提示）。

| 模型 | 语言塔（仅训练初始化） | 配置 |
| :---: | :---: | :---: |
| [WeDetect-Uni-Base](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_base_uni.pt) | [XLM-R-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | [yaml](../../../ultralytics/cfg/models/wedetect/wedetect-uni-base.yaml) |
| [WeDetect-Uni-Large](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_large_uni.pt) | [XLM-R-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | [yaml](../../../ultralytics/cfg/models/wedetect/wedetect-uni-large.yaml) |

**注意：**

- **AP<sup>minival</sup>** 是 LVIS minival 的 fixed AP（论文 Table 1 主指标）。模型名链到对应 Ultralytics `.pt`。
- **FPS** 在 COCO 上测得（`batch_size=1`，见论文表格说明）。上表 WeDetect 数字为黑色，表示该零样本评测**未把 COCO 纳入训练**。
- Large 评测分辨率为 **1280×1280**；Tiny / Base 为 **640×640**。
- 结构 YAML 另有 `wedetect-xlarge.yaml`（语言塔为 [XLM-R-large](https://huggingface.co/FacebookAI/xlm-roberta-large)），当前 [ModelScope 仓库](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/files) 未附对应 `.pt`。

### 语言塔

语言塔把类别提示词编成 768 维向量，再与检测头做对比。YAML 的 `text_model` 决定结构；[ModelScope](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/files) 的 `.pt` 通过顶层 `text_model_weights` 载入 WeDetect 训练过的编码器与投影头。分词器 / `config.json` 仍来自 Hugging Face（或本地镜像），**不是**检测权重的一部分。

|          WeDetect          |  YAML `text_model`  |      语言塔       | 隐层 → 投影 |                       用途                       |                                        链接                                         |
| :------------------------: | :-----------------: | :---------------: | :---------: | :----------------------------------------------: | :---------------------------------------------------------------------------------: |
|        Tiny / Base         | `xlm-roberta:base`  | XLM-RoBERTa-base  |  768 → 768  |                  推理 + OV 微调                  |  [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)  |
|       Large / XLarge       | `xlm-roberta:large` | XLM-RoBERTa-large | 1024 → 768  |                  推理 + OV 微调                  | [FacebookAI/xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) |
|     Uni（base / large）      | `xlm-roberta:base`  | XLM-RoBERTa-base  |  768 → 768  | **仅训练**时初始化可学习 prompt；推理不再加载 LM |  [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)  |

XLM-RoBERTa 在约 100 种语言的 CommonCrawl 上预训练（论文：[Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)；文档：[Transformers · XLM-RoBERTa](https://huggingface.co/docs/transformers/model_doc/xlm-roberta)），因此中文 / 多语 `class_texts` 可直接使用。Large 文本塔隐层为 1024，经线性头投到 768，与 `WeDetectDetect` 的 `embed` 对齐。

加载顺序（`ultralytics/nn/text_model.py`）：

1. **本地目录**（须含 `config.json`）：仓库根 `xlm-roberta-base/` 或 `xlm-roberta-large/`，以及 `checkpoints/` 下同名目录。
2. Hugging Face 缓存（Hub id 为 `xlm-roberta-base` / `xlm-roberta-large`，对应上表 FacebookAI 仓库）。
3. 均失败则联网下载，并 **先下载成功再** `save_pretrained` 到仓库根对应目录。不要事先建空目录。

离线可把 Hugging Face 快照放到上述本地路径（至少 `config.json`、分词文件；有 `pytorch_model.bin` 或 `model.safetensors` 时会用作结构初始化，随后仍由 `.pt` 里的 `text_model_weights` 覆盖）。需安装 `transformers` 与 `sentencepiece`。

## 使用示例

### Python

```python
from ultralytics import WeDetect

# 从 ModelScope 加载 Ultralytics 预训练权重（也可传本地路径或 YAML）
model = WeDetect("https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_base.pt")
# model = WeDetect("wedetect-base.yaml")  # 随机初始化

# 显示模型信息（可选）
model.info()

# 开放词汇微调（必须 cfg=wedetect_finetune.yaml，否则 default.yaml 会冻结文本塔）
results = model.train(
    data="ultralytics/cfg/datasets/wedetect_coco.yaml",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    freeze_text_encoder=False,
    epochs=12,
    imgsz=640,
)

# 验证
metrics = model.val(data="ultralytics/cfg/datasets/wedetect_coco.yaml")

# 预测（可换任意中文/多语提示词）
model.set_classes(["人", "公交车", "车"])
results = model("path/to/bus.jpg")

# 导出 dual ONNX（vision + language，导出后仍可 set_classes）
path = model.export(format="onnx", export_mode="dual")
```

### CLI

```bash
# 下载权重后训练 / 预测（文件名含 "wedetect" 时自动选择 WeDetect）
wget https://www.modelscope.cn/models/changsu/wedetect-ultralytics/resolve/master/wedetect_base.pt
yolo cfg=ultralytics/cfg/wedetect_finetune.yaml detect train model=wedetect_base.pt \
  data=ultralytics/cfg/datasets/wedetect_coco.yaml epochs=12 imgsz=640
yolo predict model=wedetect_base.pt source=path/to/bus.jpg
yolo val model=wedetect_base.pt data=ultralytics/cfg/datasets/wedetect_coco.yaml
yolo export model=wedetect_base.pt format=onnx export_mode=dual
```

> **语言塔**  
> Tiny / Base 需要 [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base)；Large 需要 [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large)。本地目录名为 `xlm-roberta-base/` 或 `xlm-roberta-large/`（须含 `config.json`）。不要留下空目录。详见[语言塔](#语言塔)。

## 支持的任务与模式

| 模型类型       | 配置 / 权重                                 | 支持任务           | 推理 | 验证 | 训练 | 导出 |
| -------------- | ------------------------------------------- | ------------------ | ---- | ---- | ---- | ---- |
| WeDetect-Tiny  | `wedetect-tiny.yaml` / `wedetect_tiny.pt`   | 开放词汇检测       | ✅   | ✅   | ✅   | ✅   |
| WeDetect-Base  | `wedetect-base.yaml` / `wedetect_base.pt`   | 开放词汇检测       | ✅   | ✅   | ✅   | ✅   |
| WeDetect-Large | `wedetect-large.yaml` / `wedetect_large.pt` | 开放词汇检测       | ✅   | ✅   | ✅   | ✅   |
| WeDetect-Uni   | `wedetect-uni-*.yaml` / `wedetect_*_uni.pt` | 可学习 prompt 检测 | ✅   | ✅   | ✅   | ✅   |

架构 YAML 位于 `ultralytics/cfg/models/wedetect/`。OV 微调请使用 `ultralytics/cfg/wedetect_finetune.yaml`（`freeze_text_encoder=False`）。

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

`YOLO("wedetect_*.pt")` 在文件名含 `wedetect` 时会变成 `WeDetect`。推荐 `from ultralytics import WeDetect`。Uni 权重请用 `WeDetectUni`。

### 训练

`WeDetectTrainer` 扩展检测训练循环：混数 `train.yolo_data` / `grounding_data`、同义词全局词表（`mix_global_texts=True`）、可选 NegQueue。`freeze_text_encoder=False` 时 `register_text_model()`，文本参数学习率为 `lr0 * text_lr_mult`（默认 0.01）。`save_model` 把 LM 写入 checkpoint 顶层 **`text_model_weights`**。

单卡第一个 epoch CUDA OOM 时会自动把 `batch` 减半并重建 pipeline（最多 3 次）。`imgsz=1280` 建议一开始就把 `batch` 设到能放下的值。

配置文件：

| 文件                                                    | 用途                                                     |
| ------------------------------------------------------- | -------------------------------------------------------- |
| `ultralytics/cfg/wedetect_finetune.yaml`                | 默认 OV 微调（`lr0=5e-6`，`close_mosaic=1`）             |
| `ultralytics/cfg/wedetect_finetune_mask_refine.yaml`    | OV + mask refine（更接近原版 `2e-5` / `close_mosaic=4`） |
| `ultralytics/cfg/wedetect_scratch.yaml`                 | 从零混数                                                 |
| `ultralytics/cfg/datasets/wedetect_mixed.yaml`          | 混数模板                                                 |
| `ultralytics/cfg/datasets/wedetect_mixed_customer.yaml` | 客户多任务 + LVIS                                        |

### 验证与 fitness

- **单集：** 用 `class_texts` 前 `nc` 行（或 `names`）编码提示词后评测。
- **混数 `val.yolo_data`：** 每个 epoch 切换 `nc` / `names` / `class_texts` 并重建 dataloader。LVIS 自动优先 `minival`。

| 列                           | 含义                                       |
| ---------------------------- | ------------------------------------------ |
| 无前缀 `metrics/mAP50-95(B)` | **第一个** val 集的拷贝（给默认曲线图）    |
| `<数据集>/metrics/...`       | 该子集自己的指标                           |
| 无前缀 `fitness`             | 各集 mAP50-95 **加权平均**，决定 `best.pt` |

`val_fitness_dynamic=true` 时：epoch 1 用 YAML `val_fitness_weights`；之后按上一轮 mAP 调权。含 LVIS 时，LVIS 目标 = `val_fitness_lvis_target_mult ×` 客户子集均值（默认 2.0）。

### 伪标签

在 **train 子集** YAML 写 `pseudo_label*`（优先于 `train()` / CLI）。教师分辨率是 **`pseudo_label_imgsz`（默认 640）**，与训练 `imgsz` 无关。不改原始 `labels/` 与源 `class_texts` JSON。

D-FINE Objects365 教师示例见 [D-FINE](dfine.md)。教师 cache 版本须等于 `DATASET_CACHE_VERSION`（当前 `1.0.4`），否则会重跑推理。

### 导出

动态换提示词请用 **`export_mode=dual`**（默认）：产出 sibling `*_vision.*` + `*_language.*`。Tokenizer 留在 Python。`whole` ONNX 不接官方 `WeDetect(...).predict`。

```python
from ultralytics import WeDetect

WeDetect("wedetect_base.pt").export(format="onnx", export_mode="dual", imgsz=640)
m = WeDetect("wedetect_base_vision.onnx")  # 同目录须有 *_language.onnx
m.set_classes(["车", "人"])
m.predict("path/to/bus.jpg")
```

逐步数据准备、导出参数与排错见 [开放词汇微调教程](../guides/wedetect-ov-finetune.md)。

## 架构说明

- **骨干：** ConvNeXt（tiny / base / large / xlarge）。
- **Neck：** CSPRepBiFPAN。
- **检测头：** `WeDetectDetect`（区域–文本对比，embed=768）。
- **语言塔：** Tiny / Base / Uni 为 [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base)（768→768）；Large / XLarge 为 [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large)（1024→768）。模型类 `WeDetectModel` 程序组装，不用 YOLO 式 `parse_model` 层表。
- **训练 / 验证 / 预测：** `WeDetectTrainer`、`WeDetectValidator`、`WeDetectPredictor`。

## 引用与致谢

若在研究或产品中使用 WeDetect，请引用[原论文](https://arxiv.org/abs/2512.12309)：

**BibTeX**

```bibtex
@article{fu2025wedetect,
      title={WeDetect: Fast Open-Vocabulary Object Detection as Retrieval},
      author={Fu, Shenghao and Su, Yukun and Rao, Fengyun and LYU, Jing and Xie, Xiaohua and Zheng, Wei-Shi},
      journal={arXiv preprint arXiv:2512.12309},
      year={2025}
}
```

感谢 [WeDetect 作者](https://github.com/WeChatCV/WeDetect) 开源实现与预训练权重。

## 常见问题

### WeDetect 与 YOLO-World / YOLOE 有何不同？

三者都是开放词汇检测。WeDetect 使用 ConvNeXt + XLM-RoBERTa（多语提示，中文友好），类别对齐走 `class_texts` 同义组；导出默认 dual 双塔以便部署后换提示词。YOLO-World / YOLOE 基于 YOLO 骨干与 CLIP/视觉提示路径，见对应模型页。

### `.pt` 权重从哪里获取？

检测权重从 [ModelScope changsu/wedetect-ultralytics](https://www.modelscope.cn/models/changsu/wedetect-ultralytics/files) 下载，或见上方[模型库](#模型库)表格。语言塔分词器 / 结构见 [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base) 与 [XLM-RoBERTa-large](https://huggingface.co/FacebookAI/xlm-roberta-large)（配对关系见[语言塔](#语言塔)）。`.pt` 为 Ultralytics 格式，内含微调后的 `text_model_weights`。

### 能否在自定义 YOLO 数据集上训练？

可以。标签为标准 YOLO 检测 txt，并提供 `class_texts` JSON（推荐中文同义组）。务必 `cfg=wedetect_finetune.yaml` 且 `freeze_text_encoder=False`。混数见 `wedetect_mixed.yaml`。完整步骤见 [OV 微调教程](../guides/wedetect-ov-finetune.md)。

### 无前缀 `metrics/mAP50-95(B)` 为什么和 `lvis/metrics/...` 一样？

混数把 **第一个** val 集再写一份无前缀列，给默认 CSV/曲线用。第一项是 LVIS 时两列相同。选 `best.pt` 看无前缀 **`fitness`**（加权平均）。

### `YOLO("wedetect-uni-base.pt")` 会加载 Uni 吗？

不会。stem 含 `wedetect` 时 `YOLO()` 会变成 `WeDetect`。请使用 `from ultralytics import WeDetectUni`。

### 导出后换不了类名？

确认 `export_mode=dual`（或 `whole` 走示例脚本），而不是把文本 fuse 进视觉头的固定词表路径。官方推理传入 `*_vision.onnx|.engine`，同目录须有 sibling `*_language.*`。
