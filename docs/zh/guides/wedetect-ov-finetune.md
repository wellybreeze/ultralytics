# WeDetect 开放词汇微调全流程教程（Ultralytics）

本文面向当前仓库中的 **Ultralytics WeDetect**，说明从数据准备、配置修改、Python 开放词汇（OV）微调、ONNX 导出到测试验证的完整流程。行为对齐原版 WeDetect 的 full_tuning（保留并更新 XLM-RoBERTa 文本塔）。

> 工作目录默认：`ultralytics/`（即本仓库中含 `pretrained_weights/`、`ultralytics/` 包的那一层）。

---

## 1. 流程总览

```text
准备 YOLO 格式数据 + 中文 class_texts（可选 grounding JSON）
        ↓
单集 yaml 或 wedetect_mixed.yaml；cfg=wedetect_finetune.yaml
        ↓
Python: WeDetect(...).train(...)   # freeze_text_encoder=False
        ↓
得到 runs/.../weights/best.pt（含 text_model_weights）
        ↓
model.export(format="onnx"|"engine", export_mode="dual")
        ↓
PyTorch / dual ONNX|engine：set_classes + predict
```

| 阶段                   | 关键产物                                                   |
| ---------------------- | ---------------------------------------------------------- |
| 微调                   | `best.pt` / `last.pt`（视觉 + `text_model_weights`） |
| 导出 dual ONNX（推荐） | `*_vision.onnx` + `*_language.onnx`                    |
| 导出 dual TensorRT     | `*_vision.engine` + `*_language.engine`                |
| 导出 whole             | `*_whole.onnx`（官方 predict/engine 不接，走示例脚本）   |
| 推理                   | 任意中文/多语提示词（分词在 Python，不进图）               |

---

## 2. 环境与预训练权重

### 2.1 依赖

```bash
cd ultralytics
pip install -e .
# 文本塔（XLM-R）+ 分词后端 + ONNX
pip install transformers sentencepiece onnx onnxruntime
```

> XLM-RoBERTa 需要 `sentencepiece`（或可用的 fast `tokenizer.json`）。缺依赖时常见误报为 tokenizer 初始化失败。

### 2.2 预训练权重

将官方权重放到仓库根目录 `pretrained_weights/`，例如：

| 文件                                     | 说明                          |
| ---------------------------------------- | ----------------------------- |
| `pretrained_weights/wedetect_tiny.pt`  | 轻量，适合冒烟                |
| `pretrained_weights/wedetect_base.pt`  | 常用微调起点（对齐原版 Base） |
| `pretrained_weights/wedetect_large.pt` | 更大模型                      |

文本编码器结构为 **XLM-RoBERTa base**（`xlm-roberta:base`）。检测权重里的 `text_model_weights` 会载入 LM；分词器/结构解析顺序（见 `ultralytics/nn/text_model.py`）：

1. **本地目录**（须含 `config.json`）：仓库根下 `xlm-roberta-base/` 或 `checkpoints/xlm-roberta-base/`
2. HuggingFace 缓存：`~/.cache/huggingface/hub/models--xlm-roberta-base/...`
3. 均失败则联网下载，并 **先下载成功再** `save_pretrained` 到仓库根 `xlm-roberta-base/`（勿事先建空目录）

> **注意：** 不要在仓库根留下空的 `xlm-roberta-base/`。HuggingFace 会把该相对名当成本地模型路径，导致 tokenizer 报错（信息常误导为缺 `sentencepiece`）。
> 原版 `WeDetect/xlm-roberta-base` **不会**被自动发现；离线请复制到上述本地候选路径，并确保含 `config.json` 与分词文件。

---

## 3. 数据格式如何准备

Ultralytics WeDetect 使用 **YOLO Detect 标注**（可加 segment 多边形做 mask refine），并通过 **类别文本**（推荐中文）做开放词汇对齐。

### 3.1 目录结构（推荐）

```text
datasets/my_dataset/
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── val/
│       ├── 1001.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── 0001.txt
│   │   └── ...
│   └── val/
│       ├── 1001.txt
│       └── ...
└── texts/
    └── class_texts_zh.json
```

### 3.2 标签文件（YOLO）

每个图像对应同名 `.txt`，每行一个目标：

```text
# 仅框（开放词汇微调默认）
<class_id> <x_center> <y_center> <width> <height>

# 带分割多边形（mask_refine=True 时需要）
<class_id> <x1> <y1> <x2> <y2> ...
```

坐标均为相对宽高的归一化值（0–1）。`class_id` 从 `0` 开始，与 `names` / `class_texts` 顺序一致。

**从 COCO JSON 转换**（Ultralytics 内置转换思路）：

```bash
# 参考官方文档：COCO → YOLO
# https://docs.ultralytics.com/guides/coco-to-yolo/
yolo convert source=path/to/instances_train2017.json format=yolo
```

也可用自定义脚本把 COCO `bbox` 写成上述 YOLO txt；若使用 mask refine，需保留 segmentation 多边形。

### 3.3 类别文本 JSON（强烈推荐中文）

格式为 **`list[list[str]]`**，与 `names` 对齐：

- **长度 ≥ `nc`**：前 `nc` 行对应已标注类 `0 .. nc-1`；每行至少一个同义提示
- **长度 > `nc`**：多出的行**只作训练负类文本**（`RandomLoadText` 采样），**不参与 val 的 nc 类指标**（val 只用前 `nc` 行）

```json
[
  ["车", "车辆", "汽车"],
  ["人", "行人"],
  ["公交车"]
]
```

仓库内模板（勿依赖外部原版路径）：

```bash
mkdir -p datasets/my_dataset/texts
cp ultralytics/cfg/datasets/texts/coco_zh_class_texts.json datasets/my_dataset/texts/
```

> 若不提供 `class_texts`，训练回退 data YAML 的 `names`（多为英文）。建议始终提供中文 JSON。

#### 同义组语义（重要）

训练时对**正类**会从该行同义组中**随机抽一个词**作为正样本文本，图上该 `class_id` 的**全部框**都当作匹配该词：

| 写法                                                | 是否正确                                          |
| --------------------------------------------------- | ------------------------------------------------- |
| `["车", "车辆", "汽车", "轿车"]`（近义/上位近义） | ✅ 可扩词面泛化                                   |
| `["车", "红色车", "白色车"]`（属性写进同义组）    | ❌ 抽到「红色车」时白车框也会当正样本，框文不对齐 |

属性/指代表达（颜色、是否戴帽等）需要：**细分类标注**、**grounding 实例级短语**（§3.6），或自定义实例级加载；不能塞进同一 YOLO 同义行。

### 3.4 数据配置 YAML

新建例如 `ultralytics/cfg/datasets/my_dataset.yaml`（或放在任意路径）：

```yaml
# my_dataset.yaml
path: /absolute/path/to/datasets/my_dataset  # 或相对 datasets 根目录的路径
train: images/train
val: images/val

# 开放词汇：优先级高于 names
class_texts: texts/class_texts_zh.json

names:
  0: person
  1: bus
  2: tie
```

官方 COCO 示例：`ultralytics/cfg/datasets/wedetect_coco.yaml`。按需修改 `path` / `train` / `val`。默认已启用：

```yaml
class_texts: texts/coco_zh_class_texts.json
```

请先把中文 JSON 放到数据集根目录对应相对路径（模板已打包）：

```bash
mkdir -p datasets/coco/texts
cp ultralytics/cfg/datasets/texts/coco_zh_class_texts.json datasets/coco/texts/
# 也可在 data YAML 里写该模板的绝对路径
```

> 训练时务必 `cfg=ultralytics/cfg/wedetect_finetune.yaml`（或 `wedetect_finetune_mask_refine.yaml`），避免落到 `default.yaml` 的 `freeze_text_encoder=True`。

### 3.5 Mask Refine 额外要求

若使用 `wedetect_finetune_mask_refine.yaml`（`mask_refine=True`）：

- 标签需含 **segmentation 多边形**（YOLO segment 行格式）
- 训练时会用 mask 修正 bbox（对齐原版 `mask2bbox`）
- 仅有框、无多边形时 refine 不会生效

### 3.6 Grounding / 指代表达（可选混数）

混数 YAML 可增加 `train.grounding_data`（见 `wedetect_mixed.yaml`）。加载器为 `GroundingDataset`，读的是 **COCO 风格整份 JSON**（**不是**按行 JSONL）：

```yaml
train:
  grounding_data:
    - img_path: /path/to/images
      json_file: /path/to/annotations.json
```

JSON 必备字段：

| 位置              | 字段                                                      | 说明                                                                                             |
| ----------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `images[]`      | `id`, `file_name`, `width`, `height`, `caption` | 每图一句 caption                                                                                 |
| `annotations[]` | `image_id`, `bbox`, `tokens_positive`, `iscrowd`  | `bbox` 为 COCO `[x,y,w,h]` 像素；`tokens_positive` 为 caption 上的字符区间 `[start,end)` |

每条 annotation → **一个短语**（多段 `tokens_positive` 会拼成一句）+ **一个框**。
同一几何框要挂多组文本：写**多条** annotation（同 `bbox`、不同 `tokens_positive`）。
去重键为 `[cls, x, y, w, h]`（短语不同则 `cls` 不同，可共存）。

这与 YOLO `class_texts` 同义组不同：grounding **没有**「一行多同义词随机抽」；同义泛化需对同一框写多条短语，或仍用 YOLO 子集 + `class_texts`。

---

## 4. 需要关心的配置文件

### 4.1 训练超参（OV 微调）

| 文件                                                   | 用途                                                                            |
| ------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `ultralytics/cfg/wedetect_finetune.yaml`             | **默认开放词汇微调**（框标注；以该文件为准）                              |
| `ultralytics/cfg/wedetect_finetune_mask_refine.yaml` | OV + mask refine（部分超参仍贴近原版 2e-5 /`close_mosaic=4`）                 |
| `ultralytics/cfg/default.yaml`                       | 全局默认；含`freeze_text_encoder` / `export_mode` / `mix_global_texts` 等 |

`wedetect_finetune.yaml` 当前关键默认值（与代码一致；`train()` / CLI 可覆盖）：

| 键                      | 默认值                           | 含义                                                      |
| ----------------------- | -------------------------------- | --------------------------------------------------------- |
| `freeze_text_encoder` | `False`                        | 在线编码并更新文本塔                                      |
| `text_lr_mult`        | `0.01`                         | 文本塔学习率 =`lr0 * text_lr_mult`                      |
| `close_set`           | `False`                        | 闭集：缓存 embeddings，丢弃在线 LM                        |
| `mask_refine`         | `False`                        | 是否 mask 修正框                                          |
| `mix_global_texts`    | `True`（在 `default.yaml`）  | 混数时按同义合并全局词表                                  |
| `use_neg_queue`       | `False`（在 `default.yaml`） | 跨子集动态负类文本队列                                    |
| `optimizer`           | `AdamW`                        |                                                           |
| `lr0`                 | `5e-6`                         | 本 cfg 默认；原版 full_tuning 多为`2e-5`                |
| `weight_decay`        | `0.05`                         |                                                           |
| `epochs`              | `12`                           |                                                           |
| `batch`               | `4`                            | 单卡；多卡时每卡 batch                                    |
| `dfl`                 | `0.375`                        |                                                           |
| `mixup`               | `0.15`                         |                                                           |
| `close_mosaic`        | `1`                            | 最后 N epoch 关 mosaic（本 cfg；mask_refine cfg 为`4`） |
| `warmup_iters`        | `1000`                         | 按**iteration** warmup（覆盖 `warmup_epochs`）    |
| `warmup_start_factor` | `0.001`                        | 从`0.001×lr` 升到当前目标 lr                           |
| `export_mode`         | `dual`（`default.yaml`）     | ONNX：`dual` / `whole`                                |

> **必须**通过 `cfg=wedetect_finetune.yaml`（或 mask_refine 变体）启动训练。若只用 `default.yaml`，默认 `freeze_text_encoder=True`，文本塔不会参与 OV 微调。

一般 **不必改模型结构 YAML**；用预训练 `.pt` 即可。结构文件在：

- `ultralytics/cfg/models/wedetect/wedetect-base.yaml`
- `wedetect-tiny.yaml` / `wedetect-large.yaml` 等

### 4.2 导出相关

`ultralytics/cfg/default.yaml`：

```yaml
export_mode: dual   # dual | whole；WeDetect ONNX 专用
```

---

## 5. Python 脚本：开放词汇微调

### 5.1 最小可运行示例

在 `ultralytics/` 下新建 `train_wedetect_ov.py`：

```python
from ultralytics import WeDetect

# 1) 加载预训练（会带上 text_model_weights → _text_sd）
model = WeDetect("pretrained_weights/wedetect_base.pt")

# 2) 开放词汇微调：文本塔参与训练
model.train(
    data="ultralytics/cfg/datasets/wedetect_coco.yaml",  # 或 customer/*.yaml / wedetect_mixed.yaml
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    # 下列参数会覆盖 cfg；也可全部写在 yaml 里
    freeze_text_encoder=False,
    text_lr_mult=0.01,
    # lr0 / close_mosaic 等以 wedetect_finetune.yaml 为准（当前默认 5e-6 / 1）
    epochs=12,
    batch=4,
    imgsz=640,
    device=0,  # 多卡: device="0,1,2,3"
    project="runs/wedetect",
    name="ov_finetune_base",
    exist_ok=True,
)
```

### 5.2 Mask Refine

```python
from ultralytics import WeDetect

model = WeDetect("pretrained_weights/wedetect_base.pt")
model.train(
    data="path/to/my_dataset.yaml",  # labels 需含多边形
    cfg="ultralytics/cfg/wedetect_finetune_mask_refine.yaml",
    freeze_text_encoder=False,
    mask_refine=True,
    device=0,
    project="runs/wedetect",
    name="ov_finetune_mask_refine",
)
```

### 5.3 多卡

```python
model.train(
    data="...",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    device="0,1,2,3",
    batch=4,  # 每卡 batch；总有效 batch ≈ 4 * GPU 数
)
```

或 CLI（等价）：

```bash
yolo cfg=ultralytics/cfg/wedetect_finetune.yaml \
  model=pretrained_weights/wedetect_base.pt \
  data=ultralytics/cfg/datasets/wedetect_coco.yaml \
  device=0,1,2,3 batch=4
```

### 5.4 训练时发生了什么（便于排错）

1. `WeDetectTrainer.setup_model`：在 `freeze_text_encoder=False` 时 `register_text_model()`，文本塔进入 `model.parameters()`。
2. 每个 batch：**在线** `encode_texts`（不走冻结缓存），文本塔带梯度，`lr *= text_lr_mult`。
3. `save_model` / `strip_optimizer`：调用 `sync_text_model_weights()`，把 LM 写回 `_text_sd`，并在 `.pt` 顶层写入 **`text_model_weights`**，保证导出/再加载一致。

训练结束后权重通常在：

```text
runs/wedetect/ov_finetune_base/weights/best.pt
runs/wedetect/ov_finetune_base/weights/last.pt
```

快速检查 LM 是否写入：

```python
import torch
ckpt = torch.load("runs/wedetect/ov_finetune_base/weights/best.pt", map_location="cpu", weights_only=False)
print("text_model_weights" in ckpt, len(ckpt.get("text_model_weights") or {}))
```

### 5.5 训练前伪标签（可选）

在构建 train dataloader **之前**（`WeDetectTrainer.get_dataset`），可用教师模型对 **train** 集推理，把伪框与 GT 合并，并扩展词表（缓解单域微调导致的开放词汇崩塌）。

实现要点（与代码一致）：

- **不修改**原始 `labels/` 与既有 `labels.cache`
- **不写**每图伪标/合并 `.txt`；只写整库 Ultralytics `*.cache`
- 训练侧 `data["labels_dir"]="labels_pseudo_merged"`，`nc` / `names` 扩为 GT + 接纳的伪标类
- 写出旁路 `*_train.json` 的中文伪标段来自 `pseudo_label_class_texts`（`kept_zh`），不是英文 `names`

#### 产物文件

| 文件                           | 位置                                                                                    | 说明                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `pseudo_labels-{model}.cache` | 数据集`path` 根目录                                                                   | 仅教师伪标（如 `sam3.pt` → `pseudo_labels-sam3.cache`）；`cls` 已 remap 为 `nc_gt+k`；归一化 xywh；含 `version`/`hash`/`labels` |
| `labels_pseudo_merged.cache` | 与 YOLO 约定一致：`…/labels_pseudo_merged.cache`（`labels_pseudo_merged/` 目录旁） | GT+伪标合并；格式同`YOLODataset.cache_labels`（`DATASET_CACHE_VERSION=1.0.3`）           |
| `pseudo_label_meta.json`     | 数据集`path` 根目录                                                                   | 幂等元信息：词表 hash、`kept_src_ids`、`conf`、两份 cache 路径/hash、`nc_gt`/`nc` 等 |
| `<stem>_train.json`          | 与 yaml`class_texts` 同目录                                                           | 合并词表（GT + 伪标中文 + leftover 负类）；**不覆盖**原 JSON；训练/微调优先用此文件    |

`get_labels`：伪标合并 cache 用**相对路径可移植 hash**（`merged_cache_hash` / `path_mode=rel_v1`）校验；命中后把 `labels[].im_file` **remap** 到本机当前绝对路径。若合并 cache 失效但 `pseudo_labels-{model}.cache` 仍在（或 meta 中记录的路径仍可读），则调用 `rebuild_merged_pseudo_cache` 从伪标 cache + GT **重合并**（不跑教师、不扫空 txt）；两者皆无则报错提示重新开启数据集侧 `pseudo_label`。

**增量写入 / 断点续跑：** 每处理 `PSEUDO_FLUSH_EVERY`（默认 50）张图后原子写入 `pseudo_labels-{model_stem}.cache`；结束时强制 flush。YOLO/WeDetect 将路径写入临时 `.txt` 再 `predict`（**禁止**把路径 list 直接传入，否则 Ultralytics 会 `LoadPilAndNumpy` 把整表当成一个超大 batch）。`rect=False` 固定 640 方图 letterbox，显存更稳。

**跨机复用：** 只要数据集相对布局不变（相对 `path` 的 `images/`、`labels/` 等），绝对挂载点不同也可直接拷贝 `pseudo_labels-*.cache` / `labels_pseudo_merged.cache` / `pseudo_label_meta.json`。hash 基于相对路径键 + 文件 size，模型/词表配置用 basename；加载时 remap `im_file`。旧版绝对路径 hash 的 cache 会 miss 一次并自动按新逻辑重建。

#### 配置项（优先写在数据集 YAML）

解析优先级（高→低）：**该数据集 YAML 的 `pseudo_label*`** → 混数顶层 YAML 同名键（仅填补子集未写的项）→ `wedetect_finetune.yaml` / CLI train args → 代码默认。

因此混数时可只给车辆子集开伪标、LVIS 不开；不必再在 `model.train(pseudo_label=True)` 里全局打开。

| 配置项                        | 默认                              | 说明                                                                                    |
| ----------------------------- | --------------------------------- | --------------------------------------------------------------------------------------- |
| `pseudo_label`              | `False`                         | 是否对本数据集 train 启用                                                               |
| `pseudo_label_model`        | `sam3.pt`                       | 教师：`sam3.pt` / YOLO `.pt` / WeDetect `.pt`                                     |
| `pseudo_label_classes`      | 空 →`coco.yaml` 的 `names`   | 教师英文/源词表（与下项同序）                                                           |
| `pseudo_label_class_texts`  | 空 →`coco_zh_class_texts.json` | 同序中文伪标词表；**写入合并 `class_texts` 的是中文**                           |
| `pseudo_label_conf`         | `0.25`                          | 教师置信度阈值                                                                          |
| `pseudo_label_batch`        | `0`（自动）                     | `≤0`：按空闲显存自动；`>0`：固定。YOLO/WeDetect=图像 batch；SAM3=文本 prompt chunk |
| `pseudo_label_mem_fraction` | `0.85`                          | 自动 batch 时占用空闲显存的目标比例，钳制到`[0.1, 0.95]`                              |
| `imgsz`                     | 训练`imgsz`（数据集可覆盖）     | 教师推理分辨率                                                                          |

示例（写在 [`wedetect_vehicle.yaml`](../../../ultralytics/cfg/datasets/customer/wedetect_vehicle.yaml)）：

```yaml
names:
  0: 车
class_texts: .../wedetect_vehicle_txt.json

pseudo_label: true
pseudo_label_model: sam3.pt
pseudo_label_conf: 0.25
pseudo_label_batch: 0
pseudo_label_mem_fraction: 0.85
```

教师 batch 策略：

- **YOLO / WeDetect**：warmup `batch=1` 测激活显存 → 按 `free×mem_fraction` 估计 → 在 `1,2,4,…` 上几何探测，OOM 回退；全量 `predict(..., batch=bsz, stream=True)`
- **SAM3**：逐图 `set_image`（API 限制 `batch=1`）；按空闲显存选择 prompt chunk（约 16–128），OOM 时 chunk 减半重试

#### 固定流水线

```text
冲突处理 + 词表排序
  → 写出 <stem>_train.json（原 class_texts 不动）
  → 教师推理（动态 batch / prompt chunk）
  → 写 pseudo_labels-{model_stem}.cache
  → 加载 GT（优先 labels.cache，否则扫 labels/*.txt）
  → 合并写 labels_pseudo_merged.cache
  → 更新 data: labels_dir / nc / names / class_texts→*_train.json
```

词表排序与 id：

1. **标注类 id 不变**：`0 .. nc_gt-1`（`names` / 原 `class_texts` 前缀）
2. **同义冲突丢弃**：伪标类与 GT 全量同义词相交 → 整类不推理、不占新 id
3. **无冲突伪标后移**：按伪标词表原相对顺序追加，新 id = `nc_gt + k`；`new_nc = nc_gt + K`
4. **原负类行**：超出 `nc` 的 `class_texts` 行接到伪标段之后（与已接纳伪标无同义重叠），仅作 OV 负样本，不增加 `nc`
5. **写出**：`new_texts = GT前缀 + kept_zh + leftover负类` → 旁路 `<stem>_train.json`（如 `wedetect_vehicle_txt_train.json`）；无 `class_texts` 时新建 `path/class_texts_train.json`。原 JSON **不修改**

框合并：伪框 `cls` 已是 `nc_gt+k` 时与 GT 直接拼接；**仅类别级去重，无 IoU 空间抑制**。混数时每个 train 子集各自写出自己的 `*_train.json` 与两份 cache。

val：仍 `labels_dir=labels`，指标按原 `names/nc`。若旁路 `*_train.json` 已存在，加载数据集时会**自动优先**用它作为 `class_texts`（yaml 可仍写原路径）。

幂等：`pseudo_label_meta.json` 的 hash + 两份 cache 的 version/hash + 磁盘 `*_train.json` 内容一致 → 跳过推理与合并。

```python
from ultralytics import WeDetect

# 伪标开关与教师参数写在数据集 YAML（如 wedetect_vehicle.yaml）即可
model = WeDetect("pretrained_weights/wedetect_base.pt")
model.train(
    data="ultralytics/cfg/datasets/customer/wedetect_vehicle.yaml",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    freeze_text_encoder=False,
    # 可选：仅当数据集 YAML 未写对应键时，以下 train args 才生效
    # pseudo_label=True,
    # pseudo_label_model="yolo11x.pt",
    epochs=12,
    device=0,
)
```

---

## 6. 验证（Val）

训练过程中若 `val=True` 会自动验证。

- **单集**：按该 yaml 的 `nc` / `names` / `class_texts`（仅前 `nc` 行）编码提示词后评测。
- **混数 `val.yolo_data`**：每个 epoch 与 `final_eval` 会**逐集**切换 `nc`/`names`/`class_texts` 并重建 dataloader；`best.pt` 的 fitness 为各集加权平均（默认平分，可用 `val_fitness_weights`）。LVIS 自动优先 `minival`。
- 独立 `model.val(...)` / `final_eval` 走 `WeDetectValidator` 时会按当前 `args.data` 清空旧 dataloader，避免「指标按 A 集、标签却仍是 B 集」的错位（例如混淆矩阵 `IndexError: index 1201 is out of bounds`）。

单独评测：

```python
from ultralytics import WeDetect

model = WeDetect("runs/wedetect/ov_finetune_base/weights/best.pt")
metrics = model.val(
    data="ultralytics/cfg/datasets/wedetect_coco.yaml",
    split="val",
    imgsz=640,
    batch=8,
    device=0,
)
print(metrics.box.map)   # mAP50-95
print(metrics.box.map50) # mAP50
```

CLI：

```bash
yolo val model=runs/wedetect/ov_finetune_base/weights/best.pt \
  data=ultralytics/cfg/datasets/wedetect_coco.yaml
```

混数请传入具体子集 yaml（或混数 yaml；独立 val 时若仍是 dict，会取 `val.yolo_data[0]`）。

---

## 7. PyTorch 推理测试（自定义提示词）

微调后的 `.pt` 仍可通过 `set_classes` 换任意提示词（中文推荐）：

```python
from ultralytics import WeDetect

model = WeDetect("runs/wedetect/ov_finetune_base/weights/best.pt")
model.set_classes(["人", "公交车", "领带"])  # 可任意改

results = model.predict(
    source="ultralytics/assets/",
    save=True,
    conf=0.25,
    device=0,
)
```

等价于仓库里的 `predict.py` 写法。
`set_classes` 会加载 XLM-R + `text_model_weights`，编码类名得到 `txt_feats`，再跑视觉检测头。

---

## 8. 导出 ONNX / TensorRT（动态开放词汇）

导出后仍要 `set_classes` 换提示词时，**必须用 dual**（语言塔 + 视觉塔）。吞吐大致：

`engine + nms=True` ≫ `torchscript + nms=True` ≫ `onnx + nms=True` ≫ 无 NMS / Python 后处理。

- **最快且开放词汇**：`format="engine", export_mode="dual", nms=True`（需 GPU / TensorRT）
- **无 TRT 时的次优**：`format="torchscript", export_mode="dual", nms=True`（vision 图内 `batched_nms` → `(B,max_det,6)`）
- **跨平台便携**：`format="onnx", export_mode="dual"`（可选 `nms=True` 用 ORT 原生 NMS）

### 8.1 Dual ONNX（便携）

语言塔只算一次，视觉塔复用 `txt_feats`；`num_classes` / `seq_len` 为动态轴，导出后可换任意提示词。

```python
from ultralytics import WeDetect

model = WeDetect("runs/wedetect/ov_finetune_base/weights/best.pt")
paths = model.export(
    format="onnx",
    export_mode="dual",   # 默认即为 dual
    imgsz=640,
    opset=17,
    simplify=False,       # 可选 True
    device="cpu",         # 或 "0"
)
# 产出（与 .pt 同目录）：
#   *_vision.onnx
#   *_language.onnx
print(paths)
```

**可选原生 NMS（ONNX Runtime 可跑）**：`nms=True` 时在 vision（或 whole）图上追加 ONNX 算子 `NonMaxSuppression`（**不是** TensorRT 的 `EfficientNMS_TRT`），输出 `bboxes` / `scores` / `nms_indices`，可用 ORT 直接推理；官方 `WeDetect("..._vision.onnx").set_classes(...).predict(...)` 同样支持。

```python
model.export(
    format="onnx",
    export_mode="dual",
    nms=True,
    max_det=300,
    conf=0.25,
    iou=0.45,
    max_classes=80,
    imgsz=640,
)
```

### 8.1.1 Dual TorchScript（开放词汇 + 可选图内 NMS）

双塔 TorchScript：语言塔编码任意提示词，视觉塔消费 `txt_feats`。`nms=True` 时在 vision 内用 `batched_nms` 直接输出 `(B, max_det, 6)`（比回传 indices 再 Python gather 更快）。**绝对吞吐仍优先 TensorRT `format=engine, nms=True`。**

```python
model.export(
    format="torchscript",
    export_mode="dual",
    nms=True,            # 推荐：图内 NMS
    max_classes=80,
    max_det=300,
    conf=0.25,
    iou=0.45,
    imgsz=640,
)
# 产出: *_vision.torchscript + *_language.torchscript
WeDetect("..._vision.torchscript").set_classes(["人", "车"]).predict("img.jpg")
```

### 8.2 Dual TensorRT（engine）

需 **GPU**。内部先导出 dual ONNX，再按输入名设置 TRT profile，生成同目录 sibling 引擎。支持 FP32 与 FP16（`quantize=16`）；**不支持 INT8**。

**默认（无 NMS，Python 后处理）**：

```python
model.export(
    format="engine",
    export_mode="dual",
    imgsz=640,
    device=0,
    quantize=16,  # 可选；省略则为 FP32
    builder_optimization_level=5,  # 可选 0-5，同 trtexec --builderOptimizationLevel；越高构建越慢、引擎往往更快
)
# 产出：*_vision.engine + *_language.engine（raw bboxes/scores）
```

**可选 EfficientNMS（GPU 端到端）**：`nms=True` 时仅在 **vision** 塔挂 `EfficientNMS_TRT`；语言塔不变。提示词文本仍可 `set_classes` 任意更换，但单次类别个数 ≤ `max_classes`（默认 80）；单图最多框数由 `max_det`（默认 300）在导出时写入插件。

```python
model.export(
    format="engine",
    export_mode="dual",
    nms=True,
    max_classes=80,  # 单次 set_classes 个数上限
    max_det=300,     # 插件 max_output_boxes
    conf=0.25,
    iou=0.45,
    imgsz=640,
    device=0,
)
```

> `nms=True` 时：TensorRT ≤10 在 vision ONNX 上挂 `EfficientNMS_TRT` 插件；**TensorRT 11+ 已移除该插件**，改为构建时注入原生 `INMSLayer`（推理 API 相同）。带插件的中间 ONNX **不能**用普通 ONNX Runtime 推理。
> TensorRT 11 为强类型网络，WeDetect 双塔暂不走 ModelOpt Autocast；TRT11 上请求 FP16 时会回退 FP32。需要 dual FP16 请使用 TensorRT 10.x。

### 8.3 Whole（单图内联双塔）

```python
model.export(format="onnx", export_mode="whole", imgsz=640, opset=17)
# 产出: *_whole.onnx
```

`whole` **不接**官方 `WeDetect(...).predict` / `format=engine`；请继续用示例脚本。

> **不要**把 `WeDetectDetect.fuse(txt_feats)` 当作动态 OV 的默认导出路径；那是固定词表加速，与换提示词冲突。当前 dual/whole 路径已禁止 fuse。

---

## 9. Dual ONNX / TensorRT 自定义提示词推理

### 9.1 官方 API（推荐）

传入 **vision** 路径即可；同目录必须存在 sibling `*_language.{onnx|engine}`（命名：`{stem}_vision.*` ↔ `{stem}_language.*`）。

```python
from ultralytics import WeDetect

m = WeDetect("runs/wedetect/ov_finetune_base/weights/best_vision.onnx")
# 或: WeDetect(".../best_vision.engine")  # TensorRT 需 GPU；nms=True 导出的 engine 同样用法
m.set_classes(["车", "人"])  # EfficientNMS engine：len(list) <= 导出时的 max_classes
m.predict(source="ultralytics/assets/bus.jpg", device=0, save=True)
```

换提示词只需再次 `set_classes([...])`，无需重新导出（EfficientNMS 路径也不锁定具体文本，只限制个数上限）。

### 9.2 底层示例脚本（仍可用）

`examples/WeDetect-ONNXRuntime/wedetect_onnx_infer.py`

```bash
python examples/WeDetect-ONNXRuntime/wedetect_onnx_infer.py \
  --vision runs/wedetect/ov_finetune_base/weights/best_vision.onnx \
  --language runs/wedetect/ov_finetune_base/weights/best_language.onnx \
  --tokenizer xlm-roberta-base \
  --source ultralytics/assets/bus.jpg \
  --classes 人,公交车,领带 \
  --imgsz 640 \
  --conf 0.25
```

流程：

1. HF Tokenizer 编码类名 → `input_ids`, `attention_mask`
2. language 塔 → `txt_feats [1,K,D]`
3. letterbox + `/255` → vision 塔 `(image, txt_feats)` → `bboxes`, `scores`
4. 置信度过滤 + NMS + 坐标还原

同一对权重换 `--classes` / `set_classes` 即可验证动态 `K`。

---

## 10. 端到端检查清单

- [ ] 图像与 `labels/*.txt` 一一对应，`class_id` 与 `names`/`class_texts` 前 `nc` 行对齐
- [ ] `class_texts` JSON 长度 **≥ `nc`**（多出行作 OV 负类）；近义可同行，**属性勿与基类同行**
- [ ] 已安装 `transformers` + `sentencepiece`；仓库根无空的 `xlm-roberta-base/`
- [ ] 若数据集 YAML 启用伪标：出现 `pseudo_labels-{model}.cache` / `labels_pseudo_merged.cache`，且旁路写出 `*_train.json`（原 `class_texts` 不变）
- [ ] `cfg=wedetect_finetune.yaml` 且 `freeze_text_encoder=False`，日志中出现文本塔 register / 独立 param group
- [ ] `best.pt` 完整（体积与同系列完整权重同量级，约数 GB）且含非空 `text_model_weights`
- [ ] `model.val(...)` / 混数多集验证指标合理
- [ ] `set_classes` 换提示词后 PyTorch 推理正常
- [ ] `export_mode=dual` 得到 vision + language 两个 ONNX（可选再导出 engine）
- [ ] `WeDetect("*_vision.onnx|.engine").set_classes(...).predict(...)` 正常出框
- [ ] ONNX/engine 下不同长度 `classes` 时类别维随 `K` 变化

---

## 11. 常见问题

**Q: 文本塔没有更新 / 显存里几乎看不到 LM？**
确认 `freeze_text_encoder=False` 且 `close_set=False`。闭集会缓存 embeddings 并丢弃在线编码路径。

**Q: 导出后换不了类名？**
确认用的是 dual/whole，而不是把文本 fuse 进视觉头的固定词表路径。

**Q: `OSError` / tokenizer 报缺 `sentencepiece`？**
先 `pip install sentencepiece`。若仍失败：检查仓库根是否存在**空的** `xlm-roberta-base/`（删除空目录后重试）；或把完整 HF 模型放到 `xlm-roberta-base/` / `checkpoints/xlm-roberta-base/`（须含 `config.json`）。也可设置 `HF_ENDPOINT` 镜像后联网下载。

**Q: `RuntimeError: failed finding central directory` 加载 `.pt`？**
权重文件不完整（拷贝/下载截断）。同系列完整 WeDetect base 微调权重通常约 **4GB** 量级；用 `md5sum` / `rsync -avP` 重新传完整文件。

**Q: 混数 `final_eval` 报 `IndexError`（混淆矩阵越界，如 index 1201）？**
曾因切换 val 集时复用上一集 dataloader、却按新 `nc` 建混淆矩阵导致。请使用已修复的 `WeDetectValidator`（独立验证会按 `args.data` 重建 loader）。升级代码后重跑验证即可。

**Q: `set_classes(["红色车"])` 框出所有车或对不齐？**
YOLO 微调若只把「红色车」写进「车」的同义组，会学成同义。属性应对应细分类框或 grounding 实例短语（§3.3 / §3.6）。

**Q: mask refine 无效？**
检查标签是否为带多边形的 segment 格式，且配置为 `mask_refine=True`。

**Q: 伪标签很慢 / 显存打不满？**
确认 `pseudo_label_batch=0`（自动）。YOLO/WeDetect 会按空闲显存抬图像 batch；SAM3 只能逐图，但会加大 prompt chunk。也可手动设 `pseudo_label_batch` 与 `pseudo_label_mem_fraction`。

**Q: 伪标签后找不到每图 txt？**
正常。当前实现只写 `pseudo_labels-{model_stem}.cache` 与 `labels_pseudo_merged.cache`，不再落盘 `labels_pseudo_merged/*.txt`。

**Q: 伪标 cache 换机器后是否要重跑教师？**
不必，只要相对 `path` 的目录布局与文件 size 不变。当前 hash 为可移植相对路径（`path_mode=rel_v1`）；加载时会把 `im_file` remap 到本机路径。旧绝对路径 hash 的 cache 会 miss 一次并自动重建。

**Q: 微调权重加载后提示词效果像随机初始化？**
检查 `.pt` 是否含 `text_model_weights`；`load` / `load_checkpoint` 应回填 `_text_sd`。旧 checkpoint 若缺该字段，需重新用当前代码训练保存。

---

## 12. 混合数据集训练（类别不一致）

Ultralytics WeDetect 已增强混数能力，对齐原版「**文本当 ID**」思路，不要求各子集本地 `class_id` 一致。

### 12.1 数据 YAML 格式

见 `ultralytics/cfg/datasets/wedetect_mixed.yaml`：

```yaml
train:
  yolo_data:
    - my_domain.yaml    # 例如只标了「车」，本地 id=0
    - coco.yaml         # COCO 80 类，本地 id 另一套
val:
  yolo_data:
    - my_domain.yaml
    - coco.yaml
# val_fitness_weights: [0.5, 0.5]  # 可选；默认对各 val 集平分
```

各子集可自带 `class_texts`；也可在顶层写一份默认 `class_texts` 下发给未配置的子集。

### 12.2 训练时发生了什么

1. `WeDetectTrainer` 识别 `train.yolo_data`，为每个子集建 `YOLOMultiModalDataset`；可选 `grounding_data` → `GroundingDataset`（§3.6）
2. **`mix_global_texts=True`（默认）**：按同义词（大小写不敏感）合并全局词表，并把各子集 `cls` remap 到全局 id（如域数据「车」与 COCO「汽车」合并为同一类）
3. Mosaic/MixUp：共享任一同义词的文本组会合并后再改 label
4. **`use_neg_queue=True`（可选）**：跨子集共享 NegQueue，把最近出现的类名动态补为负类文本
5. 单集时：`class_texts` **允许长于 `nc`**（多出的行作 OV 负类词表，不再截断）
6. **多 val**：`val.yolo_data` 可写多项；每 epoch / `final_eval` 按各自 `nc`/`names`/`class_texts` 分别验证并**重建 dataloader**。**`best.pt` fitness = 各子集 fitness 加权平均（默认平分）**；可选 `val_fitness_weights`。各子集指标写入 `results.csv` 时带 `<数据集目录名>/` 前缀。LVIS 自动优先用 `minival`。
7. **伪标签**：在各子集（或混数顶层）YAML 写 `pseudo_label*`；仅开启的 train 子集会写 cache / 旁路 `*_train.json`；val 仍读原始 `labels/`（见 §5.5）

```python
from ultralytics import WeDetect

model = WeDetect("pretrained_weights/wedetect_base.pt")
model.train(
    data="ultralytics/cfg/datasets/wedetect_mixed.yaml",
    cfg="ultralytics/cfg/wedetect_finetune.yaml",
    mix_global_texts=True,
    use_neg_queue=False,  # 子集词表都很小时可开
    freeze_text_encoder=False,
)
```

> 自有图中未标注的其它目标仍会被当背景——混 COCO/LVIS 不能消除这种标注冲突；混数主要解决「类别编号/词表不一致」与「负类文本过少」。

---

## 13. 与原版 WeDetect 对照 / 对齐说明

| 原版 (mmdet)                                      | Ultralytics                                                                         |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `wedetect_base_coco_full_tuning_*.py`           | `wedetect_finetune.yaml` + `WeDetect.train`                                     |
| `WeConcat` + 文本对齐混数                       | `wedetect_mixed.yaml` + `mix_global_texts` / `use_neg_queue`                  |
| `mask_refine` 配置                              | `wedetect_finetune_mask_refine.yaml`                                              |
| 中文`coco_zh_class_texts.json`                  | data YAML 的`class_texts`（模板：`ultralytics/cfg/datasets/texts/`）            |
| LM 在`backbone.text_model.*`                    | 顶层`text_model_weights` + `_text_sd`（格式不同，不做互通）                     |
| `LinearLR` 1000 iter、`start_factor=0.001`    | `warmup_iters=1000` + `warmup_start_factor=0.001`                               |
| `RandomLoadText` shuffle + `padding_value=''` | 已恢复 shuffle；multimodal padding 为空串                                           |
| grounding JSON（caption + tokens_positive）       | `train.grounding_data` + `GroundingDataset`（整份 JSON）                        |
| `deploy/export_onnx.py` dual/whole              | `model.export(..., export_mode="dual\|whole")`；dual 另支持 `format=engine`      |
| `eval_onnx.py`                                  | `WeDetect("*_vision.onnx\|.engine").set_classes(...).predict(...)`；示例脚本仍可用 |

### 已对齐（核心 OV 流程）

在线文本编码、`freeze_text_encoder=False`、`text_lr_mult=0.01`、AdamW、损失与 TAL、mosaic/mixup/HSV/flip/affine 主超参、val 每 epoch 刷新 `class_texts` 提示、`text_model_weights` 保存加载、dual ONNX 导出、跨数据集同义词全局词表合并、训练前伪标签整库 cache（§5.5）、混数多 val 加权 fitness。

> 默认 `wedetect_finetune.yaml` 使用 `lr0=5e-6`、`close_mosaic=1`；若需更贴近原版数值，可参考 `wedetect_finetune_mask_refine.yaml` 的 `lr0=2e-5` / `close_mosaic=4`，或在 `train()` 中覆盖。

### P1 已知差异（文档化）

| 项                             | 原版                               | Ultralytics 现状                                                                       |
| ------------------------------ | ---------------------------------- | -------------------------------------------------------------------------------------- |
| close_mosaic stage2            | 切 KeepRatio + LetterBox + affine  | 仅关闭 mosaic/mixup，无单独 letterbox 路径                                             |
| HSV hue                        | 乘法 LUT：`(x * gain) % 180`     | 加法扰动；yaml 数值同参                                                                |
| `max_aspect_ratio=100`       | RandomPerspective 过滤极端长宽比框 | 无该过滤                                                                               |
| 默认`lr0` / `close_mosaic` | 常为`2e-5` / 最后 4 epoch        | 主 cfg：`5e-6` / `1`（见上）                                                       |
| 「负文本」                     | 缺席类对比 + 可选指代              | 缺席类 / NegQueue；**语言否定句**（非红/未戴帽）无专用监督，需细分类或 grounding |

---

## 14. 一页速查命令

```bash
# 训练
python train_wedetect_ov.py

# 验证
yolo val model=runs/wedetect/ov_finetune_base/weights/best.pt data=my_dataset.yaml

# 导出 dual ONNX / TensorRT
python -c "from ultralytics import WeDetect; WeDetect('runs/wedetect/ov_finetune_base/weights/best.pt').export(format='onnx', export_mode='dual', imgsz=640)"
python -c "from ultralytics import WeDetect; WeDetect('runs/wedetect/ov_finetune_base/weights/best.pt').export(format='engine', export_mode='dual', imgsz=640, device=0)"

# dual 推理（自动找 sibling *_language.*）
python -c "from ultralytics import WeDetect; m=WeDetect('.../best_vision.onnx'); m.set_classes(['人','公交车']); m.predict('ultralytics/assets/bus.jpg', save=True)"
```
