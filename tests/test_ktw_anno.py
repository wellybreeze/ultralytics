# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.data.dataset import YOLODataset, YOLOMultiModalDataset
from ultralytics.data.utils import (
    check_det_dataset,
    ktw_plot_instances,
    ktw_unique_names,
    parse_ktw_anno,
    resolve_label_paths,
)
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace


def _ktw_v2(w=3840, h=2160):
    return {
        "shapes": [
            {
                "labels": ["未戴安全帽", "未穿反光衣"],
                "points": [[2535.0, 1160.0], [2707.0, 1387.0]],
                "group_id": 0,
                "description": "",
                "shape_type": "rectangle",
                "flags": {"difficult": False},
            }
        ],
        "imagePath": "a.jpg",
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
        "format": "ktw-anno",
        "schemaVersion": 2,
        "verified": True,
        "imageDepth": 3,
    }


def test_parse_ktw_anno_v2_rectangle():
    """Pixel xyxy rectangles become normalized xywh with the full labels list kept."""
    xywhs, tags = parse_ktw_anno(_ktw_v2(), (2160, 3840))
    assert len(xywhs) == 1
    assert tags == [["未戴安全帽", "未穿反光衣"]]
    x, y, bw, bh = xywhs[0]
    assert abs(x - (2535 + 2707) / 2 / 3840) < 1e-6
    assert abs(y - (1160 + 1387) / 2 / 2160) < 1e-6
    assert abs(bw - (2707 - 2535) / 3840) < 1e-6
    assert abs(bh - (1387 - 1160) / 2160) < 1e-6


def test_ktw_plot_instances_counts_all_tags():
    """labels.jpg must count every tag on a box, not cache cls=0."""
    boxes = np.array([[0.5, 0.5, 0.1, 0.2]], dtype=np.float32)
    labels = [{"bboxes": boxes, "instance_labels": [["人", "未戴安全帽"]], "cls": np.array([[0.0]])}]
    out = ktw_plot_instances(labels)
    assert out is not None
    b, c, names = out
    assert len(b) == 2
    assert set(names.values()) == {"人", "未戴安全帽"}
    assert {int(x) for x in c} == {0, 1}
    np.testing.assert_allclose(b[0], b[1])


def test_parse_ktw_anno_v1_boxes():
    """Schema v1 xmin/xmax boxes parse the same way."""
    obj = {
        "format": "ktw-anno",
        "version": 1,
        "boxes": [
            {"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 40, "labels": ["car", "bigcar"], "difficult": False},
            {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2, "labels": ["car"], "difficult": True},
        ],
    }
    xywhs, tags = parse_ktw_anno(obj, (100, 100))
    assert len(xywhs) == 1
    assert tags == [["car", "bigcar"]]


def test_resolve_label_paths_txt_wins(tmp_path: Path):
    """A YOLO txt sidecar keeps existing algorithm datasets unchanged when both files exist."""
    img = tmp_path / "images" / "a.jpg"
    img.parent.mkdir()
    img.touch()
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (labels / "a.json").write_text("{}")
    got = resolve_label_paths([str(img)])
    assert got[0].endswith("a.txt")


def test_resolve_label_paths_json_when_no_txt(tmp_path: Path):
    """Ktw-anno JSON is used when it is the only label file."""
    img = tmp_path / "images" / "a.jpg"
    img.parent.mkdir()
    img.touch()
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.json").write_text("{}")
    got = resolve_label_paths([str(img)])
    assert got[0].endswith("a.json")


def test_yolo_dataset_ktw_cache_and_sample(tmp_path: Path):
    """Cache stores instance_labels; train samples one class id from the box's full list."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80), color=(0, 0, 0)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [
            {
                "labels": ["hat", "vest"],
                "points": [[10, 10], [50, 50]],
                "shape_type": "rectangle",
                "flags": {"difficult": False},
            }
        ],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "hat", 1: "vest"}, "nc": 2, "channels": 3}
    ds = YOLODataset(img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False)
    assert "instance_labels" in ds.labels[0]
    assert ds.labels[0]["instance_labels"] == [["hat", "vest"]]
    ds2 = YOLODataset(img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False)
    assert ds2.labels[0]["instance_labels"] == [["hat", "vest"]]
    random.seed(0)
    seen = set()
    for _ in range(40):
        item = ds.get_image_and_label(0)
        cid = int(item["cls"].reshape(-1)[0])
        seen.add(cid)
        assert cid in {0, 1}
    assert seen == {0, 1}


def test_yolo_dataset_ktw_val_expands_all_tags(tmp_path: Path):
    """Val does not sample: each in-vocab tag becomes its own GT (same box, independent classes)."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [{"labels": ["hat", "vest"], "points": [[10, 10], [50, 50]], "shape_type": "rectangle"}],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "hat", 1: "vest"}, "nc": 2, "channels": 3}
    ds = YOLODataset(img_path=str(images), data=data, task="detect", augment=False, imgsz=64, cache=False)
    item = ds.get_image_and_label(0)
    assert sorted(int(x) for x in item["cls"].reshape(-1)) == [0, 1]
    assert len(item["instances"].bboxes) == 2
    np.testing.assert_allclose(item["instances"].bboxes[0], item["instances"].bboxes[1])


def test_multimodal_ktw_unknown_text_appended(tmp_path: Path):
    """WeDetect OV keeps instance-specific phrases that are not in class_texts as extra texts."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [{"labels": ["yellow hat"], "points": [[10, 10], [50, 50]], "shape_type": "rectangle"}],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "hat"}, "nc": 1, "channels": 3}
    ds = YOLOMultiModalDataset(img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False)
    item = ds.get_image_and_label(0)
    assert "yellow hat" in {t[0] for t in item["texts"]}
    assert int(item["cls"].reshape(-1)[0]) == len(ds.class_texts)
    np.testing.assert_array_equal(item["cls"].shape, (1, 1))


def test_check_det_dataset_names_optional_for_ktw(tmp_path: Path):
    """Ktw-anno YAML may omit names/nc; a placeholder is filled and flagged for later inference."""
    images = tmp_path / "images"
    images.mkdir()
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(f"path: {tmp_path}\ntrain: images\nval: images\n")
    data = check_det_dataset(str(yaml_path), autodownload=False)
    assert data["_ktw_auto_names"] is True
    assert data["nc"] == 1
    assert data["names"][0] == "object"


def test_yolo_dataset_ktw_infers_closed_set_names(tmp_path: Path):
    """Closed-set YOLO without names infers unique JSON strings as classes."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [{"labels": ["hat", "vest"], "points": [[10, 10], [50, 50]], "shape_type": "rectangle"}],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "object"}, "nc": 1, "channels": 3, "_ktw_auto_names": True}
    ds = YOLODataset(img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False)
    assert set(ds.data["names"].values()) == {"hat", "vest"}
    assert ds.data["nc"] == 2
    assert ds.ktw_open is False
    random.seed(0)
    seen = set()
    for _ in range(2):
        seen.add(int(ds.get_image_and_label(0)["cls"].reshape(-1)[0]))
    assert seen == {0, 1}


def test_multimodal_ktw_without_names_per_image_texts(tmp_path: Path):
    """Nameless WeDetect ktw-anno uses per-image texts, not a global phrase vocab."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [
            {"labels": ["未戴安全帽", "未穿反光衣"], "points": [[10, 10], [50, 50]], "shape_type": "rectangle"},
            {"labels": ["人"], "points": [[60, 10], [90, 40]], "shape_type": "rectangle"},
        ],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "object"}, "nc": 1, "channels": 3, "_ktw_auto_names": True}
    hyp = IterableSimpleNamespace(
        **{**vars(DEFAULT_CFG), "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}
    )
    ds = YOLOMultiModalDataset(
        img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False, hyp=hyp
    )
    assert ds.ktw_open is True
    assert ds.data["names"][0] == "object"  # train does not inflate a global vocab
    random.seed(0)
    item = ds.get_image_and_label(0)
    phrases = {t[0] if isinstance(t, list) else t for t in item["texts"] if t}
    phrases.discard("")
    assert "object" not in phrases
    assert "人" in phrases
    assert ("未戴安全帽" in phrases) or ("未穿反光衣" in phrases)
    assert int(item["cls"].shape[0]) == 2


def test_multimodal_ktw_val_infers_all_tags(tmp_path: Path):
    """Val without YAML names uses every unique JSON tag in this split as a prompt / GT class."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [
            {"labels": ["hat", "vest"], "points": [[10, 10], [50, 50]], "shape_type": "rectangle"},
            {"labels": ["person"], "points": [[60, 10], [90, 40]], "shape_type": "rectangle"},
        ],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "object"}, "nc": 1, "channels": 3, "_ktw_auto_names": True}
    ds = YOLOMultiModalDataset(img_path=str(images), data=data, task="detect", augment=False, imgsz=64, cache=False)
    assert ds.ktw_open is True
    assert set(ktw_unique_names(ds.labels)) == {"hat", "person", "vest"}
    assert set(ds.data["names"].values()) == {"hat", "person", "vest"}
    assert ds.data["nc"] == 3
    item = ds.get_image_and_label(0)
    cids = [int(x) for x in item["cls"].reshape(-1)]
    assert {ds.data["names"][c] for c in cids} == {"hat", "person", "vest"}
    assert len(cids) == 3


def test_yolo_dataset_val_placeholder_names_infers_ktw(tmp_path: Path):
    """Val with placeholder names={0: object} infers JSON tags even if _ktw_auto_names was dropped."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [{"labels": ["hat", "vest"], "points": [[10, 10], [50, 50]], "shape_type": "rectangle"}],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "object"}, "nc": 1, "channels": 3}
    ds = YOLODataset(img_path=str(images), data=data, task="detect", augment=False, imgsz=64, cache=False)
    assert set(ds.data["names"].values()) == {"hat", "vest"}
    item = ds.get_image_and_label(0)
    assert sorted(int(x) for x in item["cls"].reshape(-1)) == [0, 1]


def test_multimodal_ktw_open_texts_padded_for_batch(tmp_path: Path):
    """Per-image vocab must pad to a fixed length so encode_texts can reshape (B, N, D)."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for name, tags in (("a", ["hat"]), ("b", ["hat", "vest", "person"])):
        Image.new("RGB", (100, 80)).save(images / f"{name}.jpg")
        obj = {
            "format": "ktw-anno",
            "schemaVersion": 2,
            "imageWidth": 100,
            "imageHeight": 80,
            "shapes": [{"labels": tags, "points": [[10, 10], [50, 50]], "shape_type": "rectangle"}],
        }
        (labels / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {"names": {0: "object"}, "nc": 1, "channels": 3, "_ktw_auto_names": True}
    hyp = IterableSimpleNamespace(
        **{**vars(DEFAULT_CFG), "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}
    )
    ds = YOLOMultiModalDataset(
        img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False, hyp=hyp
    )
    batch = ds.collate_fn([ds[0], ds[1]])
    assert len(batch["texts"]) == 2
    assert len(batch["texts"][0]) == len(batch["texts"][1]) == 80


def test_ktw_realign_val_names_placeholder_vs_inferred():
    """final_eval yaml has names={0: object}; prompts must follow JSON-inferred classes."""
    from ultralytics.models.yolo.wedetect.val import ktw_realign_val_names

    data = {"names": {0: "人", 1: "未戴安全帽", 2: "未穿反光衣"}, "nc": 3, "_ktw_auto_names": True}
    assert ktw_realign_val_names(data, {0: "object"}) == ["人", "未戴安全帽", "未穿反光衣"]
    assert ktw_realign_val_names(data, {0: "人", 1: "未戴安全帽", 2: "未穿反光衣"}) is None


def test_ktw_realign_val_names_closed_set_untouched():
    """YAML/class_texts vocabs must not be replaced just because strings differ."""
    from ultralytics.models.yolo.wedetect.val import ktw_realign_val_names

    data = {"names": {0: "person", 1: "car"}, "nc": 2}
    assert ktw_realign_val_names(data, {0: "person", 1: "car"}) is None
    assert ktw_realign_val_names(data, {0: "人", 1: "车"}) is None


def test_confusion_matrix_placeholder_nc_raises_on_ktw_gt():
    """Reproduce final_eval IndexError: 1-class CM vs GT class id 2."""
    import torch

    from ultralytics.utils.metrics import ConfusionMatrix

    cm = ConfusionMatrix(names={0: "object"})
    pred = {
        "bboxes": torch.zeros(0, 4),
        "conf": torch.zeros(0),
        "cls": torch.zeros(0),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "cls": torch.tensor([2.0]),
        "ori_shape": (10, 10),
        "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
    }
    try:
        cm.process_batch(pred, batch)
        raise AssertionError("expected IndexError when GT class exceeds confusion-matrix nc")
    except IndexError:
        pass
    aligned = ConfusionMatrix(names={0: "人", 1: "未戴安全帽", 2: "未穿反光衣"})
    aligned.process_batch(pred, batch)
    assert aligned.matrix.shape == (4, 4)


def test_closed_ktw_train_keeps_one_box_multihot(tmp_path: Path):
    """Closed-vocab WeDetect train: one physical box, all tags on as multi-hot (not one sampled id)."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "a.jpg")
    obj = {
        "format": "ktw-anno",
        "schemaVersion": 2,
        "imageWidth": 100,
        "imageHeight": 80,
        "shapes": [
            {
                "labels": ["人", "未戴安全帽的人", "未穿反光衣的人"],
                "points": [[10, 10], [50, 50]],
                "shape_type": "rectangle",
            }
        ],
    }
    (labels / "a.json").write_text(json.dumps(obj), encoding="utf-8")
    data = {
        "names": {0: "人", 1: "未戴安全帽的人", 2: "未穿反光衣的人"},
        "nc": 3,
        "channels": 3,
    }
    hyp = IterableSimpleNamespace(
        **{**vars(DEFAULT_CFG), "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0}
    )
    ds = YOLOMultiModalDataset(
        img_path=str(images), data=data, task="detect", augment=True, imgsz=64, cache=False, hyp=hyp
    )
    assert ds.ktw_open is False
    item = ds.get_image_and_label(0)
    assert len(item["cls"]) == 1
    assert int(item["cls"].reshape(-1)[0]) == 0
    mh = item["cls_multihot"]
    assert mh.shape == (1, 3)
    np.testing.assert_array_equal(mh[0], [1, 1, 1])
    batch = ds.collate_fn([ds[0]])
    assert batch["cls"].shape[0] == 1
    assert batch["cls_multihot"].shape[0] == 1
    assert batch["cls_multihot"].shape[1] >= 3
    assert float(batch["cls_multihot"][0, :3].sum()) == 3.0


def test_tal_multihot_targets_keep_coexisting_classes():
    """TAL one-hot would zero co-labels; multi-hot must keep [1,1,0] on the assigned prior."""
    import torch

    from ultralytics.utils.tal import TaskAlignedAssigner

    a = TaskAlignedAssigner(num_classes=3)
    a.bs = 1
    a.n_max_boxes = 1
    gt_labels = torch.tensor([[[0.0]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    target_gt_idx = torch.tensor([[0, 0]])
    fg_mask = torch.tensor([[1.0, 1.0]])
    _, _, onehot = a.get_targets(gt_labels, gt_bboxes, target_gt_idx.clone(), fg_mask)
    assert onehot[0, 0].tolist() == [1, 0, 0]
    gt_multi = torch.tensor([[[1.0, 1.0, 0.0]]])
    _, _, multi = a.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask, gt_multi_hot=gt_multi)
    assert multi[0, 0].tolist() == [1, 1, 0]


def test_tal_multihot_alignment_uses_max_positive_class():
    """Subset-class score must be able to win assignment even if primary class logit is low."""
    import torch

    from ultralytics.utils.tal import TaskAlignedAssigner

    a = TaskAlignedAssigner(num_classes=3, alpha=1.0, beta=1.0)
    a.bs = 1
    a.n_max_boxes = 1
    pd_scores = torch.zeros(1, 2, 3)
    pd_scores[0, 0, 0] = 0.1
    pd_scores[0, 0, 1] = 0.9
    pd_scores[0, 1, 0] = 0.2
    pd_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]])
    gt_labels = torch.tensor([[[0.0]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    mask_gt = torch.ones(1, 1, 2)
    gt_multi = torch.tensor([[[1.0, 1.0, 0.0]]])
    align_one, _ = a.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)
    align_multi, _ = a.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, gt_multi_hot=gt_multi)
    assert float(align_multi[0, 0, 0]) > float(align_one[0, 0, 0])


def test_random_load_text_remaps_multihot_columns():
    """RandomLoadText must permute multi-hot columns with sampled text ids, not drop co-labels."""
    from ultralytics.data.augment import RandomLoadText
    from ultralytics.utils.instance import Instances

    labels = {
        "texts": [["人"], ["未戴安全帽的人"], ["未穿反光衣的人"]],
        "cls": np.array([[0]], dtype=np.int64),
        "cls_multihot": np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        "instances": Instances(np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32), normalized=True),
    }
    loader = RandomLoadText(max_samples=3, padding=True, padding_value=[""], neg_samples=(3, 3))
    out = loader(labels)
    texts = [t if isinstance(t, str) else t[0] for t in out["texts"]]
    mh = out["cls_multihot"][0]
    assert mh[texts.index("人")] == 1
    assert mh[texts.index("未戴安全帽的人")] == 1
    assert mh[texts.index("未穿反光衣的人")] == 0


def test_wedetect_validator_forces_multi_label():
    """Train-val and standalone ``model.val()`` both use WeDetectValidator with multi_label on."""
    import inspect

    from ultralytics.models.yolo.wedetect.val import WeDetectUniValidator, WeDetectValidator

    args = {"model": "yolo26n.pt", "data": "coco8.yaml"}
    v = WeDetectValidator(args=args)
    assert v.args.multi_label is True
    assert "multi_label=True" in inspect.getsource(WeDetectValidator.postprocess)
    uni = WeDetectUniValidator(args=args)
    assert uni.args.multi_label is True


def test_wedetect_final_eval_does_not_auto_enable_lvis_json():
    """Standalone / train-end val must not flip save_json on for LVIS (771MB JSON + coco-eval OOM)."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.wedetect.val import WeDetectValidator

    v = WeDetectValidator(args={"model": "yolo26n.pt", "data": "coco8.yaml", "save_json": False, "val": True})
    v.training = False
    v.data = {"val": "/datasets/lvis/images/val2017"}
    model = SimpleNamespace(names={0: "object"}, end2end=False)
    v.init_metrics(model)
    assert v.is_lvis is True
    assert v.args.save_json is False

    v.args.save_json = True
    v.init_metrics(model)
    assert v.args.save_json is True


def test_prepare_prompts_on_autobackend_like_wrapper():
    """Standalone model.val() wraps WeDetect in AutoBackend; init_metrics must still encode prompts."""
    from torch import nn

    from ultralytics.models.yolo.wedetect.val import prepare_wedetect_text_prompts

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.names = {0: "object"}
            self.txt_feats = None
            self.prompts = None

        def set_classes(self, text, batch=80, cache_clip_model=True):
            self.prompts = list(text)
            self.names = dict(enumerate(text))

    class Backend:
        def __init__(self, model):
            self.model = model
            self.names = {0: "object"}

    class Wrapper(nn.Module):
        def __init__(self, backend):
            super().__init__()
            self.backend = backend

        def __getattr__(self, name):
            if "backend" in self.__dict__ and hasattr(self.backend, name):
                return getattr(self.backend, name)
            return super().__getattr__(name)

    inner = Inner()
    wrapper = Wrapper(Backend(inner))
    names = ["人", "未戴安全帽的人", "未穿反光衣的人"]
    prepare_wedetect_text_prompts(wrapper, names)
    assert inner.prompts == names
    assert list(wrapper.names.values()) == names
    assert list(wrapper.backend.names.values()) == names


def test_resolve_class_texts_not_truncated_when_ktw_auto_names(tmp_path: Path):
    """Placeholder nc=1 must not keep only the first class_texts row for ktw-anno val."""
    from ultralytics.models.yolo.wedetect.val import resolve_wedetect_class_names

    texts = tmp_path / "class_texts.json"
    texts.write_text(json.dumps([["人"], ["未戴安全帽的人"], ["未穿反光衣的人"]], ensure_ascii=False), encoding="utf-8")
    data = {
        "names": {0: "object"},
        "nc": 1,
        "class_texts": str(texts),
        "_ktw_auto_names": True,
    }
    assert resolve_wedetect_class_names(data) == ["人", "未戴安全帽的人", "未穿反光衣的人"]


def test_mixed_val_yaml_paths_from_dict_and_file(tmp_path: Path):
    """Standalone val / final_eval must see every mixed val.yolo_data entry, not only [0]."""
    from ultralytics.models.yolo.wedetect.val import mixed_val_yaml_paths

    data = {"train": {"yolo_data": ["a.yaml"]}, "val": {"yolo_data": ["lvis.yaml", "test_datasets/data.yaml"]}}
    assert mixed_val_yaml_paths(data) == ["lvis.yaml", "test_datasets/data.yaml"]
    p = tmp_path / "mixed.yaml"
    p.write_text(
        "train:\n  yolo_data: [a.yaml]\nval:\n  yolo_data:\n    - lvis.yaml\n    - ppe.yaml\n", encoding="utf-8"
    )
    assert mixed_val_yaml_paths(p) == ["lvis.yaml", "ppe.yaml"]
    assert mixed_val_yaml_paths("test_datasets/data.yaml") == []
