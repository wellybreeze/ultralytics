from ultralytics import WeDetect
model = WeDetect("./runs/detect/train/wedetect_finetune_base/mixed_customer_2026-07-29/weights/best.pt")

paths = model.export(
    format="engine",
    export_mode="dual",   # 默认即为 dual
    imgsz=640,
    nms=True    
)