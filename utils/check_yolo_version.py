from ultralytics import YOLO
import ultralytics

model = YOLO("C:/Users/panosvrachnos/Documents/thesis_env/YOLO/yolo_v2/yolo_v2_weights/best.pt")

# 1. Ultralytics package currently installed
print("Installed Ultralytics:", ultralytics.__version__)

# 2. Ultralytics version that saved the checkpoint
print("Checkpoint Ultralytics:", model.ckpt.get("version", "not recorded"))

# 3. Model architecture/configuration
cfg = model.model.yaml

print("Model YAML:", cfg.get("yaml_file"))
print("Model scale:", cfg.get("scale"))   # n, s, m, l, x
print("Task:", model.task)
print("Checkpoint path:", model.ckpt_path)

# Complete architecture dictionary
print(cfg)

# Training arguments stored in the checkpoint
print(model.ckpt.get("train_args", {}))

# Human-readable architecture summary
model.info(detailed=True)

def inspect_yolo(model):
    cfg = getattr(model.model, "yaml", {}) or {}
    ckpt = getattr(model, "ckpt", {}) or {}

    return {
        "weights": str(getattr(model, "ckpt_path", None)),
        "task": getattr(model, "task", None),
        "architecture": cfg.get("yaml_file"),
        "scale": cfg.get("scale"),
        "checkpoint_ultralytics_version": ckpt.get("version"),
        "installed_ultralytics_version": ultralytics.__version__,
        "classes": len(model.names),
    }

print(inspect_yolo(model))