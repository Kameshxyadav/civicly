import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification


def load_artifacts(model_dir: Path):
    config_path = model_dir / "config.json"
    weights_path = model_dir / "best_model.pth"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    class_names = cfg["class_names"]
    model_name = cfg["model_name"]
    num_classes = int(cfg["num_classes"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, processor, class_names, device


def predict(model, processor, class_names, device, image_path: Path, top_k: int):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    top_k = max(1, min(top_k, probs.numel()))
    top_probs, top_idx = torch.topk(probs, top_k)

    items = []
    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
        items.append({"label": class_names[idx], "score": float(prob)})
    return items


def main():
    parser = argparse.ArgumentParser(description="Run waste image model inference.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--top-k", type=int, default=3, help="Top classes to return.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_dir = root / "waste_classifier_model"

    model, processor, class_names, device = load_artifacts(model_dir)
    items = predict(model, processor, class_names, device, Path(args.image), args.top_k)
    print(json.dumps({"items": items}))


if __name__ == "__main__":
    main()
