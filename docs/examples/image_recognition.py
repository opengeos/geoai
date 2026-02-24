"""
Image Recognition with EuroSAT Dataset
=======================================

End-to-end example: download EuroSAT RGB, train a ResNet50 classifier,
evaluate, and save visualizations.
"""

import os

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt

from geoai.recognize import (
    evaluate_classifier,
    load_image_dataset,
    plot_confusion_matrix,
    plot_predictions,
    plot_training_history,
    predict_images,
    train_image_classifier,
)
from geoai.utils import download_file

# ── 1. Download dataset ──────────────────────────────────────────────────────

url = "https://data.source.coop/opengeos/geoai/EuroSAT_RGB.zip"
data_dir = download_file(url)
print(f"Dataset directory: {data_dir}")

# ── 2. Explore dataset ───────────────────────────────────────────────────────

dataset_info = load_image_dataset(data_dir)

# ── 3. Train ResNet50 ────────────────────────────────────────────────────────

output_dir = "image_recognition_output/resnet50"

result = train_image_classifier(
    data_dir=data_dir,
    model_name="resnet50",
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    image_size=64,
    in_channels=3,
    pretrained=True,
    output_dir=output_dir,
    num_workers=4,
    seed=42,
)

# ── 4. Plot training history ─────────────────────────────────────────────────

fig = plot_training_history(os.path.join(output_dir, "models"))
fig.savefig("training_history.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved training_history.png")

# ── 5. Evaluate on test set ──────────────────────────────────────────────────

eval_result = evaluate_classifier(
    model=result["model"],
    dataset=result["test_dataset"],
    class_names=result["class_names"],
)

# ── 6. Plot confusion matrix ─────────────────────────────────────────────────

fig = plot_confusion_matrix(
    eval_result["confusion_matrix"],
    result["class_names"],
    normalize=True,
)
fig.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved confusion_matrix.png")

# ── 7. Visualize predictions ─────────────────────────────────────────────────

test_dataset = result["test_dataset"]
test_paths = test_dataset.image_paths
test_labels = test_dataset.labels

pred_result = predict_images(
    model=result["model"],
    image_paths=test_paths[:20],
    class_names=result["class_names"],
    image_size=64,
    in_channels=3,
)

fig = plot_predictions(
    image_paths=test_paths[:20],
    predictions=pred_result["predictions"],
    true_labels=test_labels[:20],
    class_names=result["class_names"],
    probabilities=pred_result["probabilities"],
)
fig.savefig("predictions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved predictions.png")

# ── 8. Summary ────────────────────────────────────────────────────────────────

print(f"\nResNet50 test accuracy: {eval_result['accuracy']:.4f}")
print("Done!")
