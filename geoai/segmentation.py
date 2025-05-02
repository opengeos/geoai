import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from transformers import (
    DefaultDataCollator,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)


class CustomDataset(Dataset):
    """Custom Dataset for loading images and masks."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: A.Compose = None,
        target_size: tuple = (256, 256),
        num_classes: int = 2,
    ):
        """
        Args:
            images_dir (str): Directory containing images.
            masks_dir (str): Directory containing masks.
            transform (A.Compose, optional): Transformations to be applied on the images and masks.
            target_size (tuple, optional): Target size for resizing images and masks.
            num_classes (int, optional): Number of classes in the masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            dict: A dictionary with 'pixel_values' and 'labels'.
        """
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size)
        mask = mask.resize(self.target_size)

        image = np.array(image)
        mask = np.array(mask)

        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        assert (
            mask.max() < self.num_classes
        ), f"Mask values should be less than {self.num_classes}, but found {mask.max()}"
        assert (
            mask.min() >= 0
        ), f"Mask values should be greater than or equal to 0, but found {mask.min()}"

        mask = mask.clone().detach().long()

        return {"pixel_values": image, "labels": mask}


def get_transform() -> A.Compose:
    """
    Returns:
        A.Compose: A composition of image transformations.
    """
    return A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def prepare_datasets(
    images_dir: str,
    masks_dir: str,
    transform: A.Compose,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Args:
        images_dir (str): Directory containing images.
        masks_dir (str): Directory containing masks.
        transform (A.Compose): Transformations to be applied.
        test_size (float, optional): Proportion of the dataset to include in the validation split.
        random_state (int, optional): Random seed for shuffling the dataset.

    Returns:
        tuple: Training and validation datasets.
    """
    dataset = CustomDataset(images_dir, masks_dir, transform)
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), test_size=test_size, random_state=random_state
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    pretrained_model: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    model_save_path: str = "./model",
    output_dir: str = "./results",
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
) -> str:
    """
    Trains the model and saves the fine-tuned model to the specified path.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        pretrained_model (str, optional): Pretrained model to fine-tune.
        model_save_path (str): Path to save the fine-tuned model. Defaults to './model'.
        output_dir (str, optional): Directory to save training outputs.
        num_epochs (int, optional): Number of training epochs.
        batch_size (int, optional): Batch size for training and evaluation.
        learning_rate (float, optional): Learning rate for training.

    Returns:
        str: Path to the saved fine-tuned model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model).to(
        device
    )
    data_collator = DefaultDataCollator(return_tensors="pt")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    return model_save_path


def load_model(
    model_path: str, device: torch.device
) -> SegformerForSemanticSegmentation:
    """
    Loads the fine-tuned model from the specified path.

    Args:
        model_path (str): Path to the model.
        device (torch.device): Device to load the model on.

    Returns:
        SegformerForSemanticSegmentation: Loaded model.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, target_size: tuple = (256, 256)) -> torch.Tensor:
    """
    Preprocesses the input image for prediction.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple, optional): Target size for resizing the image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    transform = A.Compose(
        [
            A.Resize(target_size[0], target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    image = np.array(image)
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0)


def predict_image(
    model: SegformerForSemanticSegmentation,
    image_tensor: torch.Tensor,
    original_size: tuple,
    device: torch.device,
) -> np.ndarray:
    """
    Predicts the segmentation mask for the input image.

    Args:
        model (SegformerForSemanticSegmentation): Fine-tuned model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        original_size (tuple): Original size of the image (width, height).
        device (torch.device): Device to perform inference on.

    Returns:
        np.ndarray: Predicted segmentation mask.
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(pixel_values=image_tensor)
        logits = outputs.logits
        upsampled_logits = F.interpolate(
            logits, size=original_size[::-1], mode="bilinear", align_corners=False
        )
        predictions = torch.argmax(upsampled_logits, dim=1).cpu().numpy()
    return predictions[0]


def segment_image(
    image_path: str,
    model_path: str,
    target_size: tuple = (256, 256),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> np.ndarray:
    """
    Segments the input image using the fine-tuned model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the fine-tuned model.
        target_size (tuple, optional): Target size for resizing the image.
        device (torch.device, optional): Device to perform inference on.

    Returns:
        np.ndarray: Predicted segmentation mask.
    """
    model = load_model(model_path, device)
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    image_tensor = preprocess_image(image_path, target_size)
    predictions = predict_image(model, image_tensor, original_size, device)
    return predictions


def visualize_predictions(
    image_path: str,
    segmented_mask: np.ndarray,
    target_size: tuple = (256, 256),
    reference_image_path: str = None,
) -> None:
    """
    Visualizes the original image, segmented mask, and optionally the reference image.

    Args:
        image_path (str): Path to the original image.
        segmented_mask (np.ndarray): Predicted segmentation mask.
        target_size (tuple, optional): Target size for resizing images.
        reference_image_path (str, optional): Path to the reference image.
    """
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize(target_size)
    segmented_image = Image.fromarray((segmented_mask * 255).astype(np.uint8))

    if reference_image_path:
        reference_image = Image.open(reference_image_path).convert("RGB")
        reference_image = reference_image.resize(target_size)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[1].imshow(reference_image)
        axes[1].set_title("Reference Image")
        axes[1].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if reference_image_path:
        axes[2].imshow(segmented_image, cmap="gray")
        axes[2].set_title("Segmented Image")
        axes[2].axis("off")
    else:
        axes[1].imshow(segmented_image, cmap="gray")
        axes[1].set_title("Segmented Image")
        axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    images_dir = "../datasets/Water-Bodies-Dataset/Images"
    masks_dir = "../datasets/Water-Bodies-Dataset/Masks"
    transform = get_transform()
    train_dataset, val_dataset = prepare_datasets(images_dir, masks_dir, transform)

    model_save_path = "./fine_tuned_model"
    train_model(train_dataset, val_dataset, model_save_path)

    image_path = "../datasets/Water-Bodies-Dataset/Images/water_body_44.jpg"
    reference_image_path = image_path.replace("Images", "Masks")
    segmented_mask = segment_image(image_path, model_save_path)

    visualize_predictions(
        image_path, segmented_mask, reference_image_path=reference_image_path
    )
