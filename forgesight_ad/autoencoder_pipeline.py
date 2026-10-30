from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
import csv
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


DEFAULT_VARIANT_CONFIGS = {
    "reconstruction_only": {
        "use_multiscale": False,
        "use_feature_enhancement": False,
        "use_ssim_map": False,
    },
    "reconstruction_multiscale": {
        "use_multiscale": True,
        "use_feature_enhancement": False,
        "use_ssim_map": True,
    },
    "feature_enhanced": {
        "use_multiscale": True,
        "use_feature_enhancement": True,
        "use_ssim_map": True,
    },
}


@dataclass
class ExperimentConfig:
    project_root: Path
    dataset_root: Path
    artifacts_root: Path
    image_size: int = 384
    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 1e-3
    latent_channels: int = 256
    image_threshold_percentile: float = 95.0
    pixel_threshold_percentile: float = 99.5
    top_k_ratio: float = 0.01
    heatmap_blur_kernel: int = 9
    heatmap_blur_sigma: float = 2.0
    mask_smoothing_kernel: int = 7
    mask_min_area_ratio: float = 0.0001
    max_mask_components: int = 2
    mse_weight: float = 0.6
    ssim_weight: float = 0.4
    feature_map_weight: float = 0.08
    use_train_augmentation: bool = True
    train_augmentation_level: str = "light"
    num_workers: int = 2
    seed: int = 42
    device_name: str | None = None
    allow_random_init: bool = False

    @property
    def device(self) -> torch.device:
        if self.device_name:
            return torch.device(self.device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["dataset_root"] = str(self.dataset_root)
        payload["artifacts_root"] = str(self.artifacts_root)
        payload["device"] = str(self.device)
        return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_with_padding(image: Image.Image, image_size: int, interpolation: InterpolationMode) -> Image.Image:
    pil_interpolation = {
        InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
        InterpolationMode.NEAREST: Image.Resampling.NEAREST,
    }
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), resample=pil_interpolation[interpolation])

    canvas = Image.new(image.mode, (image_size, image_size))
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _train_augmentations(level: str):
    if level == "none":
        return []
    if level == "light":
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        ]
    if level == "medium":
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=12),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.12, hue=0.03),
        ]
    raise ValueError(f"Unknown augmentation level: {level}")


def build_transforms(image_size: int, train: bool = False, enable_augmentation: bool = False, augmentation_level: str = "none"):
    bilinear_resize = partial(
        resize_with_padding,
        image_size=image_size,
        interpolation=InterpolationMode.BILINEAR,
    )
    nearest_resize = partial(
        resize_with_padding,
        image_size=image_size,
        interpolation=InterpolationMode.NEAREST,
    )

    image_steps = [transforms.Lambda(bilinear_resize)]
    if train and enable_augmentation:
        image_steps.extend(_train_augmentations(augmentation_level))
    image_steps.append(transforms.ToTensor())

    image_transform = transforms.Compose(image_steps)
    mask_transform = transforms.Compose([
        transforms.Lambda(nearest_resize),
        transforms.ToTensor(),
    ])
    return image_transform, mask_transform


class MVTecTrainDataset(Dataset):
    def __init__(self, dataset_root: Path, category: str, transform=None):
        self.image_paths = sorted((Path(dataset_root) / category / "train" / "good").glob("*.png"))
        self.transform = transform
        if not self.image_paths:
            raise ValueError(f"No training images found for category: {category}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(image_path)


class MVTecTestDataset(Dataset):
    def __init__(self, dataset_root: Path, category: str, image_transform=None, mask_transform=None, image_size: int = 384):
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_size = image_size
        self.samples = []

        dataset_root = Path(dataset_root)
        test_root = dataset_root / category / "test"
        ground_truth_root = dataset_root / category / "ground_truth"

        for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            for image_path in sorted(defect_dir.glob("*.png")):
                if defect_type == "good":
                    mask_path = None
                    label = 0
                else:
                    mask_path = ground_truth_root / defect_type / f"{image_path.stem}_mask.png"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")
                    label = 1
                self.samples.append((image_path, mask_path, label, defect_type))

        if not self.samples:
            raise ValueError(f"No test images found for category: {category}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label, defect_type = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        if mask_path is None:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        else:
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "defect_type": defect_type,
            "path": str(image_path),
        }


def create_dataloaders(config: ExperimentConfig, category: str):
    train_image_transform, _ = build_transforms(
        config.image_size,
        train=True,
        enable_augmentation=bool(config.use_train_augmentation),
        augmentation_level=str(config.train_augmentation_level),
    )
    test_image_transform, mask_transform = build_transforms(
        config.image_size,
        train=False,
        enable_augmentation=False,
        augmentation_level="none",
    )
    train_dataset = MVTecTrainDataset(config.dataset_root, category, transform=train_image_transform)
    test_dataset = MVTecTestDataset(
        config.dataset_root,
        category,
        image_transform=test_image_transform,
        mask_transform=mask_transform,
        image_size=config.image_size,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, test_dataset, train_loader, test_loader


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        downsampled = self.pool(features)
        return features, downsampled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HybridUNetAutoencoder(nn.Module):
    def __init__(self, latent_channels: int = 256):
        super().__init__()
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        self.bottleneck = ConvBlock(256, latent_channels)
        self.dec4 = DecoderBlock(latent_channels, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 32, 32)
        self.output_layer = nn.Sequential(nn.Conv2d(32, 3, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.output_layer(x)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        return [f1, f2]


def create_model(latent_channels: int, device: torch.device) -> HybridUNetAutoencoder:
    return HybridUNetAutoencoder(latent_channels=latent_channels).to(device)


def build_feature_extractor(device: torch.device, use_feature_enhancement: bool, allow_random_init: bool = False):
    if not use_feature_enhancement:
        return None, "disabled"

    try:
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        source = "pretrained"
    except Exception as exc:
        if not allow_random_init:
            raise RuntimeError(
                "Could not load pretrained ResNet18 weights. Refusing to silently use random features; "
                "set allow_random_init=True only for an intentional baseline."
            ) from exc
        backbone = models.resnet18(weights=None)
        source = "random_init_explicit"

    extractor = ResNetFeatureExtractor(backbone).to(device)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    return extractor, source

def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def normalize_scores_per_image(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = tensor.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (tensor - mins) / (maxs - mins + eps)


def gaussian_kernel(kernel_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def gaussian_blur(tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(kernel_size, sigma, tensor.size(1), tensor.device)
    return F.conv2d(tensor, kernel, padding=kernel_size // 2, groups=tensor.size(1))


def ssim_response(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return numerator / (denominator + 1e-8)


def reconstruction_loss(reconstructions: torch.Tensor, images: torch.Tensor, mse_weight: float, ssim_weight: float):
    mse = F.mse_loss(reconstructions, images)
    ssim_loss = 1.0 - ssim_response(reconstructions, images).mean()
    total_loss = (mse_weight * mse) + (ssim_weight * ssim_loss)
    return total_loss, float(mse.detach().cpu()), float(ssim_loss.detach().cpu())


def topk_mean_scores(error_maps: torch.Tensor, top_k_ratio: float) -> torch.Tensor:
    flattened = error_maps.flatten(start_dim=1)
    k = max(1, int(flattened.size(1) * top_k_ratio))
    topk_values, _ = torch.topk(flattened, k=k, dim=1)
    return topk_values.mean(dim=1)


def _keep_largest_components(mask_2d: np.ndarray, max_components: int, min_pixels: int) -> np.ndarray:
    if max_components <= 0:
        return mask_2d.astype(np.float32)

    height, width = mask_2d.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(height):
        for col in range(width):
            if not mask_2d[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            pixels = []

            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row, current_col))
                for delta_row, delta_col in neighbors:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if 0 <= next_row < height and 0 <= next_col < width:
                        if mask_2d[next_row, next_col] and not visited[next_row, next_col]:
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))

            if len(pixels) >= min_pixels:
                components.append(pixels)

    components.sort(key=len, reverse=True)
    kept_mask = np.zeros((height, width), dtype=np.float32)
    for pixels in components[:max_components]:
        for row, col in pixels:
            kept_mask[row, col] = 1.0
    return kept_mask


def refine_predicted_mask(
    anomaly_maps: torch.Tensor,
    pixel_threshold: float,
    smoothing_kernel: int,
    min_area_ratio: float,
    max_components: int,
) -> torch.Tensor:
    binary_mask = (anomaly_maps > pixel_threshold).float()
    smoothed_mask = F.avg_pool2d(binary_mask, kernel_size=smoothing_kernel, stride=1, padding=smoothing_kernel // 2)
    refined_mask = (smoothed_mask > 0.5).float()
    min_pixels = max(1, int(refined_mask.shape[-1] * refined_mask.shape[-2] * min_area_ratio))
    filtered_masks = []
    for sample_mask in refined_mask.detach().cpu().numpy():
        kept_mask = _keep_largest_components(sample_mask[0] > 0.5, max_components=max_components, min_pixels=min_pixels)
        filtered_masks.append(kept_mask)
    return torch.from_numpy(np.stack(filtered_masks, axis=0)).unsqueeze(1).to(anomaly_maps.device)


@torch.no_grad()
def feature_distance_map(feature_extractor, images: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    if feature_extractor is None:
        return torch.zeros((images.size(0), 1, images.size(2), images.size(3)), device=images.device)

    image_features = feature_extractor(imagenet_normalize(images))
    reconstruction_features = feature_extractor(imagenet_normalize(reconstructions))
    maps = []
    for source, reconstructed in zip(image_features, reconstruction_features):
        distance = torch.mean((source - reconstructed) ** 2, dim=1, keepdim=True)
        distance = F.interpolate(distance, size=images.shape[-2:], mode="bilinear", align_corners=False)
        maps.append(distance)
    return torch.stack(maps, dim=0).mean(dim=0)


@torch.no_grad()
def anomaly_maps_and_scores(model, feature_extractor, images: torch.Tensor, variant_config: dict, config: ExperimentConfig):
    reconstructions = model(images)
    mse_map = torch.mean((images - reconstructions) ** 2, dim=1, keepdim=True)
    map_components = [0.45 * mse_map]

    if variant_config.get("use_ssim_map", False):
        ssim_error_map = 1.0 - ssim_response(reconstructions, images).mean(dim=1, keepdim=True)
        map_components.append(0.2 * ssim_error_map)

    if variant_config.get("use_multiscale", False):
        local_scale = F.avg_pool2d(mse_map, kernel_size=3, stride=1, padding=1)
        regional_scale = F.avg_pool2d(mse_map, kernel_size=9, stride=1, padding=4)
        map_components.append(0.2 * local_scale)
        map_components.append(0.15 * regional_scale)

    if variant_config.get("use_feature_enhancement", False):
        map_components.append(config.feature_map_weight * feature_distance_map(feature_extractor, images, reconstructions))

    combined_map = torch.stack(map_components, dim=0).sum(dim=0)
    smoothed_maps = gaussian_blur(combined_map, config.heatmap_blur_kernel, config.heatmap_blur_sigma)
    image_scores = topk_mean_scores(smoothed_maps, config.top_k_ratio)
    return reconstructions, smoothed_maps, image_scores


def train_autoencoder(model, dataloader, optimizer, config: ExperimentConfig):
    history = []
    for epoch in range(config.num_epochs):
        model.train()
        running_total = 0.0
        running_mse = 0.0
        running_ssim = 0.0
        start_time = time.time()

        for images, _ in dataloader:
            images = images.to(config.device)
            optimizer.zero_grad()
            reconstructions = model(images)
            loss, mse_value, ssim_value = reconstruction_loss(reconstructions, images, config.mse_weight, config.ssim_weight)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_total += loss.item() * batch_size
            running_mse += mse_value * batch_size
            running_ssim += ssim_value * batch_size

        epoch_summary = {
            "loss": running_total / len(dataloader.dataset),
            "mse": running_mse / len(dataloader.dataset),
            "ssim_loss": running_ssim / len(dataloader.dataset),
            "time_seconds": time.time() - start_time,
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch + 1:02d}/{config.num_epochs} | "
            f"loss={epoch_summary['loss']:.6f} | mse={epoch_summary['mse']:.6f} | "
            f"ssim_loss={epoch_summary['ssim_loss']:.6f} | time={epoch_summary['time_seconds']:.1f}s"
        )

    return history


@torch.no_grad()
def collect_train_statistics(model, feature_extractor, dataloader, variant_config: dict, config: ExperimentConfig):
    model.eval()
    image_scores = []
    pixel_scores = []

    for images, _ in dataloader:
        images = images.to(config.device)
        _, anomaly_maps, batch_scores = anomaly_maps_and_scores(model, feature_extractor, images, variant_config, config)
        image_scores.extend(batch_scores.cpu().numpy().tolist())
        pixel_scores.extend(anomaly_maps.flatten(start_dim=1).cpu().numpy().reshape(-1).tolist())

    return np.array(image_scores, dtype=np.float32), np.array(pixel_scores, dtype=np.float32)


def binary_iou(prediction_mask: np.ndarray, target_mask: np.ndarray) -> float:
    prediction = prediction_mask.astype(bool)
    target = target_mask.astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(prediction, target).sum()
    return float(intersection / union)


def update_top_records(bucket: list[dict], record: dict, limit: int = 3) -> None:
    bucket.append(record)
    bucket.sort(key=lambda item: item["score"], reverse=True)
    if len(bucket) > limit:
        del bucket[limit:]


@torch.no_grad()
def evaluate_model(model, feature_extractor, dataloader, image_threshold: float, pixel_threshold: float, variant_config: dict, config: ExperimentConfig):
    model.eval()
    records = []
    normal_examples = []
    anomalous_examples = []
    pixel_label_chunks = []
    pixel_score_chunks = []
    has_positive_pixel = False
    has_negative_pixel = False

    for batch in dataloader:
        images = batch["image"].to(config.device)
        reconstructions, anomaly_maps, image_scores = anomaly_maps_and_scores(model, feature_extractor, images, variant_config, config)
        predicted_masks = refine_predicted_mask(
            anomaly_maps,
            pixel_threshold,
            config.mask_smoothing_kernel,
            config.mask_min_area_ratio,
            config.max_mask_components,
        )

        for idx in range(images.size(0)):
            score = float(image_scores[idx].cpu().item())
            label = int(batch["label"][idx].cpu().item())
            anomaly_map = anomaly_maps[idx].cpu()
            predicted_mask = predicted_masks[idx].cpu()
            mask = batch["mask"][idx].cpu()
            mask_array = (mask.numpy().reshape(-1) > 0.5).astype(np.uint8)
            pixel_label_chunks.append(mask_array)
            pixel_score_chunks.append(anomaly_map.numpy().reshape(-1).astype(np.float32, copy=False))
            has_positive_pixel = has_positive_pixel or bool(mask_array.any())
            has_negative_pixel = has_negative_pixel or bool((mask_array == 0).any())

            record = {
                "path": batch["path"][idx],
                "defect_type": batch["defect_type"][idx],
                "label": label,
                "prediction": int(score > image_threshold),
                "score": score,
                "iou": binary_iou(predicted_mask.numpy() > 0.5, mask.numpy() > 0.5),
            }
            records.append(record)

            example_record = {
                **record,
                "image": images[idx].cpu(),
                "reconstruction": reconstructions[idx].cpu(),
                "anomaly_map": anomaly_map,
                "predicted_mask": predicted_mask,
                "mask": mask,
            }
            if label == 0:
                update_top_records(normal_examples, example_record)
            else:
                update_top_records(anomalous_examples, example_record)

    labels = np.array([record["label"] for record in records], dtype=np.int64)
    scores = np.array([record["score"] for record in records], dtype=np.float32)
    predictions = np.array([record["prediction"] for record in records], dtype=np.int64)
    anomalous_ious = [record["iou"] for record in records if record["label"] == 1]
    pixel_roc_auc = float("nan")
    if has_positive_pixel and has_negative_pixel:
        pixel_labels = np.concatenate(pixel_label_chunks)
        pixel_scores = np.concatenate(pixel_score_chunks)
        pixel_roc_auc = float(roc_auc_score(pixel_labels, pixel_scores))

    metrics = {
        "image_accuracy": float((predictions == labels).mean()),
        "image_roc_auc": float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else float("nan"),
        "pixel_roc_auc": pixel_roc_auc,
        "mean_iou": float(np.mean(anomalous_ious)) if anomalous_ious else float("nan"),
    }
    return records, metrics, normal_examples + anomalous_examples


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_history(training_history: list[dict], category: str, variant_name: str, output_path: Path | None = None) -> None:
    epochs = np.arange(1, len(training_history) + 1)
    losses = [entry["loss"] for entry in training_history]
    mses = [entry["mse"] for entry in training_history]
    ssim_losses = [entry["ssim_loss"] for entry in training_history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, marker="o", label="Combined loss")
    plt.plot(epochs, mses, marker="s", label="MSE")
    plt.plot(epochs, ssim_losses, marker="^", label="SSIM loss")
    plt.title(f"{category} Training Loss ({variant_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def show_predictions(records: list[dict], output_path: Path, num_examples: int = 6) -> None:
    selected = records[:num_examples]
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 5, figsize=(16, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_axes, record in zip(axes, selected):
        row_axes[0].imshow(record["image"].permute(1, 2, 0).numpy())
        row_axes[0].set_title(f"Input\n{record['defect_type']}")
        row_axes[1].imshow(record["reconstruction"].permute(1, 2, 0).numpy())
        row_axes[1].set_title("Reconstruction")
        row_axes[2].imshow(record["anomaly_map"].squeeze(0).numpy(), cmap="hot")
        row_axes[2].set_title(f"Anomaly map\nscore={record['score']:.4f}")
        row_axes[3].imshow(record["predicted_mask"].squeeze(0).numpy(), cmap="gray")
        row_axes[3].set_title(f"Predicted mask\nIoU={record['iou']:.3f}")
        row_axes[4].imshow(record["mask"].squeeze(0).numpy(), cmap="gray")
        row_axes[4].set_title(f"Ground truth\nlabel={record['label']}")
        for ax in row_axes:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def serialize_records(records: list[dict]) -> list[dict]:
    payload = []
    for record in records:
        payload.append({
            "path": record["path"],
            "defect_type": record["defect_type"],
            "label": int(record["label"]),
            "prediction": int(record["prediction"]),
            "score": float(record["score"]),
            "iou": float(record["iou"]),
        })
    return payload


def summarize_best_variants(rows: list[dict]) -> list[dict]:
    by_category = {}
    for row in rows:
        score = 0.45 * row["image_roc_auc"] + 0.35 * row["pixel_roc_auc"] + 0.20 * row["mean_iou"]
        enriched = dict(row)
        enriched["selection_score"] = score
        current = by_category.get(row["category"])
        if current is None or enriched["selection_score"] > current["selection_score"]:
            by_category[row["category"]] = enriched
    return [by_category[key] for key in sorted(by_category)]


def summarize_variant_averages(rows: list[dict]) -> list[dict]:
    by_variant = {}
    for row in rows:
        bucket = by_variant.setdefault(row["variant"], {"image": [], "pixel": [], "iou": []})
        bucket["image"].append(row["image_roc_auc"])
        bucket["pixel"].append(row["pixel_roc_auc"])
        bucket["iou"].append(row["mean_iou"])

    summary = []
    for variant in sorted(by_variant):
        bucket = by_variant[variant]
        summary.append({
            "variant": variant,
            "image_roc_auc_mean": float(np.mean(bucket["image"])),
            "pixel_roc_auc_mean": float(np.mean(bucket["pixel"])),
            "mean_iou_mean": float(np.mean(bucket["iou"])),
            "num_categories": len(bucket["image"]),
        })
    return summary


def build_results_markdown(variant_summary: list[dict], best_by_category: list[dict], rows: list[dict]) -> str:
    lines = [
        "# Experiment Results",
        "",
        "This report was generated by `main.ipynb` and summarizes the category-wise ablation study for industrial anomaly detection.",
        "",
        "## Variant Average Performance",
        "",
        "| variant | image_roc_auc_mean | pixel_roc_auc_mean | mean_iou_mean | num_categories |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in variant_summary:
        lines.append(f"| {row['variant']} | {row['image_roc_auc_mean']:.4f} | {row['pixel_roc_auc_mean']:.4f} | {row['mean_iou_mean']:.4f} | {row['num_categories']} |")

    lines.extend([
        "",
        "## Best Variant Per Category",
        "",
        "| category | best_variant | image_roc_auc | pixel_roc_auc | mean_iou |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in best_by_category:
        lines.append(f"| {row['category']} | {row['variant']} | {row['image_roc_auc']:.4f} | {row['pixel_roc_auc']:.4f} | {row['mean_iou']:.4f} |")

    lines.extend([
        "",
        "## Full Ablation Results",
        "",
        "| category | variant | image_accuracy | image_roc_auc | pixel_roc_auc | mean_iou |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in sorted(rows, key=lambda item: (item["category"], item["variant"])):
        lines.append(f"| {row['category']} | {row['variant']} | {row['image_accuracy']:.4f} | {row['image_roc_auc']:.4f} | {row['pixel_roc_auc']:.4f} | {row['mean_iou']:.4f} |")

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- Per-category, per-variant outputs are stored under `artifacts/<category>/<variant>/`.",
        "- The master machine-readable tables are `artifacts/all_ablation_summary.json` and `artifacts/all_ablation_summary.csv`.",
        "- This Markdown report is saved to `artifacts/RESULTS.md`.",
    ])
    return "\n".join(lines)

def save_feature_extractor_state(feature_extractor):
    if feature_extractor is None:
        return None
    return feature_extractor.state_dict()


def run_category_experiment(config: ExperimentConfig, category: str, variant_name: str, variant_config: dict):
    print(f"\n===== Category: {category} | Variant: {variant_name} =====")
    category_variant_dir = config.artifacts_root / category / variant_name
    category_variant_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(config, category)
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    model = create_model(config.latent_channels, config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    feature_extractor, feature_source = build_feature_extractor(
        config.device, variant_config.get("use_feature_enhancement", False), config.allow_random_init
    )

    training_history = train_autoencoder(model, train_loader, optimizer, config)
    train_image_scores, train_pixel_scores = collect_train_statistics(model, feature_extractor, train_loader, variant_config, config)
    image_threshold = float(np.percentile(train_image_scores, config.image_threshold_percentile))
    pixel_threshold = float(np.percentile(train_pixel_scores, config.pixel_threshold_percentile))

    print(f"Image threshold ({config.image_threshold_percentile}th percentile): {image_threshold:.6f}")
    print(f"Pixel threshold ({config.pixel_threshold_percentile}th percentile): {pixel_threshold:.6f}")

    test_records, test_metrics, example_records = evaluate_model(model, feature_extractor, test_loader, image_threshold, pixel_threshold, variant_config, config)
    print(test_metrics)

    model_payload = {
        "category": category,
        "variant": variant_name,
        "config": config.to_dict(),
        "variant_config": dict(variant_config),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "feature_backbone_source": feature_source,
        "feature_extractor_state_dict": save_feature_extractor_state(feature_extractor),
        "model_state_dict": model.state_dict(),
    }
    torch.save(model_payload, category_variant_dir / "model.pt")

    np.savez(category_variant_dir / "thresholds.npz", image_threshold=image_threshold, pixel_threshold=pixel_threshold)

    plot_training_history(training_history, category, variant_name, category_variant_dir / "training_history.png")
    save_json(category_variant_dir / "training_history.json", training_history)

    ordered_records = sorted(test_records, key=lambda record: record["score"], reverse=True)
    show_predictions(example_records, category_variant_dir / "qualitative_examples.png")

    metrics_payload = {
        "category": category,
        "variant": variant_name,
        "feature_backbone_source": feature_source,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        **test_metrics,
    }
    save_json(category_variant_dir / "metrics.json", metrics_payload)
    save_json(category_variant_dir / "predictions.json", serialize_records(ordered_records))

    compact_summary = {
        "category": category,
        "variant": variant_name,
        "image_accuracy": float(test_metrics["image_accuracy"]),
        "image_roc_auc": float(test_metrics["image_roc_auc"]),
        "pixel_roc_auc": float(test_metrics["pixel_roc_auc"]),
        "mean_iou": float(test_metrics["mean_iou"]),
    }
    return compact_summary, ordered_records


def run_experiments(config: ExperimentConfig, categories: list[str], variant_configs: dict | None = None):
    set_seed(config.seed)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    variant_configs = variant_configs or DEFAULT_VARIANT_CONFIGS

    all_rows = []
    all_records = {}

    for category in categories:
        all_records[category] = {}
        for variant_name, variant_config in variant_configs.items():
            summary, records = run_category_experiment(config, category, variant_name, variant_config)
            all_rows.append(summary)
            all_records[category][variant_name] = records

    best_by_category = summarize_best_variants(all_rows)
    variant_averages = summarize_variant_averages(all_rows)

    save_json(config.artifacts_root / "all_ablation_summary.json", all_rows)
    save_csv(config.artifacts_root / "all_ablation_summary.csv", all_rows, ["category", "variant", "image_accuracy", "image_roc_auc", "pixel_roc_auc", "mean_iou"])
    save_json(config.artifacts_root / "all_category_summary.json", best_by_category)
    save_csv(config.artifacts_root / "best_variant_by_category.csv", best_by_category, ["category", "variant", "image_accuracy", "image_roc_auc", "pixel_roc_auc", "mean_iou", "selection_score"])
    save_csv(config.artifacts_root / "variant_average_summary.csv", variant_averages, ["variant", "image_roc_auc_mean", "pixel_roc_auc_mean", "mean_iou_mean", "num_categories"])
    results_markdown = build_results_markdown(variant_averages, best_by_category, all_rows)
    (config.artifacts_root / "RESULTS.md").write_text(results_markdown, encoding="utf-8")

    return {
        "all_rows": all_rows,
        "best_by_category": best_by_category,
        "variant_averages": variant_averages,
        "records_by_category": all_records,
    }
def load_model_bundle(model_path: Path, device: torch.device):
    bundle = torch.load(model_path, map_location=device)
    cfg = bundle["config"]
    model = create_model(int(cfg["latent_channels"]), device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    feature_extractor = None
    if bundle["variant_config"].get("use_feature_enhancement", False):
        backbone = models.resnet18(weights=None)
        feature_extractor = ResNetFeatureExtractor(backbone).to(device)
        state = bundle.get("feature_extractor_state_dict")
        if state:
            feature_extractor.load_state_dict(state)
        feature_extractor.eval()
        for parameter in feature_extractor.parameters():
            parameter.requires_grad = False

    transform, _ = build_transforms(int(cfg["image_size"]), train=False, enable_augmentation=False, augmentation_level="none")
    return bundle, model, feature_extractor, transform


@torch.no_grad()
def infer_image(bundle: dict, model, feature_extractor, image: Image.Image, device: torch.device):
    transform, _ = build_transforms(
        int(bundle["config"]["image_size"]),
        train=False,
        enable_augmentation=False,
        augmentation_level="none",
    )
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    cfg = bundle["config"]
    runtime_config = ExperimentConfig(
        project_root=Path(cfg["project_root"]),
        dataset_root=Path(cfg["dataset_root"]),
        artifacts_root=Path(cfg["artifacts_root"]),
        image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]),
        num_epochs=int(cfg["num_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        latent_channels=int(cfg["latent_channels"]),
        image_threshold_percentile=float(cfg["image_threshold_percentile"]),
        pixel_threshold_percentile=float(cfg["pixel_threshold_percentile"]),
        top_k_ratio=float(cfg["top_k_ratio"]),
        heatmap_blur_kernel=int(cfg["heatmap_blur_kernel"]),
        heatmap_blur_sigma=float(cfg["heatmap_blur_sigma"]),
        mask_smoothing_kernel=int(cfg["mask_smoothing_kernel"]),
        mask_min_area_ratio=float(cfg["mask_min_area_ratio"]),
        max_mask_components=int(cfg.get("max_mask_components", 3)),
        mse_weight=float(cfg["mse_weight"]),
        ssim_weight=float(cfg["ssim_weight"]),
        feature_map_weight=float(cfg.get("feature_map_weight", 0.1)),
        use_train_augmentation=bool(cfg.get("use_train_augmentation", True)),
        train_augmentation_level=str(cfg.get("train_augmentation_level", "light")),
        num_workers=int(cfg["num_workers"]),
        seed=int(cfg["seed"]),
        device_name=str(device),
    )
    reconstruction, anomaly_map, image_score = anomaly_maps_and_scores(model, feature_extractor, image_tensor, bundle["variant_config"], runtime_config)
    predicted_mask = refine_predicted_mask(
        anomaly_map,
        float(bundle["pixel_threshold"]),
        runtime_config.mask_smoothing_kernel,
        runtime_config.mask_min_area_ratio,
        runtime_config.max_mask_components,
    )
    return {
        "reconstruction": reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy(),
        "heatmap": anomaly_map.squeeze(0).squeeze(0).cpu().numpy(),
        "mask": predicted_mask.squeeze(0).squeeze(0).cpu().numpy(),
        "image_score": float(image_score.item()),
        "image_threshold": float(bundle["image_threshold"]),
        "pixel_threshold": float(bundle["pixel_threshold"]),
    }
