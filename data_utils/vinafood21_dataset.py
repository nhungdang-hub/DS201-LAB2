"""Dataset và DataLoader cho các bố cục VinaFood21 thường gặp.

Không cố định giả định ``train/class/image``: hàm discovery kiểm tra thư mục thật,
đồng thời hỗ trợ file CSV có cột đường dẫn ảnh và nhãn.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "validation": ("validation", "val", "valid"),
    "test": ("test", "testing"),
}

Sample = tuple[Path, str]


def _images_below(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _scan_class_directories(directory: Path) -> list[Sample]:
    samples: list[Sample] = []
    for class_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        samples.extend((path, class_dir.name) for path in _images_below(class_dir))
    return samples


def _find_column(fieldnames: Iterable[str], candidates: set[str]) -> str | None:
    return next((name for name in fieldnames if name.lower().strip() in candidates), None)


def _read_metadata(csv_path: Path, data_root: Path) -> list[Sample]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        path_col = _find_column(fields, {"path", "filepath", "file_path", "image", "image_path", "filename"})
        label_col = _find_column(fields, {"label", "class", "class_name", "category", "food"})
        if path_col is None or label_col is None:
            raise ValueError(
                f"{csv_path} cần có cột đường dẫn ảnh và nhãn; các cột hiện có: {fields}"
            )
        samples = []
        for row in reader:
            image_path = Path(row[path_col])
            if not image_path.is_absolute():
                image_path = data_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Ảnh trong metadata không tồn tại: {image_path}")
            samples.append((image_path, str(row[label_col]).strip()))
        return samples


def _find_named_entry(root: Path, aliases: Sequence[str], suffix: str | None = None) -> Path | None:
    candidates = {alias.lower() for alias in aliases}
    for path in root.iterdir():
        name = path.stem.lower() if suffix else path.name.lower()
        if name in candidates and (suffix is None or path.suffix.lower() == suffix):
            return path
    return None


def _discover_samples(data_root: str | Path) -> dict[str, list[Sample]]:
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy VinaFood21 tại '{root}'. Hãy giải nén dataset và truyền --data-root."
        )

    splits: dict[str, list[Sample]] = {}
    for split, aliases in SPLIT_ALIASES.items():
        split_dir = _find_named_entry(root, aliases)
        csv_file = _find_named_entry(root, aliases, suffix=".csv")
        if split_dir is not None and split_dir.is_dir():
            samples = _scan_class_directories(split_dir)
            if samples:
                splits[split] = samples
        elif csv_file is not None:
            splits[split] = _read_metadata(csv_file, root)

    # Bố cục chỉ có các thư mục lớp ngay dưới root: xem đây là tập train
    # và tách validation có phân tầng ở bước build loader.
    if not splits:
        samples = _scan_class_directories(root)
        if samples:
            splits["train"] = samples
    if not splits:
        raise ValueError(
            f"Không tìm thấy ảnh theo thư mục lớp hoặc metadata CSV trong '{root}'."
        )
    return splits


def _stratified_split(
    samples: Sequence[Sample], validation_ratio: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio phải nằm trong khoảng (0, 1).")
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    rng = random.Random(seed)
    train, validation = [], []
    for class_samples in grouped.values():
        class_samples = list(class_samples)
        rng.shuffle(class_samples)
        n_val = min(len(class_samples) - 1, max(1, round(len(class_samples) * validation_ratio)))
        if len(class_samples) < 2:
            n_val = 0
        validation.extend(class_samples[:n_val])
        train.extend(class_samples[n_val:])
    rng.shuffle(train)
    rng.shuffle(validation)
    if not validation:
        raise ValueError("Không đủ ảnh để tạo validation split có phân tầng.")
    return train, validation


def inspect_vinafood21(data_root: str | Path) -> dict[str, dict[str, int]]:
    """Trả số ảnh theo split và lớp từ dữ liệu thật trên đĩa."""
    splits = _discover_samples(data_root)
    return {
        split: dict(sorted(Counter(label for _, label in samples).items()))
        for split, samples in splits.items()
    }


def build_standard_transform(train: bool) -> Callable[[Image.Image], torch.Tensor]:
    """Augmentation cho train và preprocessing xác định cho val/test."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("Cần cài torchvision: pip install -r requirements.txt") from exc

    common = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *common,
        ])
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), *common])


class ProcessorTransform:
    """Adapter để Hugging Face processor trả trực tiếp tensor CHW."""

    def __init__(self, processor: Any, train: bool = False) -> None:
        self.processor = processor
        self.train = train

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Augment trên PIL, sau đó để processor tự resize/crop/normalize đúng
        # cấu hình checkpoint. Như vậy không normalize hai lần.
        if self.train:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        return self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)


class VinaFood21Dataset(Dataset[dict[str, torch.Tensor]]):
    """Đọc ảnh PIL RGB và trả ``image`` cùng ``label``.

    Nên tạo qua :func:`build_vinafood21_dataloaders` để mapping lớp luôn giống
    nhau ở train/validation/test.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        class_to_idx: Mapping[str, int] | None = None,
        samples: Sequence[Sample] | None = None,
    ) -> None:
        discovered = _discover_samples(data_root) if samples is None else None
        if samples is None:
            canonical = "validation" if split.lower() in {"val", "valid"} else split.lower()
            if canonical not in discovered:
                raise ValueError(f"Không có split '{canonical}' trong dataset.")
            samples = discovered[canonical]
            mapping_labels = sorted({label for values in discovered.values() for _, label in values})
        else:
            mapping_labels = sorted({label for _, label in samples})
        labels = sorted({label for _, label in samples})
        self.class_to_idx = dict(
            class_to_idx or {label: index for index, label in enumerate(mapping_labels)}
        )
        unknown = sorted(set(labels) - set(self.class_to_idx))
        if unknown:
            raise ValueError(f"Mapping thiếu các lớp: {unknown}")
        self.samples = list(samples)
        self.transform = transform or build_standard_transform(train=split.lower() == "train")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, class_name = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": torch.tensor(self.class_to_idx[class_name], dtype=torch.long),
        }


@dataclass(frozen=True)
class VinaFood21Loaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader | None
    class_to_idx: dict[str, int]
    statistics: dict[str, dict[str, int]]


def build_vinafood21_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    validation_ratio: float = 0.2,
    seed: int = 42,
    processor: Any | None = None,
) -> VinaFood21Loaders:
    """Phát hiện layout, tách validation nếu cần và tạo ba DataLoader."""
    splits = _discover_samples(data_root)
    if "train" not in splits:
        raise ValueError("Dataset cần có train split để huấn luyện.")
    if "validation" not in splits:
        splits["train"], splits["validation"] = _stratified_split(
            splits["train"], validation_ratio, seed
        )

    class_names = sorted({label for samples in splits.values() for _, label in samples})
    if len(class_names) != 21:
        raise ValueError(
            f"VinaFood21 phải có đúng 21 lớp, nhưng đã phát hiện {len(class_names)}: {class_names}"
        )
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    if processor is None:
        train_transform = build_standard_transform(train=True)
        eval_transform = build_standard_transform(train=False)
    else:
        # Processor tự resize/crop/normalize; không áp dụng Normalize lần hai.
        train_transform = ProcessorTransform(processor, train=True)
        eval_transform = ProcessorTransform(processor, train=False)

    datasets = {
        split: VinaFood21Dataset(
            data_root,
            split=split,
            samples=samples,
            class_to_idx=class_to_idx,
            transform=train_transform if split == "train" else eval_transform,
        )
        for split, samples in splits.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(datasets["validation"], shuffle=False, **loader_kwargs)
    test_loader = (
        DataLoader(datasets["test"], shuffle=False, **loader_kwargs)
        if "test" in datasets else None
    )
    statistics = {
        split: dict(sorted(Counter(label for _, label in samples).items()))
        for split, samples in splits.items()
    }
    return VinaFood21Loaders(
        train_loader, validation_loader, test_loader, class_to_idx, statistics
    )
