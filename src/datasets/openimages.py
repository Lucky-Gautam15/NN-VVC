from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OpenImagesDataset(Dataset):
    """
    Image-folder dataset for LIC training.

    The dataset recursively searches for image files inside
    the supplied root directory.

    Supported formats:
        .jpg
        .jpeg
        .png
        .bmp
        .webp
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(self, root, transform=None, crop_size=None):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        self.image_paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No supported images found in dataset directory: {self.root}"
            )

        if transform is not None:
            self.transform = transform
        elif crop_size is not None:
            self.transform = transforms.Compose([
                transforms.Resize(crop_size),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image