from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


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

    Note:
        Open Images V6 contains additional metadata and annotations.
        For the LIC image-compression stage, this prototype loader
        only needs the training images.
    """

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image