"""
Fundus preprocessing shared by training (torchvision), PyTorch inference and the torch-free ONNX runtime.

Only NumPy and Pillow are needed here, so an exported bundle runs on devices without PyTorch.
`preprocess` reproduces eval_transform exactly: FundusCrop -> PIL bicubic resize -> /255 -> ImageNet normalise -> CHW.
"""
import numpy as np
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class FundusCrop:
    """Crop the dark border around the fundus field of view and pad to square (from OculoMoE)."""

    def __init__(self, threshold=15, min_fraction=0.01):
        self.threshold = threshold
        self.min_fraction = min_fraction

    def __call__(self, img):
        arr = np.asarray(img.convert('L'))
        bright = arr > self.threshold
        rows = np.flatnonzero(bright.mean(1) > self.min_fraction)
        cols = np.flatnonzero(bright.mean(0) > self.min_fraction)
        if len(rows) > 10 and len(cols) > 10:
            img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))
        w, h = img.size
        s = max(w, h)
        if w != h:
            canvas = Image.new('RGB', (s, s), (0, 0, 0))
            canvas.paste(img, ((s - w) // 2, (s - h) // 2))
            img = canvas
        return img


def open_fundus(path_or_file, size):
    img = Image.open(path_or_file)
    img.draft('RGB', (size * 2, size * 2))          # fast DCT-domain JPEG downscale
    return img.convert('RGB')


def preprocess(path_or_file, size, draft=None):
    """-> float32 [3, size, size], identical to eval_transform(size)(open_fundus(path, draft or size))."""
    img = FundusCrop()(open_fundus(path_or_file, draft or size))
    img = img.resize((size, size), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))
