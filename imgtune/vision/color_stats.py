"""Pure numpy/OpenCV colour statistics.  No ML, no GPU.

Must run in <100 ms on a 1024 px image (resize to 512 px first).
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from imgtune.core.schemas import ColorStats


def _resize_longest_edge(img: np.ndarray, max_edge: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def compute_color_stats(image_bgr: np.ndarray) -> ColorStats:
    """Return a ColorStats from a BGR numpy image."""
    img = _resize_longest_edge(image_bgr, 512)

    # HLS conversion
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h_ch = hls[:, :, 0].astype(np.float64)
    l_ch = hls[:, :, 1].astype(np.float64)
    s_ch = hls[:, :, 2].astype(np.float64)

    mean_lightness = float(np.mean(l_ch) / 255.0)
    mean_saturation = float(np.mean(s_ch) / 255.0)

    # Contrast: normalised std of lightness (0.5 std ≈ max practical value)
    contrast = float(min(np.std(l_ch / 255.0) / 0.5, 1.0))

    # Edge density (Canny)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Warm ratio (reds / oranges / yellows in OpenCV hue 0-180)
    warm_mask = ((h_ch <= 25) | (h_ch >= 155)) & (s_ch > 30)
    warm_ratio = float(np.mean(warm_mask))

    # Colour entropy (hue histogram, Shannon, normalised)
    n_bins = 36
    hist = cv2.calcHist([hls], [0], None, [n_bins], [0, 180]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    max_entropy = np.log2(n_bins)
    color_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Dominant colours (Mini-Batch KMeans, 5 clusters)
    pixels = img.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 10_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), 10_000, replace=False)
        pixels = pixels[idx]
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, n_init=1, batch_size=256)
    kmeans.fit(pixels)
    centres = kmeans.cluster_centers_.astype(int)
    # BGR → RGB
    dominant_colors = [(int(c[2]), int(c[1]), int(c[0])) for c in centres]

    return ColorStats(
        mean_lightness=mean_lightness,
        mean_saturation=mean_saturation,
        contrast=contrast,
        edge_density=edge_density,
        warm_ratio=warm_ratio,
        color_entropy=color_entropy,
        dominant_colors=dominant_colors,
    )
