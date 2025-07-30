"""
Image-processing helper functions used by BadgeFinder.

This module centralises all pure image logic so that app.py stays minimal.
"""

import base64
import logging
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from scipy.fftpack import dct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# 1. resize_if_large
# ---------------------------------------------------------------
def resize_if_large(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Resize image if any side exceeds *max_side* pixels (keep aspect ratio)."""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        logger.info("Resizing large image from (%d,%d) to %s", w, h, new_size)
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    logger.info("No resize needed for image size (%d,%d)", w, h)
    return img


# ---------------------------------------------------------------
# 2. fdns_hash
# ---------------------------------------------------------------
def fdns_hash(image: Image.Image, hash_size: int = 12, block_size: int = 16) -> str:
    """
    Generate a perceptual hash for *image* using a frequency-domain method
    (DCT over non-overlapping blocks + median thresholding).
    """
    logger.debug("Calculating perceptual hash")
    img = image.convert("L").resize(
        (hash_size * block_size, hash_size * block_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(img, dtype=np.float32)
    pixels -= pixels.mean()

    blocks: List[Tuple[float, float]] = []
    for row in range(0, pixels.shape[0], block_size):
        for col in range(0, pixels.shape[1], block_size):
            patch = pixels[row : row + block_size, col : col + block_size]
            dct_block = dct(dct(patch.T, norm="ortho").T, norm="ortho")
            dc = dct_block[0, 0]
            ac = np.abs(dct_block[1:, 1:]).mean()
            blocks.append((dc, ac))

    dcs = np.array([b[0] for b in blocks])
    acs = np.array([b[1] for b in blocks])
    median_dc = np.median(dcs)
    median_ac = np.median(acs)

    bits = [(dc > median_dc or ac > median_ac) for dc, ac in blocks]
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    hex_str = format(hash_value, f"0{(hash_size * hash_size) // 4}x")
    assert all(c in "0123456789abcdef" for c in hex_str), f"Invalid hex hash: {hex_str}"
    logger.debug("Image hash generated: %s", hex_str)
    return hex_str


# ---------------------------------------------------------------
# 3. compute_hsv_histogram
# ---------------------------------------------------------------
def compute_hsv_histogram(image_bgr: np.ndarray, hist_size: int = 32) -> list:
    """Return a L1-normalised HSV histogram (Hue channel only)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [hist_size], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    logger.debug("Computed HSV histogram: %s...", h_hist[:5])
    return h_hist.tolist()


# ---------------------------------------------------------------
# 4. hamming_dist  (moved from identify_badge)
# ---------------------------------------------------------------
def hamming_dist(hash1: str, hash2: str) -> int:
    """Return Hamming distance between two hex strings representing binary hashes."""
    return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")


# ---------------------------------------------------------------
# 5. compare_hist  (moved from identify_badge)
# ---------------------------------------------------------------
def compare_hist(hist1: list, hist2: list) -> float:
    """Return correlation coefficient between two HSV histograms."""
    h1 = np.array(hist1, dtype=np.float32)
    h2 = np.array(hist2, dtype=np.float32)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


# ---------------------------------------------------------------
# 6. preprocess_image
# ---------------------------------------------------------------
def preprocess_image(image_file):
    """
    Full preprocessing pipeline:

    1. Read bytes ➜ decode to BGR (cv2)
    2. Resize if too large
    3. Detect circular badge ➜ crop
    4. Resize cropped badge to 256×256
    5. Generate perceptual hash + HSV histogram
    6. Return (hash, base64_png, resized_bytes_jpeg, histogram_list)
    """
    logger.debug("Starting image preprocessing")

    image_file.seek(0)
    img_bytes = image_file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode image")
        raise ValueError("Failed to decode image")
    logger.debug("Image decoded: shape %s", img.shape)

    # Step 1: optional resize
    img = resize_if_large(img, max_side=1024)

    # Step 2: circle detection (badge is circular)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=600,
    )
    if circles is None:
        logger.error("No circular badge found")
        raise ValueError("No circular badge found")
    x, y, r = np.round(circles[0, 0]).astype("int")
    logger.debug("Circle found: center=(%d,%d), radius=%d", x, y, r)

    # Step 3: mask + crop
    mask = np.zeros_like(img)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img, mask)
    x1, y1, x2, y2 = (
        max(x - r, 0),
        max(y - r, 0),
        min(x + r, img.shape[1]),
        min(y + r, img.shape[0]),
    )
    cropped = masked[y1:y2, x1:x2]
    logger.info("Cropped badge region shape: %s", cropped.shape)

    # Step 4: final resize
    img_resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LANCZOS4)

    # Step 5: hash + histogram
    pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    hash_value = fdns_hash(pil_img)
    hist_value = compute_hsv_histogram(img_resized)
    logger.info("Generated hash: %s", hash_value)

    # Step 6: binary outputs
    _, jpeg_data = cv2.imencode(".jpg", img_resized)
    resized_bytes = jpeg_data.tobytes()
    _, png_data = cv2.imencode(".png", img_resized)
    base64_img = base64.b64encode(png_data.tobytes()).decode("utf-8")

    return hash_value, base64_img, resized_bytes, hist_value
