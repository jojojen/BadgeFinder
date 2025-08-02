"""Utility functions for image preprocessing and hashing.
This module is extracted from app.py to separate concerns.
"""

import logging
import base64
from PIL import Image
import numpy as np
import cv2
from scipy.fftpack import dct

# Configure module-level logger. The application should configure logging.
logger = logging.getLogger(__name__)

def resize_if_large(img, max_side: int = 1024):
    """Resize image if any side > max_side (keep aspect ratio).

    :param img: Input image array (BGR).
    :param max_side: Maximum allowed dimension for the longest side.
    :returns: Resized image if needed, otherwise the original image.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        logger.info(f"Resizing large image from ({w},{h}) to {new_size}")
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    logger.info(f"No resize needed for image size ({w},{h})")
    return img

def fdns_hash(image: Image.Image, hash_size: int = 12, block_size: int = 16) -> str:
    """Generate a perceptual hash for the image.

    This function uses a block DCT (discrete cosine transform) approach
    inspired by the Fourier domain normalized scalar (FDNS) hashing.

    :param image: A PIL Image to hash.
    :param hash_size: Number of blocks per dimension.
    :param block_size: Size of each block.
    :returns: Hexadecimal hash string.
    """
    logger.debug("Calculating perceptual hash for image.")
    img = image.convert("L").resize((hash_size * block_size, hash_size * block_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    pixels -= pixels.mean()
    blocks = []
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
    bits = np.array(bits).astype(int)
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)
    hex_str = format(hash_value, f"0{(hash_size * hash_size) // 4}x")
    assert all(c in "0123456789abcdef" for c in hex_str), f"Invalid hex hash: {hex_str}"
    logger.debug("Image hash generated: %s", hex_str)
    return hex_str

def compute_hsv_histogram(image_bgr, hist_size: int = 32):
    """Compute and return the normalized HSV histogram for the image.

    :param image_bgr: Input image in BGR format.
    :param hist_size: Number of bins for the hue channel.
    :returns: A list representing the normalized histogram.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [hist_size], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    logger.debug("Computed HSV histogram: %s...", h_hist[:5])
    return h_hist.tolist()

def preprocess_image(image_file):
    """Preprocess uploaded image, resize if too large, crop to badge, resize and compute hash/histogram.

    The function performs the following steps:
    1. Decode the uploaded image file into an array.
    2. Optionally resize the image if it is too large.
    3. Detect the circular badge region using Hough circles.
    4. Mask and crop the badge, then resize it to 256x256.
    5. Compute the perceptual hash and HSV histogram.
    6. Return the hash, base64 PNG of the resized image, JPEG bytes of the resized image, and histogram.

    :param image_file: File-like object containing image bytes.
    :returns: Tuple (hash_value, base64_png_str, jpeg_bytes, hist_value)
    :raises ValueError: If the image cannot be decoded or no badge is found.
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

    # Resize if the image is too large (for example, smartphone photo)
    img = resize_if_large(img, max_side=1024)

    # Circle detection (assume badge is circular)
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
    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]
    logger.debug("Circle found: center=(%s,%s), radius=%s", x, y, r)

    # Mask and crop badge
    mask = np.zeros_like(img)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img, mask)
    x1, y1, x2, y2 = max(x - r, 0), max(y - r, 0), min(x + r, img.shape[1]), min(y + r, img.shape[0])
    cropped = masked[y1:y2, x1:x2]
    logger.info(f"Cropped badge region shape: {cropped.shape}")

    img_resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    logger.info("Preprocessing: Calculating hash")
    hash_value = fdns_hash(pil_img)
    logger.info("Preprocessing: Calculating HSV histogram")
    hist_value = compute_hsv_histogram(img_resized)

    logger.info("Generated hash: %s", hash_value)
    logger.debug("HSV histogram sample: %s...", hist_value[:5])

    _, jpeg_data = cv2.imencode(".jpg", img_resized)
    resized_bytes = jpeg_data.tobytes()
    _, png_data = cv2.imencode(".png", img_resized)
    base64_img = base64.b64encode(png_data.tobytes()).decode("utf-8")

    return hash_value, base64_img, resized_bytes, hist_value

# ------------------------------------------------------------------
# Image Upload Helper
# ------------------------------------------------------------------
def upload_image(image_bytes: bytes) -> str:
    import requests
    api_endpoint = "https://catbox.moe/user/api.php"
    try:
        logger.info("Uploading image to Catbox ({} bytes)".format(len(image_bytes)))
        # Prepare multipart/form-data payload. `reqtype=fileupload` signals a file upload.
        data = {"reqtype": "fileupload"}
        files = {"fileToUpload": ("image.jpg", image_bytes)}
        resp = requests.post(api_endpoint, data=data, files=files, timeout=15)
        resp.raise_for_status()
        url = resp.text.strip()
        # Catbox may return additional text if something goes wrong; only accept valid URLs
        if url.startswith("http"):
            logger.info("Uploaded image successfully: %s", url)
            return url
        else:
            logger.error("Unexpected response from Catbox: %s", url)
            return None
    except Exception as exc:
        logger.error("Failed to upload image to Catbox: %s", exc)
        return None