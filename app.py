from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import json
from PIL import Image
import io
import numpy as np
import base64
import cv2
import logging
from scipy.fftpack import dct
import requests
from dotenv import load_dotenv
load_dotenv()

# Set up Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get('BADGE_DB_PATH', 'badges.db')

# === Read environment variables ===
GROK_API_URL = os.environ.get('GROK_API_URL', 'https://api.x.ai/v1/chat/completions')
GROK_API_KEY = os.environ.get('GROK_API_KEY', 'your_api_key_here')
COLOR_SCORE_DIST_THRESHOLD_1 = float(os.environ.get('COLOR_SCORE_DIST_THRESHOLD_1', 14))
COLOR_SCORE_CORR_THRESHOLD_1 = float(os.environ.get('COLOR_SCORE_CORR_THRESHOLD_1', 0.75))
COLOR_SCORE_DIST_THRESHOLD_2 = float(os.environ.get('COLOR_SCORE_DIST_THRESHOLD_2', 30))
COLOR_SCORE_CORR_THRESHOLD_2 = float(os.environ.get('COLOR_SCORE_CORR_THRESHOLD_2', 0.92))

def init_db():
    """Initialize SQLite DB and create table if it doesn't exist."""
    logger.debug(f"Initializing database at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS badges (
            id INTEGER PRIMARY KEY,
            image_hash TEXT UNIQUE,
            source_work TEXT,
            character TEXT,
            purchase_method TEXT,
            suggested_price TEXT,
            auction_description TEXT,
            color_hist TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

init_db()

def fdns_hash(image: Image.Image, hash_size=12, block_size=16) -> str:
    """Generate perceptual hash for the image."""
    logger.debug("Calculating perceptual hash for image.")
    img = image.convert("L").resize((hash_size * block_size, hash_size * block_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    pixels -= pixels.mean()
    blocks = []
    for row in range(0, pixels.shape[0], block_size):
        for col in range(0, pixels.shape[1], block_size):
            patch = pixels[row:row + block_size, col:col + block_size]
            dct_block = dct(dct(patch.T, norm='ortho').T, norm='ortho')
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
    hex_str = format(hash_value, f'0{(hash_size * hash_size) // 4}x')
    assert all(c in '0123456789abcdef' for c in hex_str), f"Invalid hex hash: {hex_str}"
    logger.debug(f"Image hash generated: {hex_str}")
    return hex_str

def compute_hsv_histogram(image_bgr, hist_size=32):
    """Compute and return the normalized HSV histogram for the image."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [hist_size], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    logger.debug(f"Computed HSV histogram: {h_hist[:5]}...")
    return h_hist.tolist()

def preprocess_image(image_file):
    """Preprocess uploaded image, crop to badge, resize, return hash/hist."""
    logger.debug("Starting image preprocessing")
    image_file.seek(0)
    img_bytes = image_file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode image")
        raise ValueError("Failed to decode image")
    logger.debug(f"Image decoded: shape {img.shape}")

    # Circle detection (assume badge is circular)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=100, maxRadius=600)
    if circles is None:
        logger.error("No circular badge found")
        raise ValueError("No circular badge found")
    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]
    logger.debug(f"Circle found: center=({x},{y}), radius={r}")

    # Mask and crop badge
    mask = np.zeros_like(img)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
    masked = cv2.bitwise_and(img, mask)
    x1, y1, x2, y2 = max(x - r, 0), max(y - r, 0), min(x + r, img.shape[1]), min(y + r, img.shape[0])
    cropped = masked[y1:y2, x1:x2]

    img_resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    hash_value = fdns_hash(pil_img)
    hist_value = compute_hsv_histogram(img_resized)

    logger.info(f"Generated hash: {hash_value}")
    logger.debug(f"HSV histogram sample: {hist_value[:5]}...")

    _, jpeg_data = cv2.imencode('.jpg', img_resized)
    resized_bytes = jpeg_data.tobytes()
    _, png_data = cv2.imencode('.png', img_resized)
    base64_img = base64.b64encode(png_data.tobytes()).decode('utf-8')

    return hash_value, base64_img, resized_bytes, hist_value

def call_grok_api(image_bytes):
    """
    Call Grok API with image and prompt. 
    Returns dict if parse success, else empty dict.
    """
    logger.info("Calling Grok API")
    # Convert image to base64 for API
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    # Read model config and prompt from environment variables
    grok_model = os.environ.get('GROK_MODEL', 'grok-4-0709')
    grok_temperature = float(os.environ.get('GROK_TEMPERATURE', 0.5))
    grok_max_tokens = int(os.environ.get('GROK_MAX_TOKENS', 1024))
    grok_prompt = os.environ.get('GROK_PROMPT',
        '只要用日文回傳以下 JSON 格式，不要有解釋，也不要有其他文字：'
        '{"source_work":"", "character":"", "purchase_method":"", "suggested_price":"", "auction_description":""}'
    )
    payload = {
        "model": grok_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": grok_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": grok_temperature,
        "max_tokens": grok_max_tokens
    }
    headers = {
        'Authorization': f'Bearer {GROK_API_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        logger.debug(f"POST {GROK_API_URL}, payload size={len(json.dumps(payload))} bytes")
        response = requests.post(GROK_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        logger.info("Grok API response: %s", result)
        content = result['choices'][0]['message']['content']
        logger.debug(f"Grok raw content: {content}")
        try:
            parsed_result = json.loads(content)
            logger.info("Parsed Grok JSON successfully")
        except json.JSONDecodeError:
            logger.error("Failed to parse Grok API content as JSON")
            parsed_result = {}
        return parsed_result
    except Exception as e:
        logger.error(f"Grok API request failed: {e}")
        return None

@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html')

@app.route('/preprocess-image', methods=['POST'])
def preprocess_image_api():
    """Preprocess image endpoint: returns cropped badge and hash."""
    if 'image' not in request.files:
        logger.error("No image provided for preprocessing.")
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    try:
        image_hash, base64_img, _, _ = preprocess_image(image_file)
        logger.info(f"Preprocess success. Hash: {image_hash}")
        return jsonify({'processed_image': f'data:image/png;base64,{base64_img}', 'image_hash': image_hash})
    except Exception as e:
        logger.error(f"Preprocess error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/identify-badge', methods=['POST'])
def identify_badge():
    """Main endpoint: identify badge, try DB, fallback to Grok API, write result."""
    if 'image' not in request.files:
        logger.error("No image provided for identification.")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_hash, _, resized_bytes, input_hist = preprocess_image(image_file)
    logger.info(f"Computed hash for input image: {image_hash}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM badges')
    rows = cursor.fetchall()
    logger.debug(f"Loaded {len(rows)} rows from DB for comparison.")

    # Distance functions
    def hamming_dist(h1, h2):
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    def compare_hist(h1, h2):
        h1 = np.array(h1, dtype=np.float32)
        h2 = np.array(h2, dtype=np.float32)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    best_match = None
    best_score = -1

    for row in rows:
        stored_hash = row[1]
        if len(stored_hash) != len(image_hash):
            logger.warning(f"Skipping DB ID {row[0]} due to hash length mismatch")
            continue
        try:
            dist = hamming_dist(image_hash, stored_hash)
        except ValueError as e:
            logger.error(f"Hash comparison error with DB ID {row[0]}: {e}")
            continue
        logger.info(f"Comparing with DB ID {row[0]} → Hash dist: {dist}")
        color_score = 0
        color_hist_str = row[7]
        if color_hist_str:
            db_hist = json.loads(color_hist_str)
            color_score = compare_hist(input_hist, db_hist)
            logger.info(f"Color correlation with ID {row[0]}: {color_score:.4f}")
        match = (
            (dist <= COLOR_SCORE_DIST_THRESHOLD_1 and color_score >= COLOR_SCORE_CORR_THRESHOLD_1) or
            (dist <= COLOR_SCORE_DIST_THRESHOLD_2 and color_score >= COLOR_SCORE_CORR_THRESHOLD_2)
        )
        logger.debug(f"Match flag for DB ID {row[0]}: {match}")
        if match and (best_match is None or dist < hamming_dist(image_hash, best_match[1])):
            best_match = row
            best_score = color_score

    if best_match:
        logger.info(f"✅ Match found: ID={best_match[0]} | score={best_score}")
        conn.close()
        return jsonify({
            'image_hash': image_hash,
            'source_work': best_match[2],
            'character': best_match[3],
            'purchase_method': best_match[4],
            'suggested_price': best_match[5],
            'auction_description': best_match[6],
            'matched': True
        })

    # === If not found in DB, call Grok API ===
    logger.info(f"❌ No match found → calling Grok API for image hash: {image_hash}")
    grok_result = call_grok_api(resized_bytes)
    logger.debug(f"Grok result: {grok_result}")

    # Parse Grok API result (fallback to [unknown] if missing)
    source_work = '[unknown]'
    character = '[unknown]'
    purchase_method = '[unknown]'
    suggested_price = '[unknown]'
    auction_description = '[unknown]'

    if grok_result:
        source_work = grok_result.get('source_work', '[unknown]')
        character = grok_result.get('character', '[unknown]')
        purchase_method = grok_result.get('purchase_method', '[unknown]')
        suggested_price = grok_result.get('suggested_price', '[unknown]')
        auction_description = grok_result.get('auction_description', '[unknown]')
        logger.info(f"Inserting Grok result into DB for hash {image_hash}.")
        cursor.execute('''
            INSERT OR REPLACE INTO badges (image_hash, source_work, character, purchase_method, suggested_price, auction_description, color_hist)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (image_hash, source_work, character, purchase_method, suggested_price, auction_description, json.dumps(input_hist)))
        conn.commit()
    else:
        logger.warning("Grok API did not return a valid result. Inserting [unknown].")

    conn.close()

    return jsonify({
        'image_hash': image_hash,
        'source_work': source_work,
        'character': character,
        'purchase_method': purchase_method,
        'suggested_price': suggested_price,
        'auction_description': auction_description,
        'matched': False
    })

if __name__ == '__main__':
    logger.info("Starting Flask server ...")
    app.run(debug=True)
