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

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DB_PATH = 'badges.db'

def init_db():
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

init_db()

def fdns_hash(image: Image.Image, hash_size=12, block_size=16) -> str:
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
    return hex_str

def compute_hsv_histogram(image_bgr, hist_size=32):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [hist_size], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    return h_hist.tolist()

def preprocess_image(image_file):
    logger.debug("Starting image preprocessing")
    image_file.seek(0)
    img_bytes = image_file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    logger.debug(f"Image decoded: shape {img.shape}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=100, maxRadius=600)

    if circles is None:
        raise ValueError("No circular badge found")

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = circles[0]
    logger.debug(f"Circle found: center=({x},{y}), radius={r}")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess-image', methods=['POST'])
def preprocess_image_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    try:
        image_hash, base64_img, _, _ = preprocess_image(image_file)
        return jsonify({'processed_image': f'data:image/png;base64,{base64_img}', 'image_hash': image_hash})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/identify-badge', methods=['POST'])
def identify_badge():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_hash, _, _, input_hist = preprocess_image(image_file)
    logger.info(f"Computed hash for input image: {image_hash}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM badges')
    rows = cursor.fetchall()

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
            (dist <= 14 and color_score >= 0.75) or
            (dist <= 30 and color_score >= 0.92)
        )

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

    logger.info(f"❌ No match found → inserting hash: {image_hash}")
    cursor.execute('''
        INSERT INTO badges (image_hash, source_work, character, purchase_method, suggested_price, auction_description, color_hist)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (image_hash, '[unknown]', '[unknown]', '[unknown]', '[unknown]', '[unknown]', json.dumps(input_hist)))
    conn.commit()
    conn.close()

    return jsonify({
        'image_hash': image_hash,
        'source_work': '[unknown]',
        'character': '[unknown]',
        'purchase_method': '[unknown]',
        'suggested_price': '[unknown]',
        'auction_description': '[unknown]',
        'matched': False
    })

if __name__ == '__main__':
    app.run(debug=True)
