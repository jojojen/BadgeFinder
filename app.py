from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import requests
from PIL import Image
import io
import numpy as np
import base64
import cv2  # OpenCV for circle detection and cropping
import logging  # For debug logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database setup
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
            auction_description TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def preprocess_image(image_file):
    logger.debug("Starting image preprocessing")

    # Read image bytes from uploaded file
    image_file.seek(0)
    img_bytes = image_file.read()

    # Decode bytes into OpenCV image format (BGR)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    logger.debug(f"Image decoded: shape {img.shape}")

    # Convert to grayscale and blur for circle detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=600
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]
        logger.debug(f"Circle found: center=({x},{y}), radius={r}")

        # Create circular mask and apply it
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        masked = cv2.bitwise_and(img, mask)

        # Crop square around the circle
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, img.shape[1])
        y2 = min(y + r, img.shape[0])
        cropped = masked[y1:y2, x1:x2]
    else:
        raise ValueError("No circular badge found")

    # Resize to 256x256 for consistency
    FIXED_SIZE = 256
    img_resized = cv2.resize(cropped, (FIXED_SIZE, FIXED_SIZE), interpolation=cv2.INTER_LANCZOS4)

    # Generate perceptual hash
    HASH_SIZE = 32
    img_hash_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hash_img = cv2.resize(img_hash_gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_LANCZOS4)
    pixels = hash_img.flatten()
    avg = pixels.mean()
    diff = pixels > avg
    hash_value = hex(sum(2 ** i if bit else 0 for i, bit in enumerate(diff)))[2:].zfill(HASH_SIZE * HASH_SIZE // 4)

    # Encode for Grok (JPEG) and frontend (PNG Base64)
    _, jpeg_data = cv2.imencode('.jpg', img_resized)
    resized_bytes = jpeg_data.tobytes()
    _, png_data = cv2.imencode('.png', img_resized)
    base64_img = base64.b64encode(png_data.tobytes()).decode('utf-8')

    logger.debug("Preprocessing complete with circular crop and hash generated.")
    return hash_value, base64_img, resized_bytes

# Grok API call (placeholder; adjust per https://x.ai/api)
GROK_API_URL = 'https://api.x.ai/v1/analyze'  # Check official docs
GROK_API_KEY = os.getenv('GROK_API_KEY')

def call_grok_api(resized_data):
    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not set")

    files = {'image': ('badge.jpg', resized_data, 'image/jpeg')}
    headers = {'Authorization': f'Bearer {GROK_API_KEY}'}
    prompt = """
    Analyze the badge image and provide in JSON:
    {"source_work": "...", "character": "...", "purchase_method": "...", "suggested_price": "...", "auction_description": "..."}
    """
    data = {'prompt': prompt}

    response = requests.post(GROK_API_URL, headers=headers, data=data, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.text}")

@app.route('/')
def index():
    return render_template('index.html')  # Serve frontend

@app.route('/preprocess-image', methods=['POST'])
def preprocess_image_api():
    logger.debug("Received request to /preprocess-image")
    if 'image' not in request.files:
        logger.warning("No image in request.files")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    try:
        _, base64_img, _ = preprocess_image(image_file)
        return jsonify({'processed_image': f'data:image/png;base64,{base64_img}'})
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/identify-badge', methods=['POST'])
def identify_badge():
    logger.debug("Received request to /identify-badge")
    if 'image' not in request.files:
        logger.warning("No image in request.files")
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_hash, _, resized_bytes = preprocess_image(image_file)  # Get hash and resized bytes

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM badges WHERE image_hash = ?', (image_hash,))
    row = cursor.fetchone()

    if row:
        response = {
            'source_work': row[2],
            'character': row[3],
            'purchase_method': row[4],
            'suggested_price': row[5],
            'auction_description': row[6]
        }
        conn.close()
        return jsonify(response)

    try:
        api_response = call_grok_api(resized_bytes)  # Use preprocessed resized image for API

        cursor.execute('''
            INSERT INTO badges (image_hash, source_work, character, purchase_method, suggested_price, auction_description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_hash, api_response['source_work'], api_response['character'],
              api_response['purchase_method'], api_response['suggested_price'],
              api_response['auction_description']))
        conn.commit()
        conn.close()

        return jsonify(api_response)
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
