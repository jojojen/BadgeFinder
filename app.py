from flask import Flask, request, jsonify, render_template
import os
import json
from PIL import Image
import numpy as np
import base64
import cv2
import logging
from scipy.fftpack import dct
import requests
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse
import time

from utils import preprocess_image, resize_if_large, fdns_hash, compute_hsv_histogram

load_dotenv()

# ------------------------------------------------------------------
# Flask & logging setup
# ------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Database config (PostgreSQL only)
# ------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. PostgreSQL connection string is required.")

def log_db_info():
    """Log non-secret DB connection info once at startup."""
    p = urlparse(DATABASE_URL)
    logger.info(
        "DB TYPE: PostgreSQL | host=%s | port=%s | database=%s",
        p.hostname,
        p.port,
        p.path.lstrip("/")
    )
log_db_info()

def get_conn():
    """Return a PostgreSQL connection."""
    logger.debug("Opening PostgreSQL connection")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Create table if it doesn't exist."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS badges (
            id              SERIAL PRIMARY KEY,
            image_hash      TEXT UNIQUE,
            source_work     TEXT,
            character       TEXT,
            purchase_method TEXT,
            suggested_price TEXT,
            auction_description TEXT,
            color_hist      TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    logger.info("Database initialized.")

init_db()

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
GROK_API_URL  = os.environ.get("GROK_API_URL", "https://api.x.ai/v1/chat/completions")
GROK_API_KEY  = os.environ.get("GROK_API_KEY", "your_api_key_here")

COLOR_SCORE_DIST_THRESHOLD_1 = float(os.environ.get("COLOR_SCORE_DIST_THRESHOLD_1", 14))
COLOR_SCORE_CORR_THRESHOLD_1 = float(os.environ.get("COLOR_SCORE_CORR_THRESHOLD_1", 0.75))
COLOR_SCORE_DIST_THRESHOLD_2 = float(os.environ.get("COLOR_SCORE_DIST_THRESHOLD_2", 30))
COLOR_SCORE_CORR_THRESHOLD_2 = float(os.environ.get("COLOR_SCORE_CORR_THRESHOLD_2", 0.92))
GROK_API_TIMEOUT = int(os.environ.get("GROK_API_TIMEOUT", 15))

# === 一次性列出與超時相關的環境設定
logger.info("GROK timeout=%ss | payload max_tokens=%s | API=%s", GROK_API_TIMEOUT, os.environ.get("GROK_MAX_TOKENS", 1024), GROK_API_URL)

# ------------------------------------------------------------------
# Image hashing & helpers
# ------------------------------------------------------------------
# The implementations of resize_if_large, fdns_hash, compute_hsv_histogram and
# preprocess_image have been moved to utils.py. They are imported above.

# ------------------------------------------------------------------
# Grok API
# ------------------------------------------------------------------
def call_grok_api(image_bytes):
    """Call Grok API with image and prompt; parse response JSON."""
    start_grok = time.time()                       
    logger.info("Calling Grok API")
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    grok_model       = os.environ.get("GROK_MODEL", "grok-4-0709")
    grok_temperature = float(os.environ.get("GROK_TEMPERATURE", 0.5))
    grok_max_tokens  = int(os.environ.get("GROK_MAX_TOKENS", 1024))
    grok_prompt      = os.environ.get(
        "GROK_PROMPT",
        '只要用日文回傳以下 JSON 格式，不要有解釋，也不要有其他文字：'
        '{"source_work":"", "character":"", "purchase_method":"", "suggested_price":"", "auction_description":""}',
    )
    payload = {
        "model": grok_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": grok_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        "temperature": grok_temperature,
        "max_tokens": grok_max_tokens,
    }
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    try:
        logger.debug("POST %s, payload size=%s bytes", GROK_API_URL, len(json.dumps(payload)))
        response = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=GROK_API_TIMEOUT)
        logger.debug("Grok HTTP status: %s | time %.2fs", response.status_code, time.time()-start_grok)
        response.raise_for_status()
        result = response.json()
        logger.info("Grok API response JSON parsed in %.2fs", time.time()-start_grok)
        content = result["choices"][0]["message"]["content"]
        logger.debug("Grok full JSON preview=%s…", json.dumps(result)[:400])
        logger.debug("Grok raw content: %s", content)
        try:
            parsed_result = json.loads(content)
            logger.info("Parsed Grok JSON successfully (%.2fs)", time.time()-start_grok)
        except json.JSONDecodeError:
            logger.error("Failed to parse Grok API content as JSON (%.2fs)", time.time()-start_grok)
            parsed_result = {}
        return parsed_result
    except Exception as e:
        logger.error("Grok API request failed after %.2fs: %s", time.time()-start_grok, e)
        return None

# ------------------------------------------------------------------
# Flask routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    """Home page with upload form."""
    return render_template("index.html")

@app.route("/preprocess-image", methods=["POST"])
def preprocess_image_api():
    """Preprocess image endpoint: returns cropped badge and hash."""
    if "image" not in request.files:
        logger.error("No image provided for preprocessing.")
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files["image"]
    try:
        image_hash, base64_img, _, _ = preprocess_image(image_file)
        logger.info("Preprocess success. Hash: %s", image_hash)
        return jsonify({"processed_image": f"data:image/png;base64,{base64_img}", "image_hash": image_hash})
    except Exception as e:
        logger.error("Preprocess error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/identify-badge", methods=["POST"])
def identify_badge():
    """Identify badge: try DB, fallback to Grok API, write result."""
    request_start = time.time()                              
    if "image" not in request.files:
        logger.error("No image provided for identification.")
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files["image"]
    image_hash, _, resized_bytes, input_hist = preprocess_image(image_file)
    logger.info("Computed hash for input image: %s", image_hash)
    conn = get_conn()
    cur = conn.cursor()
    db_fetch_start = time.time()                             
    cur.execute("SELECT * FROM badges")
    rows = cur.fetchall()
    logger.debug("Loaded %s rows from DB for comparison in %.2fs.", len(rows), time.time()-db_fetch_start)  
    def hamming_dist(h1, h2):
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    def compare_hist(h1, h2):
        h1 = np.array(h1, dtype=np.float32)
        h2 = np.array(h2, dtype=np.float32)
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    best_match = None
    best_score = -1
    loop_start = time.time()                                
    for row in rows:
        stored_hash = row[1]
        if len(stored_hash) != len(image_hash):
            logger.warning("Skipping DB ID %s due to hash length mismatch", row[0])
            continue
        try:
            dist = hamming_dist(image_hash, stored_hash)
        except ValueError as e:
            logger.error("Hash comparison error with DB ID %s: %s", row[0], e)
            continue
        logger.info("Comparing with DB ID %s → Hash dist: %s", row[0], dist)
        color_score = 0
        color_hist_str = row[7]
        if color_hist_str:
            db_hist = json.loads(color_hist_str)
            color_score = compare_hist(input_hist, db_hist)
            logger.info("Color correlation with ID %s: %.4f", row[0], color_score)
        match = (
            (dist <= COLOR_SCORE_DIST_THRESHOLD_1 and color_score >= COLOR_SCORE_CORR_THRESHOLD_1) or
            (dist <= COLOR_SCORE_DIST_THRESHOLD_2 and color_score >= COLOR_SCORE_CORR_THRESHOLD_2)
        )
        logger.debug("Match flag for DB ID %s: %s", row[0], match)
        if match and (best_match is None or dist < hamming_dist(image_hash, best_match[1])):
            best_match = row
            best_score = color_score
    logger.debug("DB loop finished in %.2fs", time.time()-loop_start)  
    if best_match:
        logger.info("✅ Match found: ID=%s | score=%s", best_match[0], best_score)
        conn.close()
        logger.debug("Total identify_badge time %.2fs (from start)", time.time()-request_start)  
        return jsonify(
            {
                "image_hash": image_hash,
                "source_work": best_match[2],
                "character": best_match[3],
                "purchase_method": best_match[4],
                "suggested_price": best_match[5],
                "auction_description": best_match[6],
                "matched": True,
            }
        )
    # Not found → call Grok API
    logger.info("❌ No match found → calling Grok API for image hash: %s", image_hash)
    grok_result = call_grok_api(resized_bytes)
    logger.debug("Grok result: %s", grok_result)
    # Fallback values
    source_work = "[unknown]"
    character = "[unknown]"
    purchase_method = "[unknown]"
    suggested_price = "[unknown]"
    auction_description = "[unknown]"
    if grok_result:
        source_work = grok_result.get("source_work", "[unknown]")
        character = grok_result.get("character", "[unknown]")
        purchase_method = grok_result.get("purchase_method", "[unknown]")
        suggested_price = grok_result.get("suggested_price", "[unknown]")
        auction_description = grok_result.get("auction_description", "[unknown]")
        logger.info("Inserting Grok result into DB for hash %s.", image_hash)
        cur.execute(
            """
            INSERT INTO badges (
                image_hash, source_work, character, purchase_method,
                suggested_price, auction_description, color_hist
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (image_hash) DO UPDATE SET
                source_work         = EXCLUDED.source_work,
                character           = EXCLUDED.character,
                purchase_method     = EXCLUDED.purchase_method,
                suggested_price     = EXCLUDED.suggested_price,
                auction_description = EXCLUDED.auction_description,
                color_hist          = EXCLUDED.color_hist
            """,
            (
                image_hash,
                source_work,
                character,
                purchase_method,
                suggested_price,
                auction_description,
                json.dumps(input_hist),
            ),
        )
        conn.commit()
    else:
        logger.warning("Grok API did not return a valid result. Inserting [unknown].")
    conn.close()
    logger.debug("Total identify_badge time %.2fs (from start)", time.time()-request_start)  
    return jsonify(
        {
            "image_hash": image_hash,
            "source_work": source_work,
            "character": character,
            "purchase_method": purchase_method,
            "suggested_price": suggested_price,
            "auction_description": auction_description,
            "matched": False,
        }
    )

# ------------------------------------------------------------------
# Local testing entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Flask server ...")
    app.run(debug=True)