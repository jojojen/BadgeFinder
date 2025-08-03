import os
import json
import base64
import time
import logging

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import numpy as np
import cv2
import requests
from requests.exceptions import Timeout  # import Timeout exception for handling request timeouts

from utils import preprocess_image, resize_if_large, fdns_hash, compute_hsv_histogram, upload_image
from models import Badge, FailedGrok, get_session_factory

# Load environment variables
load_dotenv()

# ------------------------------------------------------------------
# Flask & logging setup
# ------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# SQLAlchemy session factory
# ------------------------------------------------------------------
SessionLocal = get_session_factory()

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

logger.info(
    "GROK timeout=%ss | payload max_tokens=%s | API=%s",
    GROK_API_TIMEOUT,
    os.environ.get("GROK_MAX_TOKENS", 1024),
    GROK_API_URL,
)

# ------------------------------------------------------------------
# Grok API helper
# ------------------------------------------------------------------
def call_grok_api(image_bytes):
    """Call Grok API with image and prompt; parse response JSON.

    Returns a tuple ``(parsed_result, raw_content, preview_json)``. If
    parsing fails, ``parsed_result`` will be ``None``.
    """
    start_grok = time.time()
    logger.info("Calling Grok API")
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    grok_model       = os.environ.get("GROK_MODEL", "grok-4-0709")
    grok_temperature = float(os.environ.get("GROK_TEMPERATURE", 0.5))
    grok_max_tokens  = int(os.environ.get("GROK_MAX_TOKENS", 1024))
    grok_prompt      = os.environ.get(
        "GROK_PROMPT",
        '只要用日文回傳以下 JSON 格式，不要有解釋，也不要有其他文字：'
        '{"source_work":"", "character":"", "acquisition_difficulty":"", "auction_description":""}',
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
        logger.debug("Grok HTTP status: %s | time %.2fs", response.status_code, time.time() - start_grok)
    except Exception as e:
        logger.error("Grok API request failed after %.2fs: %s", time.time() - start_grok, e)
        return None, None, None
    try:
        response.raise_for_status()
        result = response.json()
        logger.info("Grok API response JSON parsed in %.2fs", time.time() - start_grok)
        preview = json.dumps(result)[:400]
        logger.debug("Grok full JSON preview=%s…", preview)
        content = result["choices"][0]["message"].get("content", "")
        logger.debug("Grok raw content: %s", content)
        try:
            parsed_result = json.loads(content)
            logger.info("Parsed Grok JSON successfully (%.2fs)", time.time() - start_grok)
            return parsed_result, content, preview
        except json.JSONDecodeError:
            logger.error("Failed to parse Grok API content as JSON (%.2fs)", time.time() - start_grok)
            return None, content, preview
    except Exception as e:
        logger.error("Grok API response processing failed after %.2fs: %s", time.time() - start_grok, e)
        return None, None, None

# ------------------------------------------------------------------
# Flask routes
# ------------------------------------------------------------------
@app.route("/")
def index():
    """Home page with upload form."""
    return render_template("index.html")

@app.route("/preprocess-image", methods=["POST"])
def preprocess_image_api():
    """Preprocess image endpoint: returns cropped badge and hash with upload URL."""
    start_time = time.time()
    if "image" not in request.files:
        logger.error("No image provided for preprocessing.")
        logger.info("preprocess_image_api completed in %.2fs", time.time() - start_time)
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files["image"]
    try:
        image_hash, base64_img, resized_bytes, _ = preprocess_image(image_file)
        logger.info("Preprocess success. Hash: %s", image_hash)
        url = upload_image(resized_bytes)
        logger.info("preprocess_image_api completed in %.2fs", time.time() - start_time)
        return jsonify({"processed_image": f"data:image/png;base64,{base64_img}", "image_hash": image_hash, "url": url})
    except Exception as e:
        logger.error("Preprocess error: %s", e)
        logger.info("preprocess_image_api completed in %.2fs", time.time() - start_time)
        return jsonify({"error": str(e)}), 500

@app.route("/identify-badge", methods=["POST"])
def identify_badge():
    request_start = time.time()
    if "image" not in request.files:
        logger.error("No image provided for identification.")
        logger.info("identify_badge completed in %.2fs", time.time() - request_start)
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files["image"]
    # Preprocess the image and capture the base64 representation for potential storage
    image_hash, base64_img, resized_bytes, input_hist = preprocess_image(image_file)
    logger.info("Computed hash for input image: %s", image_hash)
    existing_url = request.form.get("url")
    if existing_url:
        url_uploaded = existing_url
    else:
        url_uploaded = upload_image(resized_bytes)
    # Start a new database session
    session = SessionLocal()
    try:
        # Load all badges and compute best match
        badges = session.query(Badge).all()
        def hamming_dist(h1, h2):
            return bin(int(h1, 16) ^ int(h2, 16)).count("1")
        def compare_hist(h1, h2):
            h1 = np.array(h1, dtype=np.float32)
            h2 = np.array(h2, dtype=np.float32)
            return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        best_match = None
        best_score = -1
        for row in badges:
            stored_hash = row.image_hash
            if len(stored_hash) != len(image_hash):
                logger.warning("Skipping DB ID %s due to hash length mismatch", row.id)
                continue
            try:
                dist = hamming_dist(image_hash, stored_hash)
            except ValueError as e:
                logger.error("Hash comparison error with DB ID %s: %s", row.id, e)
                continue
            logger.info("Comparing with DB ID %s → Hash dist: %s", row.id, dist)
            color_score = 0
            if row.color_hist:
                try:
                    db_hist = json.loads(row.color_hist)
                    color_score = compare_hist(input_hist, db_hist)
                    logger.info("Color correlation with ID %s: %.4f", row.id, color_score)
                except Exception as exc:
                    logger.warning("Failed to compare histogram for ID %s: %s", row.id, exc)
            match = (
                (dist <= COLOR_SCORE_DIST_THRESHOLD_1 and color_score >= COLOR_SCORE_CORR_THRESHOLD_1) or
                (dist <= COLOR_SCORE_DIST_THRESHOLD_2 and color_score >= COLOR_SCORE_CORR_THRESHOLD_2)
            )
            logger.debug("Match flag for DB ID %s: %s", row.id, match)
            if match and (best_match is None or dist < hamming_dist(image_hash, best_match.image_hash)):
                best_match = row
                best_score = color_score
        if best_match:
            logger.info("✅ Match found: ID=%s | score=%s", best_match.id, best_score)
            # Return the URL stored in DB
            db_url = best_match.url or None
            response = {
                "image_hash": image_hash,
                "source_work": best_match.source_work or "",
                "character": best_match.character or "",
                "acquisition_difficulty": best_match.acquisition_difficulty or "",
                "auction_description": best_match.auction_description or "",
                "matched": True,
                "url": db_url,
                "color_hist": input_hist,
            }
            logger.info("identify_badge completed in %.2fs", time.time() - request_start)
            return jsonify(response)
        # No match found → call Grok API
        logger.info("❌ No match found → calling Grok API for image hash: %s", image_hash)
        grok_result, grok_raw, grok_preview = call_grok_api(resized_bytes)
        if grok_result:
            source_work = grok_result.get("source_work", "")
            character = grok_result.get("character", "")
            acquisition_difficulty = grok_result.get("acquisition_difficulty", "")
            auction_description = grok_result.get("auction_description", "")
        else:
            source_work = ""
            character = ""
            acquisition_difficulty = ""
            auction_description = ""
            # Construct an error detail using the preview JSON if available. If not, fall back to raw content.
            if grok_preview:
                error_detail = f"Grok full JSON preview={grok_preview}"
            elif grok_raw:
                error_detail = f"length={len(grok_raw)} content={grok_raw[:200]}"
            else:
                error_detail = "No valid Grok response"
            # Record the failure in failed_grok table
            try:
                failed = FailedGrok(image_hash=image_hash, url=url_uploaded, error_detail=error_detail)
                session.add(failed)
                session.commit()
            except Exception as e:
                logger.error("Failed to insert failed_grok record for hash %s: %s", image_hash, e)
                session.rollback()
        # Upsert badge information
        try:
            badge = session.query(Badge).filter_by(image_hash=image_hash).first()
            if not badge:
                badge = Badge(image_hash=image_hash)
                session.add(badge)
            # Only overwrite fields if new data is non-empty
            if source_work:
                badge.source_work = source_work
            if character:
                badge.character = character
            if acquisition_difficulty:
                badge.acquisition_difficulty = acquisition_difficulty
            if auction_description:
                badge.auction_description = auction_description
            badge.color_hist = json.dumps(input_hist)
            badge.url = url_uploaded
            session.commit()
        except Exception as e:
            logger.error("Failed to insert badge for hash %s: %s", image_hash, e)
            session.rollback()
        response = {
            "image_hash": image_hash,
            "source_work": source_work,
            "character": character,
            "acquisition_difficulty": acquisition_difficulty,
            "auction_description": auction_description,
            "matched": False,
            "url": url_uploaded,
            "color_hist": input_hist,
        }
        logger.info("identify_badge completed in %.2fs", time.time() - request_start)
        return jsonify(response)
    finally:
        session.close()

# ------------------------------------------------------------------
# Feedback endpoint
# ------------------------------------------------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    """Accept user feedback to insert or update a badge entry in the database."""
    image_hash = request.form.get("image_hash")
    source_work = request.form.get("source_work", "[unknown]")
    character = request.form.get("character", "[unknown]")
    auction_description = request.form.get("auction_description", "[unknown]")
    # update the badge entry using user feedback
    session = SessionLocal()
    try:
        badge = session.query(Badge).filter_by(image_hash=image_hash).first()
        if not badge:
            # If no badge exists (should rarely happen), create a new one
            badge = Badge(image_hash=image_hash)
            session.add(badge)
        badge.source_work = source_work
        badge.character = character
        badge.auction_description = auction_description
        session.commit()
    except Exception as e:
        logger.error("Failed to insert feedback for hash %s: %s", image_hash, e)
        session.rollback()
        session.close()
        return jsonify({"error": "Database error"}), 500
    session.close()
    return jsonify({"status": "success"})

# ------------------------------------------------------------------
# Local testing entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Flask server ...")
    app.run(debug=True)
