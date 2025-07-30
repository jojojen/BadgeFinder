# BadgeFinder

> **Version:** 2025-07-30  
> **Author:** Your Name

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [System Architecture](#system-architecture)  
3. [Execution Flow](#execution-flow)  
4. [Code Layout](#code-layout)  
5. [Environment Variables](#environment-variables)  
6. [Database Schema](#database-schema)  
7. [REST API](#rest-api)  
8. [Deployment & Startup](#deployment--startup)  
9. [FAQ](#faq)

---

## Project Overview
**BadgeFinder** is a Flask web service that **detects, crops, and identifies** anime/game badge images.

* **FDNS perceptual hash + HSV color histogram** prevent duplicates by feature matching.  
* **xAI Grok API** supplies metadata when no database match is found.  
* **PostgreSQL** stores hashes, color histograms, and Grok-generated fields.

---

## System Architecture
    Browser ──► Flask (app.py)
                    │
                    ├─► /preprocess-image
                    │        ▲
                    │        └─ utils.image_utils.*
                    │             ├─ resize_if_large
                    │             ├─ fdns_hash
                    │             ├─ compute_hsv_histogram
                    │             ├─ preprocess_image
                    │             ├─ hamming_dist
                    │             └─ compare_hist
                    │
                    └─► /identify-badge
                             │ (fallback)
                             ▼
                        xAI Grok API

---

## Execution Flow
1. **Front-end** (`templates/index.html`)  
   * Upload image (HEIC → PNG auto-conversion if needed).  
   * `POST /preprocess-image` → cropped preview + `image_hash`.  
   * `POST /identify-badge` → DB lookup → Grok fallback.

2. **Pre-processing** (`utils.image_utils.preprocess_image`)  
   1. Decode → OpenCV BGR  
   2. Optional down-scale (`resize_if_large`)  
   3. Hough-circle detection → crop badge  
   4. Resize to **256 × 256**  
   5. Generate **FDNS hash** (`fdns_hash`)  
   6. Generate **HSV histogram** (`compute_hsv_histogram`)  
   7. Return `(hash, base64_png, resized_bytes_jpeg, histogram_list)`

3. **Identification** (`/identify-badge`)  
   * Compare with every row in `badges` table:  

        Rule A  Hamming ≤ 14 AND color corr ≥ 0.75  
        Rule B  Hamming ≤ 30 AND color corr ≥ 0.92  

   * **Match** → return DB record.  
   * **No match** → call Grok API, insert result into DB, return to client.

---

## Code Layout
| Path | Purpose | Notes |
|------|---------|-------|
| `app.py` | Flask server, DB access, REST endpoints | All image logic lives in `utils`. |
| `utils/image_utils.py` | Image utilities (pure functions) | Easy to unit-test. |
| `templates/index.html` | Single-page front-end | Handles HEIC → PNG, AJAX uploads. |
| `README.md` | Project documentation | This file. |

> `utils/__init__.py` is optional (PEP 420 implicit namespace).

---

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (`postgres://user:pass@host:port/db`) |
| `GROK_API_URL` | `https://api.x.ai/v1/chat/completions` | Grok endpoint |
| `GROK_API_KEY` | — | Grok API token |
| `GROK_MODEL` | `grok-4-0709` | Model name |
| `GROK_TEMPERATURE` | `0.5` | Sampling temperature |
| `GROK_MAX_TOKENS` | `1024` | Response length |
| `COLOR_SCORE_DIST_THRESHOLD_1` | `14` | Rule A Hamming threshold |
| `COLOR_SCORE_CORR_THRESHOLD_1` | `0.75` | Rule A correlation threshold |
| `COLOR_SCORE_DIST_THRESHOLD_2` | `30` | Rule B Hamming threshold |
| `COLOR_SCORE_CORR_THRESHOLD_2` | `0.92` | Rule B correlation threshold |
| `GROK_API_TIMEOUT` | `15` | Seconds before HTTP timeout |

---

## Database Schema
    CREATE TABLE badges (
        id                  SERIAL PRIMARY KEY,
        image_hash          TEXT UNIQUE,  -- FDNS hash
        source_work         TEXT,
        character           TEXT,
        purchase_method     TEXT,
        suggested_price     TEXT,
        auction_description TEXT,
        color_hist          TEXT          -- JSON-encoded HSV histogram
    );

---

## REST API
| Method & Path | Form Field(s) | Description |
|---------------|--------------|-------------|
| `POST /preprocess-image` | `image=<file>` | Returns cropped badge (`data:image/png;base64,…`) and `image_hash`. |
| `POST /identify-badge` | `image=<file>` | DB lookup → Grok fallback; returns metadata + `matched` flag. |

**Sample response (`/identify-badge`):**
```json
{
  "image_hash": "3f87fe7f...",
  "source_work": "Fate/stay night",
  "character": "Saber",
  "purchase_method": "Booth A-11, Comiket 104",
  "suggested_price": "¥600",
  "auction_description": "Limited edition metallic badge",
  "matched": true
}
