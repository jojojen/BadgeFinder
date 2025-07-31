# BadgeFinder Project Overview (featureRefactor branch)

## Project structure

### Visual overview

The directory layout of the `BadgeFinder` project is simple. The main directory contains the Python source files and any front‑end templates. Here is an illustrative tree using icons:

* 📁 **BadgeFinder**

  * 📄 **app.py** – main Flask application
  * 📄 **utils.py** – image processing and hashing helpers
  * 📁 **templates** – front‑end HTML (e.g., index.html)

    * 📄 **index.html**

In this tree, `app.py` houses the web server logic and calls into `utils.py` for image processing. The `templates` folder contains HTML files served by Flask, such as `index.html`, which renders the upload form shown at `/`. Additional static or template files (if any) would live alongside `index.html` in this folder.

* **app.py** – main Flask application. It loads environment variables, sets up logging and a PostgreSQL connection, initializes the `badges` table if it does not exist, and defines the web routes and the function that calls the Grok API. Core processing functions are imported from `utils.py`.
* **utils.py** – contains helper functions for image processing, including:

  * `resize_if_large(img, max_side=1024)`: checks the maximum dimension of the input image and resizes it using OpenCV’s `cv2.resize` with `INTER_AREA` interpolation if the image is larger than `max_side`.
  * `fdns_hash(image, hash_size=12, block_size=16)`: computes a perceptual hash. The function converts the image to grayscale, partitions it into blocks, applies a two‑dimensional Discrete Cosine Transform (DCT) to each block, compares direct‑current (DC) and average absolute coefficient (AC) values across all blocks and builds a bit string that is encoded as a hexadecimal hash.
  * `compute_hsv_histogram(image_bgr, hist_size=32)`: converts the image from BGR to HSV and computes a normalised histogram of the hue channel with 32 bins using `cv2.calcHist`.
  * `preprocess_image(image_file)`: handles file upload input, decodes the JPEG data into an OpenCV array, calls `resize_if_large`, detects a circular badge using the Hough Circle transform (`cv2.HoughCircles` with the gradient method), crops the badge, resizes it to 256×256 pixels with Lanczos interpolation, computes the FDNS hash and HSV histogram, and returns the hash and encoded images.

## Execution logic

1. **Initialization** – `app.py` loads environment variables using `python-dotenv` and configures logging. It creates a PostgreSQL connection string from `DATABASE_URL` and ensures that the `badges` table exists in the database.
2. **Importing image helpers** – The heavy image processing functions are imported from `utils.py`.
3. **Grok API function** – `call_grok_api` builds a request payload containing the image (encoded in base64) and a Japanese prompt, and sends it to the external Grok API using the `requests` library. The response JSON is parsed and returned.
4. **Flask routes**:

   * `/` simply renders `index.html`.
   * `/preprocess-image` expects a form with an image. It calls `preprocess_image` from `utils.py` to resize, crop, hash and histogram the badge. The endpoint returns the base64‑encoded processed image and its hash in JSON.
   * `/identify-badge` calls `preprocess_image` to obtain the hash and colour histogram, then queries all entries from the `badges` table. It computes the Hamming distance between the uploaded image hash and each stored hash and compares the HSV histograms using `cv2.compareHist`. Two threshold pairs control when an entry is considered a match (`COLOR_SCORE_DIST_THRESHOLD_1`/`COLOR_SCORE_CORR_THRESHOLD_1` and `COLOR_SCORE_DIST_THRESHOLD_2`/`COLOR_SCORE_CORR_THRESHOLD_2`). If a match is found, the metadata is returned; otherwise the image is sent to the Grok API to generate metadata, which is then inserted into the database and returned to the caller.

## Image processing algorithms

* **Resizing large images** – The `resize_if_large` function calculates the ratio between the maximum allowed side length and the largest of the image’s width and height. It applies this scale to both dimensions and calls OpenCV’s `cv2.resize` with `INTER_AREA` interpolation for downscaling, preserving aspect ratio.
* **Circular badge detection** – Inside `preprocess_image`, the grayscale image is blurred with a median filter and circles are detected using OpenCV’s Hough Circle transform (`cv2.HoughCircles`) configured with the gradient method, an accumulator resolution of 1.2, minimum distance of 100 pixels between circles, and specific thresholds for edge detection and circle accumulation. Only the first detected circle is used to define the crop region.
* **Perceptual hashing (FDNS)** – The `fdns_hash` function implements a block‑based perceptual hash. The image is converted to grayscale and resized so each block is `block_size×block_size` pixels. For each block, a 2‑D DCT is computed. The direct‑current coefficient (DC) and mean absolute value of the remaining coefficients (AC) are recorded. After processing all blocks, the median DC and AC are computed and used as thresholds: bits are set for blocks where either DC or AC exceeds the corresponding median. The resulting bit sequence is packed into a hexadecimal string.
* **Colour histogram** – `compute_hsv_histogram` converts the BGR image to HSV and computes a histogram of the hue channel using 32 bins spanning 0–180°, normalises the histogram with `cv2.normalize` and flattens it to a 1‑D array.
* **Hamming distance and histogram correlation** – `identify_badge` defines a simple Hamming distance function to compare the hexadecimal hash strings and uses OpenCV’s `cv2.compareHist` with `HISTCMP_CORREL` to measure correlation between HSV histograms. These metrics are compared against configured thresholds to decide whether two badges match.

## How to start the project

1. **Install dependencies and set up the environment** – The README suggests creating a dedicated environment, for example by activating a Conda environment (`conda activate py312`) or sourcing a virtual environment (`source env/bin/activate`), and then installing dependencies with `pip install -r requirements.txt`. Ensure Python 3 and the required packages (Flask, requests, python‑dotenv, Pillow, numpy, opencv‑python, scipy, psycopg2) are installed in your environment.
2. **Database setup** – Create a PostgreSQL database and set the `DATABASE_URL` environment variable to the appropriate connection string. The application will create a `badges` table on startup if it does not already exist.
3. **Environment variables** – Optionally set `GROK_API_URL`, `GROK_API_KEY`, `GROK_MODEL`, `GROK_TEMPERATURE`, `GROK_MAX_TOKENS`, `COLOR_SCORE_DIST_THRESHOLD_1`, `COLOR_SCORE_CORR_THRESHOLD_1`, `COLOR_SCORE_DIST_THRESHOLD_2`, and `COLOR_SCORE_CORR_THRESHOLD_2` to tune API behaviour and matching thresholds. Defaults are provided in `app.py`.
4. **Running the server** – From the project root, execute `python app.py`. This starts the Flask development server (`app.run(debug=True)`) on the default port (5000). Alternatively, the README shows you can run the application with Gunicorn using `gunicorn app:app --bind 0.0.0.0:5000 --workers 2` or use Flask’s built‑in runner by setting `export FLASK_APP=app.py` and then running `flask run`. After the server starts, open a browser at `http://127.0.0.1:5000/` to access the upload form. For production deployments, run the Flask application with a WSGI server such as Gunicorn and set `FLASK_ENV=production`.
5. **Using the API** – Submit a POST request to `/preprocess-image` with a form field named `image` to get the processed badge and hash. Submit a POST to `/identify-badge` to identify a badge; the endpoint will return any matching metadata from the database or call the external Grok API if no match is found.


## venv setting
conda activate py312
source env/bin/activate
pip install -r requirements.txt

### exec method 1
python app.py
### exec method 2
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
### exec method 3
export FLASK_APP=app.py
flask run

### check at
http://127.0.0.1:5000/