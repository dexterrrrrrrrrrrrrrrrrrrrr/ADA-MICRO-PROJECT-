"""
=============================================================================
  LICENSE PLATE RECOGNITION SYSTEM (LPR)
  VTU - Analysis and Design of Algorithms (ADA) Micro Project
=============================================================================
  Subject   : Analysis and Design of Algorithms (ADA) — BCS401
  Institute : Sai Vidya Institute of Technology
  Language  : Python 3
  Libraries : OpenCV, Pytesseract, NumPy, imutils

  TEAM MEMBERS:
    1. Anurag Paul
    2. Alisha Pandit
    3. Md Zeeshan
    4. Srishti Mishra
=============================================================================

  ALGORITHMS USED:
    1. Canny Edge Detection Algorithm  - O(W x H)
    2. Contour Detection               - O(W x H)
    3. OCR (Tesseract)                 - O(n)
    4. Dictionary Search (Hash-based)  - O(1) average
    5. Bilateral Filtering             - O(W x H x d^2)

  WHERE: W = image width, H = image height, d = filter diameter, n = text length
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — IMPORTS
# Standard libraries and third-party packages required for the system
# ─────────────────────────────────────────────────────────────────────────────

import cv2               # OpenCV: image loading, processing, contour detection
import numpy as np       # NumPy: array/matrix operations on image pixel data
import pytesseract       # Pytesseract: Python wrapper for Tesseract OCR engine
import re                # re: Regular expressions for text cleaning
import sys               # sys: system-level operations (exit, args)
import os                # os: file path checks

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Tesseract OCR engine path
# Update this path if Tesseract is installed in a different location.
# Windows default: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Linux/macOS: usually already on PATH, so this line can be skipped.
# ─────────────────────────────────────────────────────────────────────────────

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — VEHICLE DATABASE
# A simple in-memory dictionary acting as the vehicle registry.
# Key   : Vehicle registration number (uppercase, no spaces)
# Value : Dictionary containing owner info and vehicle type
#
# Data Structure: Python Dictionary (Hash Map)
# Search Time Complexity: O(1) average, O(n) worst case
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_DATABASE = {
    "KA01AB1234": {"owner": "Rajesh Kumar",    "type": "Car   (Sedan)"},
    "KA02CD5678": {"owner": "Priya Sharma",    "type": "Motorcycle"},
    "MH12EF9012": {"owner": "Amit Patil",      "type": "Car   (SUV)"},
    "DL8CAS3456": {"owner": "Sneha Verma",     "type": "Car   (Hatchback)"},
    "TN09GH7890": {"owner": "Suresh Rajan",    "type": "Truck"},
    "KA03IJ2345": {"owner": "Meena Reddy",     "type": "Auto Rickshaw"},
    "UP16KL6789": {"owner": "Vikram Singh",    "type": "Bus"},
    "GJ01MN0123": {"owner": "Hetal Desai",     "type": "Car   (Sedan)"},
    "KA05OP4567": {"owner": "Kiran Nair",      "type": "Van"},
    "AP28QR8901": {"owner": "Lakshmi Devi",    "type": "Car   (SUV)"},
}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — BANNER & UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    """Print a decorative project banner on the console."""
    print("\n" + "=" * 65)
    print("   LICENSE PLATE RECOGNITION SYSTEM")
    print("   VTU — Analysis and Design of Algorithms (ADA) Micro Project")
    print("=" * 65)


def print_section(title):
    """Print a section separator with title."""
    print(f"\n{'─' * 65}")
    print(f"  ► {title}")
    print('─' * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — IMAGE LOADING
# Loads the vehicle image from the given file path using OpenCV.
# OpenCV reads images as BGR (Blue-Green-Red) format by default.
# Time Complexity: O(W × H) — must read every pixel
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_path):
    """
    Load an image from disk.

    Parameters:
        image_path (str): Path to the vehicle image file.

    Returns:
        image (ndarray): The loaded BGR image, or None on failure.
    """
    if not os.path.exists(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return None

    image = cv2.imread(image_path)

    if image is None:
        print(f"  [ERROR] Could not read image. Check format (JPG/PNG supported).")
        return None

    print(f"  [OK] Image loaded  →  Size: {image.shape[1]}×{image.shape[0]} px")
    return image


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — IMAGE PREPROCESSING
# Converts the image into a form that makes number plate detection easier.
#
# Steps:
#   a) Grayscale Conversion  — reduces 3-channel BGR to 1-channel
#   b) Bilateral Filtering   — removes noise while preserving edges
#   c) Canny Edge Detection  — highlights sharp boundaries in the image
#
# Canny Edge Detection Algorithm (Canny, 1986):
#   1. Gaussian Smoothing     — suppress high-frequency noise
#   2. Gradient Computation   — Sobel operators in X and Y direction
#   3. Non-Maximum Suppression— thin edges to single-pixel width
#   4. Double Thresholding    — classify strong vs weak edges
#   5. Edge Tracking by Hysteresis — finalize edge map
#
# Time Complexity of Preprocessing: O(W × H)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image):
    """
    Apply preprocessing pipeline: grayscale → bilateral filter → Canny edges.

    Parameters:
        image (ndarray): Original BGR image.

    Returns:
        gray     (ndarray): Grayscale version.
        filtered (ndarray): Noise-reduced grayscale image.
        edges    (ndarray): Binary edge map from Canny detector.
    """
    # Step a: Convert BGR to Grayscale
    # Formula: Gray = 0.114·B + 0.587·G + 0.299·R
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("  [OK] Grayscale conversion complete.")

    # Step b: Bilateral Filter — smooths noise but keeps edges sharp
    # Parameters: d=11 (neighbourhood diameter), sigmaColor=17, sigmaSpace=17
    filtered = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)
    print("  [OK] Bilateral filtering complete.")

    # Step c: Canny Edge Detection
    # Low threshold = 30, High threshold = 200
    # Pixels above 200 → strong edges (kept)
    # Pixels 30–200   → weak edges (kept only if connected to strong edges)
    # Pixels below 30 → discarded
    edges = cv2.Canny(filtered, threshold1=30, threshold2=200)
    print("  [OK] Canny edge detection complete.")

    return gray, filtered, edges


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — CONTOUR DETECTION & NUMBER PLATE LOCALISATION
# Finds closed contours in the edge image and filters for rectangular shapes
# that resemble a license plate (4-sided polygon with appropriate aspect ratio).
#
# Contour Detection Algorithm:
#   - Uses cv2.findContours with RETR_TREE and CHAIN_APPROX_SIMPLE
#   - Contours are sorted by area (largest first) to prioritise big regions
#   - Each contour is approximated to a polygon with cv2.approxPolyDP
#   - A 4-vertex (quadrilateral) approximation = candidate plate
#
# Time Complexity: O(W × H) for contour extraction; O(n log n) for sorting
# where n = number of contours found.
# ─────────────────────────────────────────────────────────────────────────────

def detect_number_plate(image, edges):
    """
    Detect the license plate region using contour analysis.

    Parameters:
        image (ndarray): Original BGR image (for drawing).
        edges (ndarray): Canny edge map.

    Returns:
        plate_image   (ndarray | None): Cropped plate region, or None.
        output_image  (ndarray)       : Original image with plate highlighted.
        plate_coords  (ndarray | None): Corner coordinates of the plate.
    """
    output_image = image.copy()
    plate_image  = None
    plate_coords = None

    # Find all contours in the edge map
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours by area, largest first (plates tend to be relatively large)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    print(f"  [INFO] Analysing top {len(contours)} contours for plate shape...")

    for contour in contours:
        # Approximate the contour to a polygon
        perimeter   = cv2.arcLength(contour, True)
        approx      = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

        # A rectangle has exactly 4 vertices
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            # License plates typically have aspect ratio between 1.5 and 6.0
            if 1.5 <= aspect_ratio <= 6.0 and w > 80 and h > 20:
                plate_coords = approx
                plate_image  = image[y:y + h, x:x + w]

                # Draw a green rectangle on the detected plate area
                cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 3)
                cv2.putText(
                    output_image,
                    "License Plate Detected",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                print(f"  [OK] Plate region found  →  {w}×{h} px at ({x},{y})")
                break

    if plate_image is None:
        print("  [WARN] No rectangular plate region detected.")

    return plate_image, output_image, plate_coords


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — OCR: OPTICAL CHARACTER RECOGNITION
# Extracts text from the cropped number plate image using Tesseract OCR.
#
# Tesseract uses LSTM-based neural networks (v4+) internally.
# Config '--psm 8' tells Tesseract to treat the image as a single word.
# Config '--psm 7' treats the image as a single line of text.
#
# Time Complexity: O(n) where n = number of characters in the plate
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_ocr(plate_image):
    """
    Run OCR on the cropped plate image to extract the plate number.

    Parameters:
        plate_image (ndarray): Cropped license plate image.

    Returns:
        cleaned_text (str): Extracted and cleaned plate number.
        raw_text     (str): Raw OCR output before cleaning.
    """
    # Upscale the plate for better OCR accuracy (2× zoom)
    scale        = 2
    plate_large  = cv2.resize(
        plate_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    # Convert to grayscale for OCR
    plate_gray   = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make text stand out clearly
    _, plate_thresh = cv2.threshold(
        plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Tesseract OCR configuration
    # --psm 8  → single word mode
    # --oem 3  → default LSTM engine
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    raw_text = pytesseract.image_to_string(plate_thresh, config=config)
    print(f"  [OCR] Raw output → '{raw_text.strip()}'")

    # Clean: remove spaces, newlines, special characters; uppercase
    cleaned_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    print(f"  [OCR] Cleaned    → '{cleaned_text}'")

    return cleaned_text, raw_text


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — DATABASE SEARCH
# Searches the vehicle database using the extracted plate number.
#
# Algorithm: Dictionary (Hash Map) Lookup
# - Python dicts are implemented as hash tables.
# - Average case: O(1) — direct hash lookup
# - Worst case:   O(n) — hash collision chain traversal
#
# Fallback: Linear Search for partial matches when exact match fails.
# Linear Search Time Complexity: O(n) where n = number of database entries.
# ─────────────────────────────────────────────────────────────────────────────

def search_vehicle_database(plate_number):
    """
    Look up the vehicle plate number in the database.

    Parameters:
        plate_number (str): Cleaned plate number from OCR.

    Returns:
        result (dict | None): Vehicle info if found, else None.
        matched_key (str | None): The key that matched (for display).
    """
    # --- Strategy 1: Exact Hash Map Lookup — O(1) average ---
    if plate_number in VEHICLE_DATABASE:
        print(f"  [DB] Exact match found: '{plate_number}'")
        return VEHICLE_DATABASE[plate_number], plate_number

    # --- Strategy 2: Partial / Fuzzy Linear Search — O(n) ---
    # Useful when OCR introduces minor character errors
    print("  [DB] No exact match. Attempting partial match search...")
    best_match = None
    best_key   = None
    best_score = 0

    for key in VEHICLE_DATABASE:
        # Count matching characters at correct positions
        score = sum(1 for a, b in zip(plate_number, key) if a == b)
        match_ratio = score / max(len(key), 1)

        # Accept if ≥ 80% characters match (tolerates 1–2 OCR errors)
        if match_ratio >= 0.8 and score > best_score:
            best_score = score
            best_match = VEHICLE_DATABASE[key]
            best_key   = key

    if best_match:
        print(f"  [DB] Partial match: '{best_key}' (score={best_score})")
        return best_match, best_key

    print("  [DB] Vehicle not found in database.")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 9 — DISPLAY RESULTS
# Formats and prints the final lookup result to the console.
# ─────────────────────────────────────────────────────────────────────────────

def display_results(plate_number, vehicle_info, matched_key):
    """Print the final recognition and database lookup result."""
    print_section("RECOGNITION RESULT")

    print(f"  Plate Recognised  :  {plate_number if plate_number else 'Could not read'}")

    if vehicle_info and matched_key:
        print(f"  Database Match    :  {matched_key}")
        print()
        print("  ┌─────────────────────────────────────┐")
        print("  │        ✅  VEHICLE FOUND             │")
        print("  ├─────────────────────────────────────┤")
        print(f"  │  Owner       : {vehicle_info['owner']:<22} │")
        print(f"  │  Vehicle Type: {vehicle_info['type']:<22} │")
        print(f"  │  Plate No.   : {matched_key:<22} │")
        print("  └─────────────────────────────────────┘")
    else:
        print()
        print("  ┌─────────────────────────────────────┐")
        print("  │        ❌  VEHICLE NOT FOUND         │")
        print("  │  The plate is not in the database.  │")
        print("  └─────────────────────────────────────┘")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 10 — DISPLAY WINDOWS
# Opens multiple OpenCV windows to visualise each processing stage.
# ─────────────────────────────────────────────────────────────────────────────

def display_windows(original, edges, detected_image, plate_image):
    """
    Display output windows for each processing stage.

    Windows shown:
        1. Original Image
        2. Edge Detection Output (Canny)
        3. Detected Plate (original with bounding box)
        4. Cropped Number Plate

    Press any key to close all windows.
    """
    print_section("DISPLAYING OUTPUT WINDOWS")
    print("  Press any key in any window to close all windows.")

    # Resize for consistent display height (max 500 px tall)
    def fit(img, max_h=500):
        h, w = img.shape[:2]
        if h > max_h:
            scale = max_h / h
            img = cv2.resize(img, (int(w * scale), max_h))
        return img

    cv2.imshow("1 — Original Image",          fit(original))
    cv2.imshow("2 — Edge Detection (Canny)",  fit(edges))
    cv2.imshow("3 — Plate Detected",          fit(detected_image))

    if plate_image is not None:
        # Upscale small plate crops so they're visible
        ph, pw = plate_image.shape[:2]
        if pw < 300:
            plate_image = cv2.resize(plate_image, (300, 100))
        cv2.imshow("4 — Cropped Number Plate", plate_image)
    else:
        print("  [WARN] No plate crop to display.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 11 — DEMO MODE (No real image)
# When no image path is given, run a quick database lookup demo.
# ─────────────────────────────────────────────────────────────────────────────

def run_demo_mode():
    """Demonstrate database search without requiring an actual image."""
    print_section("DEMO MODE — Database Search Demonstration")
    test_plates = ["KA01AB1234", "MH12EF9012", "XX99ZZ0000"]

    for plate in test_plates:
        print(f"\n  Searching for plate: {plate}")
        vehicle_info, matched_key = search_vehicle_database(plate)
        display_results(plate, vehicle_info, matched_key)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 12 — MAIN FUNCTION
# Entry point. Orchestrates all pipeline stages.
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print_banner()

    # ── Get image path from user ──────────────────────────────────────────────
    print("\n  Enter the path to a vehicle image.")
    print("  (Leave blank and press Enter to run DEMO mode)\n")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("  Image path: ").strip().strip('"').strip("'")

    # ── Demo mode if no path given ────────────────────────────────────────────
    if not image_path:
        run_demo_mode()
        return

    # ── STAGE 1: Load Image ───────────────────────────────────────────────────
    print_section("STAGE 1 — Loading Image")
    image = load_image(image_path)
    if image is None:
        sys.exit(1)

    # ── STAGE 2: Preprocess ───────────────────────────────────────────────────
    print_section("STAGE 2 — Image Preprocessing")
    gray, filtered, edges = preprocess_image(image)

    # ── STAGE 3: Detect Plate ─────────────────────────────────────────────────
    print_section("STAGE 3 — Number Plate Detection (Contour Analysis)")
    plate_image, detected_image, _ = detect_number_plate(image, edges)

    # ── STAGE 4: OCR ──────────────────────────────────────────────────────────
    plate_number = ""
    if plate_image is not None:
        print_section("STAGE 4 — OCR Text Extraction")
        plate_number, _ = extract_text_ocr(plate_image)
    else:
        print("\n  [SKIP] Stage 4 — No plate region to process.")

    # ── STAGE 5: Database Search ──────────────────────────────────────────────
    print_section("STAGE 5 — Database Search")
    vehicle_info, matched_key = search_vehicle_database(plate_number)

    # ── STAGE 6: Display Results ──────────────────────────────────────────────
    display_results(plate_number, vehicle_info, matched_key)

    # ── STAGE 7: Show Windows ─────────────────────────────────────────────────
    print_section("STAGE 6 — Visual Output Windows")
    # Convert edges to BGR for consistent window display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    display_windows(image, edges_bgr, detected_image, plate_image)

    print("\n  [DONE] LPR System finished.\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
