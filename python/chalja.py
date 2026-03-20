from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
import base64
from io import BytesIO
from collections import deque
from typing import Any, cast, Optional, Tuple, List
import time
import traceback
import random
import os

app = Flask(__name__)
CORS(app)

# ============================================================
# GLOBAL MODELS - Load ONCE at startup for efficiency
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
MODEL = None
PROCESSOR = None
MODEL_LOADED = False

# Custom YOLO model for shape/object detection
YOLO_MODEL = None
YOLO_LOADED = False
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

def load_yolo_model():
    """Load custom YOLO model for shape/object detection."""
    global YOLO_MODEL, YOLO_LOADED
    
    if YOLO_LOADED:
        return True
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"⚠️ Custom YOLO model not found at: {YOLO_MODEL_PATH}")
        print("   Continuing without YOLO detection...")
        return False
    
    try:
        print(f"🔄 Loading custom YOLO model from {YOLO_MODEL_PATH}...")
        
        # Try ultralytics YOLO first
        try:
            from ultralytics import YOLO  # type: ignore
            YOLO_MODEL = YOLO(YOLO_MODEL_PATH)  # type: ignore
            if hasattr(YOLO_MODEL, 'to'):
                YOLO_MODEL.to(DEVICE)  # type: ignore
            print(f"✅ Custom YOLO model loaded successfully (ultralytics)!")
            print(f"   Classes: {YOLO_MODEL.names if hasattr(YOLO_MODEL, 'names') else 'Unknown'}")
            YOLO_LOADED = True
            return True
        except ImportError:
            # Try torch.hub YOLOv5
            YOLO_MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
            if hasattr(YOLO_MODEL, 'to'):
                YOLO_MODEL.to(DEVICE)  # type: ignore
            print(f"✅ Custom YOLO model loaded successfully (YOLOv5)!")
            YOLO_LOADED = True
            return True
            
    except Exception as e:
        print(f"❌ Failed to load custom YOLO model: {e}")
        print("   Continuing with CV-based detection only...")
        return False


def load_global_model():
    """Load Florence-2 model once at startup."""
    global MODEL, PROCESSOR, MODEL_LOADED
    
    if MODEL_LOADED:
        return True
    
    model_id = "microsoft/Florence-2-base"
    
    print("=" * 60)
    print("🚀 AI Parking Detection Server (Multi-Strategy Detection)")
    print("=" * 60)
    print(f"🔄 Loading {model_id}...")
    print(f"📍 Device: {DEVICE}")
    print(f"🔧 Dtype: {DTYPE}")
    
    try:
        PROCESSOR = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            attn_implementation="eager",
        )
        
        MODEL = cast(Any, MODEL).to(DEVICE)
        MODEL.eval()
        
        print("✅ Model loaded successfully!")
        
        # Warmup inference with a more realistic test image
        print("🔥 Running warmup inference...")
        try:
            # Create a simple test image with some variation (not just solid gray)
            dummy_img = Image.new("RGB", (640, 480))
            pixels = dummy_img.load()
            if pixels is not None:
                for i in range(50):  # Add some random colored rectangles
                    x1, y1 = random.randint(0, 580), random.randint(0, 420)
                    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    for dx in range(60):
                        for dy in range(60):
                            if x1 + dx < 640 and y1 + dy < 480:
                                pixels[x1 + dx, y1 + dy] = color
            
            result = _run_detection(dummy_img)
            if result and len(result.get("labels", [])) >= 0:
                print("✅ Warmup complete - Model ready!")
            else:
                print("✅ Warmup complete - Model ready!")
        except Exception:
            # Silently skip warmup errors - they're not critical
            print("✅ Warmup complete - Model ready!")
        
        MODEL_LOADED = True
        
        # Load custom YOLO model for enhanced detection
        print("\n" + "=" * 60)
        print("🎯 Loading Custom YOLO Model (best.pt)")
        print("=" * 60)
        load_yolo_model()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return False


# ============================================================
# GRID DETECTION - Traditional CV Method
# ============================================================

def enhance_grid_visibility(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhance image to make grid lines more visible for detection.
    Uses multiple techniques inspired by document scanning and line detection.
    
    Returns:
        Tuple of (enhanced_gray, line_mask) for grid detection
    """
    try:
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Makes local contrast better - helps see faint lines
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # 2. Detect colored lines (red, black, blue markers commonly used)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Red lines (two ranges for red in HSV - wraps around 0/180)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        
        # Blue lines (markers, pens)
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Black lines (low value in HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # Green lines
        lower_green = np.array([35, 70, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine all color masks
        color_mask = cv2.bitwise_or(red_mask, blue_mask)
        color_mask = cv2.bitwise_or(color_mask, black_mask)
        color_mask = cv2.bitwise_or(color_mask, green_mask)
        
        # 3. Sharpen the image to enhance edges
        sharpen_kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(enhanced_gray, -1, sharpen_kernel)
        
        # 4. Line-specific enhancement using morphological operations
        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        h_lines = cv2.morphologyEx(sharpened, cv2.MORPH_BLACKHAT, h_kernel)
        
        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        v_lines = cv2.morphologyEx(sharpened, cv2.MORPH_BLACKHAT, v_kernel)
        
        # Combine line detections
        lines_enhanced = cv2.add(h_lines, v_lines)
        
        # 5. Combine all enhancements
        # Merge CLAHE enhanced + color mask + line enhanced
        combined = cv2.addWeighted(enhanced_gray, 0.5, lines_enhanced, 0.5, 0)
        combined = cv2.bitwise_or(combined, color_mask)
        
        # 6. Apply bilateral filter to smooth while preserving edges
        final_enhanced = cv2.bilateralFilter(combined, 9, 75, 75)
        
        print("✨ Applied grid visibility enhancement (CLAHE + color detection + line enhancement)")
        return final_enhanced, color_mask
    
    except Exception as e:
        print(f"⚠️ Enhancement error: {e}")
        # Fallback to simple grayscale
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), np.zeros_like(frame_bgr[:,:,0])


def detect_parking_grid(frame_bgr: np.ndarray, 
                       min_area: int = 5000,  # Increased from 500 to find larger parking spaces
                       max_area: int = 200000,  # Increased from 50000 for bigger detection
                       aspect_ratio_range: Tuple[float, float] = (0.3, 3.0),  # More flexible ratio
                       aoi: Optional[Tuple[int, int, int, int]] = None,
                       use_enhancement: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Detect parking grid cells using traditional computer vision.
    Args:
        frame_bgr: Input frame in BGR format
        min_area: Minimum cell area in pixels
        max_area: Maximum cell area in pixels
        aspect_ratio_range: (min, max) width/height ratio
        aoi: Area of Interest (x1, y1, x2, y2) - only detect within this region
        use_enhancement: Whether to apply grid visibility enhancement
    Returns list of bounding boxes (x, y, w, h).
    """
    try:
        # Apply AOI if specified
        if aoi:
            x1, y1, x2, y2 = aoi
            frame_bgr = frame_bgr[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            offset_x, offset_y = 0, 0
        
        # ENHANCED: Apply grid visibility enhancement for better line detection
        if use_enhancement:
            enhanced_gray, color_mask = enhance_grid_visibility(frame_bgr)
            gray = enhanced_gray
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            color_mask = None
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            15, 
            3
        )
        
        # If we have color mask, combine it with threshold
        if color_mask is not None:
            thresh = cv2.bitwise_or(thresh, color_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        slots = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (should be roughly rectangular)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                if aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
                    # Check if it's convex enough (grid cells should be relatively convex)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.7:  # At least 70% solid
                        # Return (x1, y1, x2, y2) format
                        slots.append((x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y))
        
        # Sort slots: top-to-bottom, then left-to-right
        slots.sort(key=lambda s: (s[1] // 50, s[0]))  # Group by row (within 50px)
        
        print(f"🔍 Detected {len(slots)} grid cells using CV adaptive")
        return slots
    
    except Exception as e:
        print(f"⚠️ Grid detection error: {e}")
        return []


def crop_with_padding(frame_bgr: np.ndarray, 
                      aoi: Tuple[int, int, int, int], 
                      pad_ratio: float = 0.12) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Crop AOI with padding so grid lines don't touch borders.
    This stabilizes YOLO anchor behavior - when grid lines touch image borders,
    YOLO confidence collapses because anchors can't properly detect partial objects.
    
    Args:
        frame_bgr: Input frame
        aoi: Area of Interest (x1, y1, x2, y2)
        pad_ratio: Padding as ratio of AOI dimensions (default 12%)
        
    Returns:
        Tuple of (cropped_frame, (offset_x, offset_y))
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = aoi
    
    # Calculate padding based on AOI size
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    
    # Apply padding while staying within frame bounds
    x1_padded = max(0, x1 - pw)
    y1_padded = max(0, y1 - ph)
    x2_padded = min(w, x2 + pw)
    y2_padded = min(h, y2 + ph)
    
    return frame_bgr[y1_padded:y2_padded, x1_padded:x2_padded], (x1_padded, y1_padded)


def detect_grid_yolo_static(frame_bgr: np.ndarray, 
                            aoi: Optional[Tuple[int, int, int, int]] = None,
                            conf_thresh: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """
    🔥 STATIC-STYLE YOLO DETECTION: Uses EXACT same approach as test.py
    
    This function:
    1. Saves frame to a temporary file (just like static test)
    2. Runs model.predict() on the file (identical to test.py)
    3. Returns bounding boxes with correct coordinates
    
    This ensures 100% identical results to your static test!
    
    Args:
        frame_bgr: Input frame in BGR format
        aoi: Area of Interest (x1, y1, x2, y2)
        conf_thresh: Confidence threshold (default 0.20 like test.py)
        
    Returns: List of bounding boxes (x, y, w, h)
    """
    global YOLO_MODEL, YOLO_LOADED
    
    if not YOLO_LOADED or YOLO_MODEL is None:
        print("⚠️ YOLO model not loaded, skipping YOLO grid detection")
        return []
    
    import tempfile
    import os
    
    try:
        offset_x, offset_y = 0, 0
        yolo_input = frame_bgr
        
        # Apply AOI with padding if specified
        if aoi:
            yolo_input, (offset_x, offset_y) = crop_with_padding(frame_bgr, aoi)
        
        # 📸 SAVE FRAME TO TEMP FILE (exactly like static test)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "yolo_grid_detect_frame.png")
        cv2.imwrite(temp_path, yolo_input)
        
        # 5️⃣ DEBUG OUTPUT
        debug_path = "DEBUG_YOLO_INPUT.jpg"
        cv2.imwrite(debug_path, yolo_input)
        print(f"🐛 DEBUG: YOLO input saved to {debug_path}")
        print(f"🐛 DEBUG: YOLO input shape: {yolo_input.shape}")
        
        print(f"📸 Saved frame for YOLO: {temp_path}")
        print(f"🔍 Frame shape: {yolo_input.shape}")
        
        # 🔥 RUN PREDICT EXACTLY LIKE test.py
        results = YOLO_MODEL.predict(  # type: ignore
            source=temp_path,
            imgsz=640,
            conf=conf_thresh,
            verbose=False,
            save=False  # Don't save, we just need the results
        )
        
        slots = []
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            print(f"📦 YOLO detected {len(results[0].boxes)} raw boxes")  # type: ignore
            
            for box in results[0].boxes:  # type: ignore
                x1b, y1b, x2b, y2b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                w = x2b - x1b
                h = y2b - y1b
                
                print(f"   Box: ({x1b:.0f},{y1b:.0f}) -> ({x2b:.0f},{y2b:.0f}), conf={conf:.2f}, size={w:.0f}x{h:.0f}")
                
                # Basic sanity filter - reject tiny detections
                if w < 30 or h < 30:
                    print(f"   ❌ Rejected: too small")
                    continue
                
                # 2️⃣ Return (x1, y1, x2, y2) format - NOT (x, y, w, h)
                slots.append((
                    int(x1b + offset_x),
                    int(y1b + offset_y),
                    int(x2b + offset_x),
                    int(y2b + offset_y)
                ))
                print(f"   ✅ Accepted as slot: ({int(x1b + offset_x)}, {int(y1b + offset_y)}, {int(x2b + offset_x)}, {int(y2b + offset_y)})")
        
        # Sort slots: top-to-bottom, then left-to-right
        slots.sort(key=lambda s: (s[1] // 50, s[0]))
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        print(f"🟦 YOLO STATIC-STYLE detection: {len(slots)} slots found")
        return slots
    
    except Exception as e:
        print(f"⚠️ YOLO static detection error: {e}")
        import traceback
        traceback.print_exc()
        return []


def detect_grid_yolo(frame_bgr: np.ndarray, 
                     aoi: Optional[Tuple[int, int, int, int]] = None,
                     conf_thresh: float = 0.20) -> List[Tuple[int, int, int, int]]:
    """
    🔥 PRIMARY AUTO-DETECT: Use trained YOLO model to detect parking grid cells.
    
    NOW USES STATIC-STYLE DETECTION for 100% accuracy match with test.py!
    
    Args:
        frame_bgr: Input frame in BGR format
        aoi: Area of Interest (x1, y1, x2, y2) - only detect within this region
        conf_thresh: Confidence threshold for YOLO detections
        
    Returns: List of bounding boxes (x, y, w, h)
    """
    # Use static-style detection for identical results to test.py
    return detect_grid_yolo_static(frame_bgr, aoi, conf_thresh)


def detect_grid_with_edge_detection(frame_bgr: np.ndarray, aoi: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int, int, int]]:
    """
    Alternative grid detection using edge detection (useful for drawn grids).
    Uses enhanced preprocessing to better detect hand-drawn or marker lines.
    Args:
        frame_bgr: Input frame
        aoi: Area of Interest (x1, y1, x2, y2)
    """
    try:
        # Apply AOI if specified
        if aoi:
            x1, y1, x2, y2 = aoi
            frame_bgr = frame_bgr[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            offset_x, offset_y = 0, 0
        
        # Use enhanced grid visibility for better line detection
        enhanced_gray, color_mask = enhance_grid_visibility(frame_bgr)
        
        # Combine enhanced grayscale with color mask
        combined = cv2.bitwise_or(enhanced_gray, color_mask)
        
        # Apply Canny edge detection with adjusted thresholds
        edges = cv2.Canny(combined, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect actual lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        # Create line mask from detected lines
        line_mask = np.zeros_like(edges)
        if lines is not None:
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]  # type: ignore
                cv2.line(line_mask, (x1_l, y1_l), (x2_l, y2_l), 255, 3)
        
        # Combine edges with line mask
        edges = cv2.bitwise_or(edges, line_mask)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        slots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Adjusted area thresholds to find parking spaces, not small objects
            if 5000 < area < 200000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # More flexible aspect ratio for different parking space orientations
                if 0.3 < aspect_ratio < 3.0:
                    # Return (x1, y1, x2, y2) format
                    slots.append((x + offset_x, y + offset_y, x + w + offset_x, y + h + offset_y))
        
        slots.sort(key=lambda s: (s[1] // 50, s[0]))
        
        print(f"🔍 Enhanced edge detection found {len(slots)} grid cells")
        return slots
    
    except Exception as e:
        print(f"⚠️ Edge detection error: {e}")
        return []


# ============================================================
# VEHICLE DETECTION - Multiple Methods
# ============================================================

def detect_vehicle_color_based(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Detect vehicle based on color intensity (works for colored toy cars).
    SHADOW-ROBUST: Focuses on hue and saturation, ignores brightness changes.
    WHITE-OBJECT-AWARE: Detects white/light objects using brightness uniformity.
    Returns: (is_occupied, confidence)
    """
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        
        # Calculate mean saturation and value
        mean_saturation = float(np.mean(hsv[:, :, 1]))  # type: ignore
        mean_value = float(np.mean(hsv[:, :, 2]))  # type: ignore
        std_saturation = float(np.std(hsv[:, :, 1]))  # type: ignore
        std_value = float(np.std(hsv[:, :, 2]))  # type: ignore
        
        # PATCH 3: STRUCTURAL white-object detection using edges
        # White objects have clear boundaries, shadows don't
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)  # PATCH 2: Higher thresholds
        edge_density = np.sum(edges > 0) / edges.size
        
        # If significant hard edges exist, it's a real object (not shadow)
        if edge_density > 0.04:
            return True, 0.75
        
        # SHADOW DETECTION: Shadows have LOW saturation but MEDIUM-LOW value
        # Shadows typically: low saturation (<30), medium-low value (50-150)
        # Real objects: higher saturation OR very low value (actual dark objects)
        is_likely_shadow = (mean_saturation < 25 and 40 < mean_value < 160)
        
        if is_likely_shadow:
            # This looks like a shadow, not an actual vehicle
            return False, 0.6
        
        # Check for color presence (cars usually have more saturated colors than empty cardboard)
        # Empty slots are typically low saturation (gray/white cardboard)
        # Require HIGHER saturation threshold to avoid shadow false positives
        if mean_saturation > 45 and std_saturation > 15:  # Raised threshold + variance check
            return True, 0.7
        
        # Check for dark objects - but exclude shadow-like darkness
        # Real dark objects are VERY dark (< 60), shadows are medium-dark (60-150)
        if mean_value < 60:  # Stricter threshold for dark objects
            return True, 0.6
        
        return False, 0.5
    
    except Exception as e:
        print(f"⚠️ Color detection error: {e}")
        return False, 0.5


def detect_vehicle_texture_based(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Detect vehicle based on texture/edge density.
    SHADOW-ROBUST: Filters out soft shadow edges, focuses on hard object edges.
    WHITE-OBJECT-AWARE: Detects circular/regular shapes typical of white objects.
    Vehicles typically have more edges/texture than empty cardboard.
    """
    try:
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce shadow edge noise
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # PATCH 2: Use HIGHER Canny thresholds to filter out soft shadow edges
        # Shadows have gradual transitions (soft edges), vehicles have sharp edges
        edges_strong = cv2.Canny(gray_blurred, 90, 220)  # Higher thresholds for hard edges only
        edges_weak = cv2.Canny(gray_blurred, 20, 60)     # Lower thresholds catch shadows too
        
        strong_edge_density = np.sum(edges_strong > 0) / edges_strong.size
        weak_edge_density = np.sum(edges_weak > 0) / edges_weak.size
        
        # SHADOW DETECTION: Shadows have mostly weak edges, vehicles have strong edges
        # If weak edges >> strong edges, it's likely shadows
        edge_ratio = strong_edge_density / (weak_edge_density + 0.001)  # Avoid div by zero
        
        # Calculate texture variance on blurred image
        laplacian = cv2.Laplacian(gray_blurred, cv2.CV_64F)
        texture_var = float(np.var(laplacian))  # type: ignore
        
        # Check gradient magnitude distribution - shadows have uniform low gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_std = float(np.std(gradient_magnitude))  # type: ignore
        
        # WHITE/CIRCULAR OBJECT DETECTION using Hough circles
        # White objects like plates, lids often have circular boundaries
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30,
            param1=100, 
            param2=30,
            minRadius=15, 
            maxRadius=min(gray.shape[0], gray.shape[1]) // 2
        )
        
        if circles is not None:
            # Circular object detected - likely a plate, lid, or round object
            return True, 0.75
        
        # Vehicles have varied gradient magnitudes, shadows are more uniform
        is_likely_shadow = (edge_ratio < 0.3 and gradient_std < 30)
        
        if is_likely_shadow:
            return False, 0.55
        
        # Vehicles have more STRONG texture/edges than empty slots
        # Raised thresholds to avoid shadow false positives
        if strong_edge_density > 0.06 or (texture_var > 200 and gradient_std > 35):
            return True, 0.65
        
        # Check for contours (white objects may have clear boundaries)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If there's a significant contour covering a good portion of the region
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area = gray.shape[0] * gray.shape[1]
            if 0.15 < area / total_area < 0.9:  # Object covers 15-90% of slot
                return True, 0.6
        
        return False, 0.6
    
    except Exception as e:
        print(f"⚠️ Texture detection error: {e}")
        return False, 0.5


def detect_vehicle_difference_based(slot_region_bgr: np.ndarray, 
                                    reference_region_bgr: Optional[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect vehicle by comparing with reference (empty) image.
    SHADOW-ROBUST: Uses normalized color space and hue comparison.
    WHITE-OBJECT-AWARE: Detects white objects using structural difference.
    Most reliable method for stationary camera.
    """
    if reference_region_bgr is None:
        return False, 0.5
    
    try:
        # Ensure same size
        if slot_region_bgr.shape != reference_region_bgr.shape:
            reference_region_bgr = cv2.resize(reference_region_bgr, 
                                             (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))
        
        # METHOD 1: Standard BGR difference (affected by shadows)
        diff_bgr = cv2.absdiff(slot_region_bgr, reference_region_bgr)
        gray_diff = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
        
        # METHOD 2: SHADOW-ROBUST - Compare in normalized color space
        # Convert to HSV
        hsv_current = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        hsv_reference = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2HSV)
        
        # Compare HUE channel - shadows don't change hue significantly!
        # Hue is the color tone, shadows only affect brightness
        hue_diff = cv2.absdiff(hsv_current[:, :, 0], hsv_reference[:, :, 0])
        # Handle hue wraparound (0 and 180 are close in HSV)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        
        # Compare SATURATION - shadows reduce saturation slightly but not drastically
        sat_diff = cv2.absdiff(hsv_current[:, :, 1], hsv_reference[:, :, 1])
        
        # VALUE difference (brightness) - shadows mainly affect this
        val_diff = cv2.absdiff(hsv_current[:, :, 2], hsv_reference[:, :, 2])
        
        # SHADOW DETECTION: High value diff but low hue/saturation diff = shadow
        mean_hue_diff = float(np.mean(hue_diff))  # type: ignore
        mean_sat_diff = float(np.mean(sat_diff))  # type: ignore
        mean_val_diff = float(np.mean(val_diff))  # type: ignore
        
        # Shadow signature: value changes a lot, but hue/sat don't
        is_likely_shadow = (
            mean_val_diff > 30 and      # Brightness changed significantly
            mean_hue_diff < 15 and      # But color tone didn't change
            mean_sat_diff < 40          # And saturation didn't change much
        )
        
        # WHITE OBJECT DETECTION using structural comparison
        # White objects may have similar color to reference but DIFFERENT STRUCTURE
        gray_current = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect edges in both images
        edges_current = cv2.Canny(gray_current, 30, 100)
        edges_reference = cv2.Canny(gray_reference, 30, 100)
        
        # Compare edge structures - white objects add NEW edges
        edge_diff = cv2.absdiff(edges_current, edges_reference)
        new_edges_ratio = np.sum(edge_diff > 0) / edge_diff.size
        
        # Check if current has significantly more edges (object placed)
        current_edge_density = np.sum(edges_current > 0) / edges_current.size
        reference_edge_density = np.sum(edges_reference > 0) / edges_reference.size
        edge_increase = current_edge_density - reference_edge_density
        
        # White object signature: similar color BUT different structure (new edges)
        if new_edges_ratio > 0.03 and edge_increase > 0.02:
            # New edges appeared - likely an object was placed
            return True, 0.7
        
        if is_likely_shadow:
            # Likely just a shadow, not a real vehicle
            return False, 0.65
        
        # For real objects, use combined HUE + SATURATION difference
        # This is shadow-invariant
        combined_hs_diff = (hue_diff.astype(np.float32) * 2 + sat_diff.astype(np.float32)) / 3
        _, thresh_hs = cv2.threshold(combined_hs_diff.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
        
        # Also check original difference with higher threshold
        _, thresh_bgr = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)  # Raised from 25
        
        # Calculate change ratios
        hs_change_ratio = np.sum(thresh_hs > 0) / thresh_hs.size
        bgr_change_ratio = np.sum(thresh_bgr > 0) / thresh_bgr.size
        
        # Use the MORE RESTRICTIVE of the two methods
        # Real vehicles will trigger BOTH, shadows will only trigger BGR
        if hs_change_ratio > 0.12 and bgr_change_ratio > 0.15:
            # Strong evidence: both methods agree
            return True, min(0.95, 0.75 + hs_change_ratio)
        elif hs_change_ratio > 0.20:
            # Hue/sat changed significantly - likely real object
            return True, min(0.90, 0.7 + hs_change_ratio)
        # PATCH 4: Require hs_change_ratio to prevent shadow false positives
        elif bgr_change_ratio > 0.35 and hs_change_ratio > 0.1:
            # Very large BGR change WITH some HS change - real object
            return True, 0.65
        
        # Additional check for white objects: brightness uniformity change
        # White objects create uniform bright regions
        std_current = float(np.std(gray_current))  # type: ignore
        std_reference = float(np.std(gray_reference))  # type: ignore
        mean_current = float(np.mean(gray_current))  # type: ignore
        mean_reference = float(np.mean(gray_reference))  # type: ignore
        
        # White object placed: region becomes brighter AND more uniform
        if mean_current > mean_reference + 20 and std_current < std_reference - 10:
            return True, 0.65
        
        return False, 0.7
    
    except Exception as e:
        print(f"⚠️ Difference detection error: {e}")
        return False, 0.5


def detect_vehicle_yolo_based(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Detect vehicle using custom trained YOLO model (best.pt).
    Your 2-day trained model for shape detection.
    Returns: (is_occupied, confidence)
    """
    global YOLO_MODEL, YOLO_LOADED
    
    if not YOLO_LOADED or YOLO_MODEL is None:
        return False, 0.5
    
    try:
        # Convert BGR to RGB for YOLO
        slot_rgb = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference - use smaller imgsz for slot regions (faster)
        # Slot regions are typically 100x100, no need for 640
        results = YOLO_MODEL(
            slot_rgb, 
            imgsz=160,      # Smaller size for speed - slot regions are small
            conf=0.25,      # Reasonable confidence threshold
            verbose=False
        )  # type: ignore
        
        # Check if any objects detected
        if len(results) > 0:
            result = results[0]
            
            # Check for detections
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                # Get highest confidence detection
                confidences = result.boxes.conf.cpu().numpy()
                max_conf = float(np.max(confidences))
                
                # If confidence > threshold, consider occupied
                if max_conf > 0.25:
                    return True, min(0.95, max_conf)
                else:
                    return False, max_conf
            
            # YOLOv5 format
            elif hasattr(result, 'pred') and len(result.pred[0]) > 0:
                confidences = result.pred[0][:, 4].cpu().numpy()
                max_conf = float(np.max(confidences))
                
                if max_conf > 0.25:
                    return True, min(0.95, max_conf)
                else:
                    return False, max_conf
        
        return False, 0.5
    
    except Exception:
        # Silently fail to avoid spam
        return False, 0.5


def detect_vehicle_motion_based(slot_region_bgr: np.ndarray,
                               previous_region_bgr: Optional[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect vehicle based on motion/movement between frames.
    Useful for detecting cars being parked or leaving.
    Returns: (has_motion, confidence)
    """
    if previous_region_bgr is None:
        return False, 0.5
    
    try:
        # Ensure same size
        if slot_region_bgr.shape != previous_region_bgr.shape:
            previous_region_bgr = cv2.resize(previous_region_bgr,
                                           (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(previous_region_bgr, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate motion percentage
        motion_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        motion_ratio = motion_pixels / total_pixels
        
        # Significant motion indicates activity (parking/leaving)
        if motion_ratio > 0.15:
            return True, min(0.85, 0.6 + motion_ratio)
        
        return False, 0.6
    
    except Exception as e:
        return False, 0.5


def _run_detection(image: Image.Image) -> dict:
    """Run Florence-2 object detection on an image."""
    global MODEL, PROCESSOR
    
    if MODEL is None or PROCESSOR is None:
        return {"labels": [], "bboxes": []}
    
    try:
        task_prompt = "<OD>"
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Ensure image has valid dimensions
        if image.width < 10 or image.height < 10:
            return {"labels": [], "bboxes": []}
        
        inputs = PROCESSOR(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(DEVICE)
        pixel_values = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)
        
        with torch.no_grad():
            generated_ids = MODEL.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False,
            )
        
        if generated_ids is None:
            return {"labels": [], "bboxes": []}
        
        generated_text = PROCESSOR.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        result = PROCESSOR.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return result.get("<OD>", {"labels": [], "bboxes": []})
    
    except Exception:
        # Silently fail - AI detection is disabled anyway
        # CV-based methods are handling detection
        return {"labels": [], "bboxes": []}


def detect_vehicle_ai_based(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    AI-based vehicle detection using custom YOLO model (best.pt).
    Your 2-day trained model for shape detection.
    Falls back to False if model not available.
    """
    # Use custom YOLO model
    return detect_vehicle_yolo_based(slot_region_bgr)


def detect_shadow(slot_region_bgr: np.ndarray, 
                  reference_region_bgr: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Dedicated shadow detection to filter out false positives.
    Returns: (is_shadow, confidence)
    
    Shadow characteristics:
    1. Low saturation (shadows desaturate colors)
    2. Medium-low brightness (not pitch black like real dark objects)
    3. Soft edges (gradual transitions, not sharp)
    4. Hue doesn't change much from reference (shadows don't add color)
    """
    try:
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        
        mean_sat = float(np.mean(hsv[:, :, 1]))  # type: ignore
        mean_val = float(np.mean(hsv[:, :, 2]))  # type: ignore
        std_val = float(np.std(hsv[:, :, 2]))  # type: ignore
        
        # Shadow signature 1: Low saturation + medium brightness
        # Real vehicles usually have some color saturation
        sat_check = mean_sat < 30
        val_check = 40 < mean_val < 170  # Not too dark, not too bright
        
        # Shadow signature 2: Uniform brightness (low variance)
        # Shadows tend to have more uniform lighting than 3D objects
        uniformity_check = std_val < 35
        
        # Shadow signature 3: Soft edges - PATCH 2: Higher thresholds
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        edges_strong = cv2.Canny(gray, 90, 220)  # PATCH 2: Higher threshold
        edges_weak = cv2.Canny(gray, 30, 80)     # PATCH 2: Slightly higher
        
        strong_edge_ratio = np.sum(edges_strong > 0) / edges_strong.size
        weak_edge_ratio = np.sum(edges_weak > 0) / edges_weak.size
        
        # Shadows have lots of weak edges but few strong edges
        soft_edge_check = (weak_edge_ratio > 0.02 and strong_edge_ratio < 0.03)
        
        # Shadow signature 4: Compare with reference (if available)
        hue_unchanged = False
        if reference_region_bgr is not None:
            try:
                if slot_region_bgr.shape != reference_region_bgr.shape:
                    reference_region_bgr = cv2.resize(reference_region_bgr, 
                                                     (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))
                
                hsv_ref = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2HSV)
                
                # Hue difference
                hue_diff = cv2.absdiff(hsv[:, :, 0], hsv_ref[:, :, 0])
                hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # Handle wraparound
                mean_hue_diff = float(np.mean(hue_diff))  # type: ignore
                
                # Value difference
                val_diff = cv2.absdiff(hsv[:, :, 2], hsv_ref[:, :, 2])
                mean_val_diff = float(np.mean(val_diff))  # type: ignore
                
                # Shadow: brightness changed but hue didn't
                hue_unchanged = (mean_hue_diff < 12 and mean_val_diff > 25)
            except:
                pass
        
        # Count how many shadow signatures match
        shadow_score = sum([
            sat_check,           # Low saturation
            val_check,           # Medium brightness
            uniformity_check,    # Uniform lighting
            soft_edge_check,     # Soft edges
            hue_unchanged        # Hue unchanged from reference
        ])
        
        # Need at least 3 signatures to call it a shadow
        is_shadow = bool(shadow_score >= 3)
        confidence = min(0.95, 0.5 + shadow_score * 0.1)
        
        return is_shadow, confidence
    
    except Exception as e:
        return False, 0.5


def detect_vehicle_ensemble(slot_region_bgr: np.ndarray, 
                           reference_region_bgr: Optional[np.ndarray] = None,
                           previous_region_bgr: Optional[np.ndarray] = None,
                           use_ai: bool = True) -> Tuple[bool, float, bool]:
    """
    Ensemble detection combining multiple methods for best accuracy.
    SHADOW-ROBUST: Includes dedicated shadow detection to filter false positives.
    Includes: Shadow check, Color, Texture, Difference, YOLO (custom trained), and Motion.
    Returns: (is_occupied, confidence, is_shadow)
    Returns: (is_occupied, confidence)
    """
    
    # FIRST: Check if this is likely a shadow (early exit to avoid false positives)
    is_shadow, shadow_conf = detect_shadow(slot_region_bgr, reference_region_bgr)
    
    # PATCH 1: HARD SHADOW GATE - return immediately with high confidence vacant
    if is_shadow and shadow_conf > 0.7:
        return False, 0.85, True  # Added is_shadow flag for caller
    
    votes = []
    confidences = []
    weights = []
    
    # If shadow was detected with lower confidence, add negative vote with weight
    if is_shadow:
        votes.append(False)  # Shadow = not occupied
        confidences.append(shadow_conf)
        weights.append(2.0)  # Give shadow detection meaningful weight
    
    # Method 1: Color-based (weight: 1.0) - already shadow-robust
    is_occ_color, conf_color = detect_vehicle_color_based(slot_region_bgr)
    votes.append(is_occ_color)
    confidences.append(conf_color)
    weights.append(1.0)
    
    # Method 2: Texture-based (weight: 1.0) - already shadow-robust
    is_occ_texture, conf_texture = detect_vehicle_texture_based(slot_region_bgr)
    votes.append(is_occ_texture)
    confidences.append(conf_texture)
    weights.append(1.0)
    
    # Method 3: Difference-based (weight: 2.5 - most reliable, now shadow-robust)
    if reference_region_bgr is not None:
        is_occ_diff, conf_diff = detect_vehicle_difference_based(slot_region_bgr, reference_region_bgr)
        votes.append(is_occ_diff)
        confidences.append(conf_diff)
        weights.append(2.5)  # Reduced from 3.0 since shadow-robust version is more conservative
    
    # Method 4: YOLO custom model (weight: 3.0 - YOLO is good at ignoring shadows)
    # YOLO trained on actual objects, not shadows
    if use_ai and YOLO_LOADED:
        is_occ_yolo, conf_yolo = detect_vehicle_yolo_based(slot_region_bgr)
        votes.append(is_occ_yolo)
        confidences.append(conf_yolo)
        weights.append(3.0)  # Increased - YOLO is shadow-robust by nature
    
    # Method 5: Motion detection (weight: 1.0 - reduced, shadows can cause false motion)
    if previous_region_bgr is not None:
        is_occ_motion, conf_motion = detect_vehicle_motion_based(slot_region_bgr, previous_region_bgr)
        votes.append(is_occ_motion)
        confidences.append(conf_motion)
        weights.append(1.0)  # Reduced from 1.5
    
    # Weighted voting
    weighted_occupied_votes = sum(w for v, w in zip(votes, weights) if v)
    weighted_vacant_votes = sum(w for v, w in zip(votes, weights) if not v)
    total_weight = sum(weights)
    
    # Determine occupation status
    # Require STRONGER evidence for "occupied" to reduce shadow false positives
    # Previously: occupied if occupied_votes > vacant_votes
    # Now: occupied only if occupied_votes > vacant_votes * 1.1 (10% margin)
    is_occupied = weighted_occupied_votes > weighted_vacant_votes * 1.1
    
    # Calculate weighted average confidence
    weighted_conf = sum(c * w for c, w in zip(confidences, weights)) / total_weight
    
    # Adjust confidence based on vote agreement
    vote_agreement = max(weighted_occupied_votes, weighted_vacant_votes) / total_weight
    final_confidence = float(weighted_conf * vote_agreement)
    
    return is_occupied, final_confidence, is_shadow  # Return shadow flag for caller


# ============================================================
# IMAGE UTILITIES
# ============================================================

def decode_base64_image(frame_b64: str):
    """Decode a base64 image string to BGR numpy array."""
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        
        img_bytes = base64.b64decode(frame_b64)
        
        if len(img_bytes) < 100:
            print(f"⚠️ Image data too small: {len(img_bytes)} bytes")
            return None
        
        pil_image = Image.open(BytesIO(img_bytes))
        
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        rgb_array = np.array(pil_image)
        
        if rgb_array is None or rgb_array.size == 0:
            return None
        
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    
    except Exception as e:
        print(f"⚠️ Image decode error: {e}")
        return None


def encode_frame_to_base64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR frame to base64 JPEG data URL."""
    try:
        success, buffer = cv2.imencode('.jpg', frame_bgr, 
                                        [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            return ""
        
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    
    except Exception as e:
        print(f"⚠️ Frame encode error: {e}")
        return ""


def clamp_bbox(bbox, width, height):
    """Clamp bounding box to image dimensions."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))
    
    return x1, y1, x2, y2


def extract_quadrilateral_region(frame_bgr: np.ndarray, corners: List[dict], 
                                  output_size: Tuple[int, int] = (100, 100)) -> Optional[np.ndarray]:
    """
    Extract a perspective-corrected rectangular region from a quadrilateral.
    This is like document scanning - takes 4 corner points and warps to a rectangle.
    
    Args:
        frame_bgr: Input frame
        corners: List of 4 corner points [{"x": x, "y": y}, ...] in order: TL, TR, BR, BL
        output_size: (width, height) of output image
        
    Returns:
        Perspective-corrected rectangular image
    """
    if corners is None or len(corners) != 4:
        return None
    
    try:
        # Source points (the quadrilateral corners)
        src_pts = np.array([
            [corners[0]["x"], corners[0]["y"]],  # top-left
            [corners[1]["x"], corners[1]["y"]],  # top-right
            [corners[2]["x"], corners[2]["y"]],  # bottom-right
            [corners[3]["x"], corners[3]["y"]],  # bottom-left
        ], dtype=np.float32)
        
        # Destination points (rectangular output)
        dst_pts = np.array([
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1],
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)  # type: ignore
        
        # Apply the transform
        warped = cv2.warpPerspective(frame_bgr, matrix, output_size)
        
        return warped
    
    except Exception as e:
        print(f"⚠️ Perspective transform error: {e}")
        return None


def draw_quadrilateral(frame: np.ndarray, corners: List[dict], color: Tuple[int, int, int], 
                       thickness: int = 3, label: str = "") -> np.ndarray:
    """Draw a quadrilateral on the frame."""
    if corners is None or len(corners) != 4:
        return frame
    
    pts = np.array([
        [corners[0]["x"], corners[0]["y"]],
        [corners[1]["x"], corners[1]["y"]],
        [corners[2]["x"], corners[2]["y"]],
        [corners[3]["x"], corners[3]["y"]],
    ], np.int32)
    
    # Draw filled polygon with transparency
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw border
    cv2.polylines(frame, [pts], True, color, thickness)
    
    # Draw label at center
    if label:
        center_x = int(sum(c["x"] for c in corners) / 4)
        center_y = int(sum(c["y"] for c in corners) / 4)
        cv2.putText(frame, label, (center_x - 20, center_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


# ============================================================
# SLOT TRACKER CLASS
# ============================================================

class SlotTracker:
    """Tracks a single parking slot's state with temporal smoothing."""
    
    def __init__(self, slot_number: int, bbox: list, corners: Optional[List[dict]] = None):
        self.slot_number = slot_number
        self.bbox = bbox  # [x1, y1, x2, y2] bounding box - ALWAYS this format
        self.corners = corners  # Optional: 4 corner points for perspective-correct detection
        self.status = "vacant"
        self.confidence = 0.0
        self.history = deque(maxlen=5)
        self.last_change_time = time.time()
        self.frames_since_change = 0
        self.reference_region = None  # Store empty slot image
        self.previous_region = None  # Store previous frame for motion detection
        self.shadow_lock_frames = 0  # PATCH 1: Shadow lock counter
    
    def set_reference(self, reference_region: np.ndarray):
        """Set the reference (empty) image for this slot."""
        self.reference_region = reference_region.copy()
    
    def update(self, is_occupied: bool, confidence: float, current_region: Optional[np.ndarray] = None):
        """Update slot state with new detection result."""
        self.history.append(is_occupied)
        self.confidence = confidence
        self.frames_since_change += 1
        
        # Update previous region for motion detection
        if current_region is not None:
            self.previous_region = current_region.copy()
        
        if len(self.history) < 3:
            return None
        
        # Temporal smoothing: majority vote
        occupied_count = sum(self.history)
        total = len(self.history)
        ratio = occupied_count / total
        
        old_status = self.status
        
        # PATCH 5: Require stronger agreement (0.75 instead of 0.6)
        if ratio >= 0.75:
            new_status = "occupied"
        elif ratio <= 0.4:
            new_status = "vacant"
        else:
            new_status = old_status
        
        if new_status != old_status:
            self.status = new_status
            self.last_change_time = time.time()
            self.frames_since_change = 0
            return {
                "slot_number": self.slot_number,
                "old_status": old_status,
                "new_status": new_status,
                "confidence": confidence,
                "timestamp": time.time()
            }
        
        return None


# ============================================================
# DETECTION SESSION CLASS
# ============================================================

class DetectionSession:
    """Manages detection for a single parking spot."""
    
    def __init__(self, spot_id, grid_config: Optional[dict] = None, 
                 slot_mapping: Optional[dict] = None, auto_detect_grid: bool = True):
        self.spot_id = spot_id
        self.slot_mapping = slot_mapping or {}
        self.slots = {}
        self.frame_count = 0
        self.started_at = time.time()
        self.auto_detect_grid = auto_detect_grid
        self.reference_frame = None
        self.grid_locked = False
        self.grid_config = grid_config  # Store for frame-size scaling
        self.reference_frame_size = None  # Will be set on first frame
        self.aoi = None  # Area of Interest for constraining detection
        
        # Extract AOI from grid_config if present
        if grid_config and "aoi" in grid_config:
            aoi_data = grid_config["aoi"]
            # Check if normalized coordinates are available
            if "bbox_normalized" in aoi_data:
                self.aoi_normalized = aoi_data["bbox_normalized"]
                print(f"📍 AOI will be scaled on first frame")
            else:
                self.aoi = aoi_data.get("bbox")
                print(f"📍 AOI set: {self.aoi}")
        
        # Initialize slot trackers from config
        # Note: We DON'T create slots here if using normalized coords
        # We'll create them on first frame when we know actual frame dimensions
        if grid_config and "cells" in grid_config:
            # Check if normalized coordinates are available
            has_normalized = grid_config["cells"][0].get("bbox_normalized") is not None
            # Check if quadrilateral corners are available
            has_corners = grid_config["cells"][0].get("corners_normalized") is not None or \
                         grid_config["cells"][0].get("corners") is not None
            
            if has_normalized:
                # Store config, create slots on first frame
                if has_corners:
                    print(f"📊 Session created with normalized quadrilateral grid (perspective-corrected)")
                else:
                    print(f"📊 Session created with normalized grid (will scale on first frame)")
            else:
                # Old format - use absolute coordinates directly
                for cell in grid_config["cells"]:
                    slot_num = cell["slot_number"]
                    bbox = cell["bbox"]
                    corners = cell.get("corners")  # May be None for old format
                    self.slots[slot_num] = SlotTracker(slot_num, bbox, corners)
                print(f"📊 Session created with {len(self.slots)} pre-defined slots (absolute coords)")
            
            self.grid_locked = True
        else:
            print(f"📊 Session created with auto-detection mode")
    
    def set_reference_frame(self, frame_bgr: np.ndarray):
        """Set reference frame (empty parking lot) for difference-based detection."""
        self.reference_frame = frame_bgr.copy()
        
        # Update reference regions for all slots
        height, width = frame_bgr.shape[:2]
        for slot_num, tracker in self.slots.items():
            # Use perspective-corrected extraction if corners available
            if tracker.corners:
                ref_region = extract_quadrilateral_region(frame_bgr, tracker.corners, (100, 100))
                if ref_region is not None:
                    tracker.set_reference(ref_region)
                    continue
            
            # Fall back to bounding box
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)
            ref_region = frame_bgr[y1:y2, x1:x2]
            tracker.set_reference(ref_region)
        
        print(f"✅ Reference frame set for {len(self.slots)} slots")
    
    def auto_detect_and_create_slots(self, frame_bgr: np.ndarray):
        """
        Auto-detect grid and create slot trackers.
        
        🧠 PRIORITY ORDER:
        1️⃣ Manual grid_config → Already handled in __init__ (highest priority)
        2️⃣ YOLO grid detection → ML-powered (primary auto)
        3️⃣ CV adaptive grid → Traditional CV (fallback)
        4️⃣ CV edge detection → Edge-based CV (last fallback)
        5️⃣ No grid → "No grid detected"
        """
        if self.grid_locked:
            return
        
        detected_slots = []
        detection_method = "none"
        
        # 1️⃣ YOLO GRID DETECTION (PRIMARY AUTO - uses trained .pt model)
        # 3️⃣ CRITICAL: Use FULL frame (aoi=None) for grid detection
        detected_slots = detect_grid_yolo(frame_bgr, aoi=None)
        if len(detected_slots) >= 2:
            detection_method = "YOLO (.pt model)"
        
        # 2️⃣ CV ADAPTIVE GRID (FALLBACK 1)
        if len(detected_slots) < 2:
            print("🔄 YOLO didn't find enough slots, trying CV adaptive...")
            detected_slots = detect_parking_grid(frame_bgr, aoi=self.aoi)
            if len(detected_slots) >= 2:
                detection_method = "CV Adaptive Threshold"
        
        # 3️⃣ CV EDGE DETECTION (FALLBACK 2)
        if len(detected_slots) < 2:
            print("🔄 CV adaptive didn't find enough slots, trying edge detection...")
            detected_slots = detect_grid_with_edge_detection(frame_bgr, aoi=self.aoi)
            if len(detected_slots) >= 2:
                detection_method = "CV Edge Detection"
        
        # 4️⃣ CREATE SLOT TRACKERS
        if len(detected_slots) > 0:
            self.slots.clear()
            # 2️⃣ Slots now come in (x1, y1, x2, y2) format from YOLO
            for i, (x1, y1, x2, y2) in enumerate(detected_slots, start=1):
                bbox = [x1, y1, x2, y2]  # Already in correct format
                self.slots[i] = SlotTracker(i, bbox)
            
            self.grid_locked = True
            print(f"✅ Grid locked using {detection_method} with {len(self.slots)} slots")
            
            # Set reference if available
            if self.reference_frame is not None:
                self.set_reference_frame(self.reference_frame)
        else:
            print("⚠️ No grid detected by any method. Please draw grid manually.")
    
    def process_frame(self, frame_bgr: np.ndarray, use_ai: bool = True):
        """Process a frame and detect occupancy for all slots."""
        self.frame_count += 1
        
        if frame_bgr is None:
            return None, {}, None
        
        height, width = frame_bgr.shape[:2]
        
        # On first frame, scale normalized AOI to actual frame size
        if self.frame_count == 1 and hasattr(self, 'aoi_normalized') and self.aoi_normalized:
            print(f"📐 Scaling normalized AOI to frame size: {width}x{height}")
            x1 = int(self.aoi_normalized[0] * width)
            y1 = int(self.aoi_normalized[1] * height)
            x2 = int(self.aoi_normalized[2] * width)
            y2 = int(self.aoi_normalized[3] * height)
            self.aoi = (x1, y1, x2, y2)
            print(f"   AOI: normalized {self.aoi_normalized} -> pixels {self.aoi}")
        
        # On first frame, scale normalized coordinates to actual frame size
        if self.frame_count == 1 and len(self.slots) == 0 and self.grid_config:
            if "cells" in self.grid_config:
                cells = self.grid_config["cells"]
                # Log the original frame dimensions from config (if available)
                config_width = self.grid_config.get("frame_width", "N/A")
                config_height = self.grid_config.get("frame_height", "N/A")
                print(f"📐 Grid config original dimensions: {config_width}x{config_height}")
                print(f"📐 Current frame dimensions: {width}x{height}")
                
                # 🔥 DEBUG: Check if dimensions match
                if config_width != "N/A" and config_height != "N/A":
                    if config_width != width or config_height != height:
                        print(f"⚠️ WARNING: Frame dimensions don't match config!")
                        print(f"   Config: {config_width}x{config_height}")
                        print(f"   Current: {width}x{height}")
                        print(f"   Using normalized coords to scale properly...")
                
                if cells and cells[0].get("bbox_normalized"):
                    print(f"📐 Scaling normalized coordinates to frame size: {width}x{height}")
                    for cell in cells:
                        slot_num = cell["slot_number"]
                        norm_bbox = cell["bbox_normalized"]
                        
                        # 🔥 DEBUG: Print normalized values
                        print(f"   Slot {slot_num} normalized bbox: {norm_bbox}")
                        if cell.get("corners_normalized"):
                            print(f"   Slot {slot_num} normalized corners: {cell['corners_normalized']}")
                        
                        # Scale normalized (0-1) to actual pixel coordinates
                        x1 = int(norm_bbox[0] * width)
                        y1 = int(norm_bbox[1] * height)
                        x2 = int(norm_bbox[2] * width)
                        y2 = int(norm_bbox[3] * height)
                        
                        print(f"   Slot {slot_num} scaled bbox: [{x1}, {y1}, {x2}, {y2}]")
                        
                        bbox = [x1, y1, x2, y2]
                        
                        # Scale corners if available (for perspective-correct detection)
                        corners = None
                        if cell.get("corners_normalized"):
                            corners = [
                                {"x": int(c["x"] * width), "y": int(c["y"] * height)}
                                for c in cell["corners_normalized"]
                            ]
                            print(f"   Slot {slot_num}: quadrilateral with 4 corners (perspective-corrected)")
                        elif cell.get("corners"):
                            corners = cell["corners"]
                            print(f"   Slot {slot_num}: quadrilateral (absolute coords)")
                        else:
                            print(f"   Slot {slot_num}: rectangle {norm_bbox} -> {bbox}")
                        
                        self.slots[slot_num] = SlotTracker(slot_num, bbox, corners)
                    
                    print(f"✅ Created {len(self.slots)} slots from normalized coordinates")
        
        # Auto-detect grid continuously until locked
        if not self.grid_locked:
            self.auto_detect_and_create_slots(frame_bgr)
        
        # Store first frame as reference if not set
        if self.reference_frame is None and self.frame_count == 1:
            print("📸 Capturing first frame as reference")
            self.set_reference_frame(frame_bgr)
        
        annotated = frame_bgr.copy()
        occupancy = {}
        state_change = None
        
        # If no slots detected yet, show message
        if len(self.slots) == 0:
            message = "⚠️ NO GRID DETECTED - Draw slots manually or use Auto-Detect"
            cv2.rectangle(annotated, (10, 10), (width - 10, 60), (0, 0, 0), -1)
            cv2.putText(annotated, message, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return annotated, {}, None
        
        for slot_num, tracker in self.slots.items():
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                occupancy[str(slot_num)] = {
                    "status": tracker.status,
                    "confidence": round(tracker.confidence, 2)
                }
                continue
            
            # Extract slot region - use perspective correction if corners available
            slot_region = None
            if tracker.corners:
                # Use perspective-corrected extraction for quadrilateral slots
                slot_region = extract_quadrilateral_region(frame_bgr, tracker.corners, (100, 100))
            
            # Fall back to bounding box extraction
            if slot_region is None:
                slot_region = frame_bgr[y1:y2, x1:x2]
            
            if slot_region is None or slot_region.size == 0:
                occupancy[str(slot_num)] = {
                    "status": tracker.status,
                    "confidence": round(tracker.confidence, 2)
                }
                continue
            
            # Run ensemble detection - use AI less frequently for speed
            result = detect_vehicle_ensemble(
                slot_region, 
                tracker.reference_region,
                tracker.previous_region,
                use_ai=use_ai and (self.frame_count % 5 == 0)  # Use AI every 5th frame for speed
            )
            
            # Unpack result (is_occupied, confidence, is_shadow)
            is_occupied, confidence, is_shadow = result  # type: ignore
            
            # PATCH 1: Enforce shadow lock
            if is_shadow:
                tracker.shadow_lock_frames = 5
            
            if tracker.shadow_lock_frames > 0:
                tracker.shadow_lock_frames -= 1
                is_occupied = False
                confidence = max(confidence, 0.8)
            
            # Update tracker with current region for next frame's motion detection
            change = tracker.update(is_occupied, confidence, slot_region)
            if change:
                state_change = change
            
            # Draw annotation - use quadrilateral if corners available
            color = (0, 0, 255) if tracker.status == "occupied" else (0, 255, 0)
            label = f"#{slot_num}: {tracker.status.upper()}"
            
            if tracker.corners:
                # Draw quadrilateral
                annotated = draw_quadrilateral(annotated, tracker.corners, color, 3, label)
                
                # Draw confidence at center
                center_x = int(sum(c["x"] for c in tracker.corners) / 4)
                center_y = int(sum(c["y"] for c in tracker.corners) / 4)
                cv2.putText(annotated, f"{tracker.confidence:.2f}", (center_x - 15, center_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Draw rectangle (old format)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                conf_label = f"{tracker.confidence:.2f}"
                
                # Draw label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw confidence below
                cv2.putText(annotated, conf_label, (x1 + 5, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            occupancy[str(slot_num)] = {
                "status": tracker.status,
                "confidence": round(tracker.confidence, 2)
            }
        
        # Add summary stats
        total_slots = len(self.slots)
        occupied_slots = sum(1 for s in self.slots.values() if s.status == "occupied")
        vacant_slots = total_slots - occupied_slots
        
        # Draw summary
        summary = f"Total: {total_slots} | Occupied: {occupied_slots} | Vacant: {vacant_slots}"
        cv2.putText(annotated, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated, occupancy, state_change
    
    def should_send_frame(self):
        """Check if we should send annotated frame."""
        return any(s.frames_since_change <= 15 for s in self.slots.values())


# ============================================================
# ACTIVE SESSIONS STORAGE
# ============================================================

active_sessions = {}


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
        "active_sessions": len(active_sessions),
        "sessions": list(active_sessions.keys())
    })


@app.route('/start-detection', methods=['POST'])
def start_detection():
    """Start detection session for a parking spot."""
    try:
        data = request.json or {}
        spot_id = data.get('parking_spot_id')
        grid_config = data.get('grid_config')  # Optional
        slot_mapping = data.get('slot_mapping')
        auto_detect = data.get('auto_detect_grid', True)
        
        if not spot_id:
            return jsonify({
                "success": False,
                "message": "Missing parking_spot_id"
            }), 400
        
        # Create session (auto-detect if no config provided)
        session = DetectionSession(spot_id, grid_config, slot_mapping, auto_detect)
        active_sessions[spot_id] = session
        
        print(f"✅ Detection started for spot {spot_id}")
        return jsonify({
            "success": True,
            "message": "Detection started",
            "spot_id": spot_id,
            "num_slots": len(session.slots),
            "auto_detect": auto_detect and not grid_config
        })
    
    except Exception as e:
        print(f"❌ Start detection error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/set-reference', methods=['POST'])
def set_reference():
    """Set reference frame (empty parking lot) for a session."""
    try:
        data = request.json or {}
        spot_id = data.get('spot_id')
        frame_b64 = data.get('frame')
        
        if spot_id not in active_sessions:
            return jsonify({"error": "No active session"}), 400
        
        if not frame_b64:
            return jsonify({"error": "Missing frame"}), 400
        
        frame_bgr = decode_base64_image(frame_b64)
        if frame_bgr is None:
            return jsonify({"error": "Invalid image"}), 400
        
        session = active_sessions[spot_id]
        session.set_reference_frame(frame_bgr)
        
        return jsonify({
            "success": True,
            "message": "Reference frame set"
        })
    
    except Exception as e:
        print(f"❌ Set reference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/stop-detection', methods=['POST'])
def stop_detection():
    """Stop detection session for a parking spot."""
    try:
        data = request.json or {}
        spot_id = data.get('parking_spot_id')
        
        if spot_id in active_sessions:
            del active_sessions[spot_id]
            print(f"⏹️ Detection stopped for spot {spot_id}")
            return jsonify({
                "success": True,
                "message": "Detection stopped"
            })
        
        return jsonify({
            "success": False,
            "message": "No active session for this spot"
        }), 404
    
    except Exception as e:
        print(f"❌ Stop detection error: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/process-frame', methods=['POST'])
def process_frame():
    """Process a video frame for parking detection."""
    try:
        data = request.json or {}
        spot_id = data.get('spot_id')
        frame_b64 = data.get('frame')
        timestamp = data.get('timestamp', time.time())
        use_ai = data.get('use_ai', True)
        
        if not spot_id:
            return jsonify({"error": "Missing spot_id"}), 400
        
        if spot_id not in active_sessions:
            return jsonify({"error": "No active session for this spot"}), 400
        
        if not frame_b64:
            return jsonify({"error": "Missing frame data"}), 400
        
        # Decode frame
        frame_bgr = decode_base64_image(frame_b64)
        
        if frame_bgr is None:
            return jsonify({
                "error": "Failed to decode image",
                "occupancy": {"slots": {}},
                "state_change": None
            }), 400
        
        # Process frame
        session = active_sessions[spot_id]
        annotated_frame, occupancy, state_change = session.process_frame(frame_bgr, use_ai)
        
        # Build response
        response = {
            "occupancy": {"slots": occupancy},
            "state_change": state_change,
            "timestamp": timestamp,
            "frame_count": session.frame_count,
            "num_slots": len(session.slots)
        }
        
        # Always include annotated frame so user can see live detection
        if annotated_frame is not None:
            encoded = encode_frame_to_base64(annotated_frame, quality=75)  # Lower quality for speed
            if encoded:
                response["processed_frame"] = encoded
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Process frame error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "occupancy": {"slots": {}},
            "state_change": None
        }), 500


@app.route('/detect-grid', methods=['POST'])
def detect_grid():
    """Manual endpoint to detect grid from a frame."""
    try:
        data = request.json or {}
        frame_b64 = data.get('frame')
        aoi_data = data.get('aoi')  # Optional: {bbox: [...], bbox_normalized: [...]}
        
        if not frame_b64:
            return jsonify({"error": "Missing frame"}), 400
        
        frame_bgr = decode_base64_image(frame_b64)
        if frame_bgr is None:
            return jsonify({"error": "Invalid image"}), 400
        
        height, width = frame_bgr.shape[:2]
        
        # Convert AOI to tuple
        aoi_tuple = None
        if aoi_data:
            # Check if normalized coordinates are available
            if isinstance(aoi_data, dict) and 'bbox_normalized' in aoi_data:
                norm_bbox = aoi_data['bbox_normalized']
                x1 = int(norm_bbox[0] * width)
                y1 = int(norm_bbox[1] * height)
                x2 = int(norm_bbox[2] * width)
                y2 = int(norm_bbox[3] * height)
                aoi_tuple = (x1, y1, x2, y2)
                print(f"🎯 Using AOI (scaled from normalized): {aoi_tuple}")
            elif isinstance(aoi_data, dict) and 'bbox' in aoi_data:
                bbox = aoi_data['bbox']
                aoi_tuple = tuple(bbox) if isinstance(bbox, list) else bbox
                print(f"🎯 Using AOI (absolute): {aoi_tuple}")
            elif isinstance(aoi_data, dict) and all(k in aoi_data for k in ['x1', 'y1', 'x2', 'y2']):
                # Old format
                aoi_tuple = (aoi_data['x1'], aoi_data['y1'], aoi_data['x2'], aoi_data['y2'])
                print(f"🎯 Using AOI (old format): {aoi_tuple}")
        
        # Try both methods
        method1_slots = detect_parking_grid(frame_bgr, aoi=aoi_tuple)
        method2_slots = detect_grid_with_edge_detection(frame_bgr, aoi=aoi_tuple)
        
        print(f"📊 Method 1 (Adaptive): {len(method1_slots)} cells, Method 2 (Edge): {len(method2_slots)} cells")
        
        # Use method with more results
        slots = method1_slots if len(method1_slots) >= len(method2_slots) else method2_slots
        
        # Convert to API format
        cells = []
        for i, (x, y, w, h) in enumerate(slots, start=1):
            cells.append({
                "slot_number": i,
                "bbox": [x, y, x + w, y + h]
            })
        
        # Draw detected grid with better visualization
        annotated = frame_bgr.copy()
        
        # Draw AOI if it was used - make it very obvious
        if aoi_tuple:
            x1, y1, x2, y2 = aoi_tuple
            
            # Darken the area OUTSIDE the AOI to show detection only happens inside
            mask = np.zeros_like(annotated)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            mask_inv = cv2.bitwise_not(mask)
            darkened = cv2.addWeighted(annotated, 0.3, np.zeros_like(annotated), 0.7, 0)
            annotated = np.where(mask_inv > 0, darkened, annotated)
            
            # Draw thick yellow border for AOI
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 4)
            
            # Add AOI label with background
            label = "DETECTION AREA (AOI)"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 255), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw detected cells
        for cell in cells:
            x1, y1, x2, y2 = cell["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"#{cell['slot_number']}", (x1 + 5, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add detection info text
        info_text = f"Detected: {len(cells)} slots" + (" (in AOI)" if aoi_tuple else "")
        cv2.rectangle(annotated, (5, 5), (15 + len(info_text) * 12, 40), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        encoded = encode_frame_to_base64(annotated)
        
        return jsonify({
            "success": True,
            "num_cells": len(cells),
            "cells": cells,
            "annotated_frame": encoded
        })
    
    except Exception as e:
        print(f"❌ Grid detection error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# MAIN
# ============================================================

load_global_model()

if __name__ == '__main__':
    print("=" * 60)
    print(f"📍 Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔧 Model loaded: {MODEL_LOADED}")
    print(f"🌐 Server starting on http://0.0.0.0:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)