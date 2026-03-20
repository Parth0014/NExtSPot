"""
YOLO Grid Detector Helper - Mimics test.py's exact approach
This module handles grid detection with proper coordinate translation
"""
from ultralytics import YOLO # type: ignore
import cv2
import numpy as np
import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Optional, Dict

# Global model instance (loaded once)
YOLO_MODEL = None
YOLO_LOADED = False

def load_yolo_model(model_path: str) -> bool:
    """
    Load the YOLO model once at startup.
    Args:
        model_path: Path to best.pt model file
    Returns:
        True if loaded successfully, False otherwise
    """
    global YOLO_MODEL, YOLO_LOADED
    
    if YOLO_LOADED:
        print("✅ YOLO model already loaded")
        return True
    
    if not os.path.exists(model_path):
        print(f"❌ YOLO model not found at: {model_path}")
        return False
    
    try:
        print(f"🔄 Loading YOLO model from {model_path}...")
        YOLO_MODEL = YOLO(model_path)
        YOLO_LOADED = True
        print(f"✅ YOLO model loaded successfully!")
        print(f"   Classes: {YOLO_MODEL.names if hasattr(YOLO_MODEL, 'names') else 'Unknown'}")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        return False


def decode_base64_image(frame_b64: str) -> Optional[np.ndarray]:
    """
    Decode a base64 image string to BGR numpy array.
    Args:
        frame_b64: Base64 encoded image (with or without data URL prefix)
    Returns:
        BGR numpy array or None if decoding fails
    """
    try:
        # Remove data URL prefix if present
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]
        
        # Decode base64
        img_bytes = base64.b64decode(frame_b64)
        
        if len(img_bytes) < 100:
            print(f"⚠️ Image data too small: {len(img_bytes)} bytes")
            return None
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(img_bytes))
        
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array
        rgb_array = np.array(pil_image)
        
        if rgb_array is None or rgb_array.size == 0:
            return None
        
        # Convert RGB to BGR (OpenCV format)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    
    except Exception as e:
        print(f"⚠️ Image decode error: {e}")
        return None


def encode_frame_to_base64(frame_bgr: np.ndarray, quality: int = 85) -> str:
    """
    Encode a BGR frame to base64 JPEG data URL.
    Args:
        frame_bgr: BGR numpy array
        quality: JPEG quality (0-100)
    Returns:
        Base64 data URL string
    """
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


def preprocess_for_detection(frame_bgr: np.ndarray) -> np.ndarray:
    """
    🎨 COLOR-AGNOSTIC PREPROCESSING
    
    Enhances the image to make grid lines/borders more visible regardless of color.
    Works with red, blue, black, or any color ink/marker.
    
    Techniques:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
    2. Edge enhancement to make borders stand out
    3. Saturation boost to make colored lines more visible
    """
    try:
        # Make a copy to avoid modifying original
        enhanced = frame_bgr.copy()
        
        # Method 1: Apply CLAHE to L channel in LAB color space
        # This enhances local contrast without color distortion
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Method 2: Boost saturation to make colored lines pop
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Boost saturation (helps colored ink stand out)
        s_boosted = cv2.convertScaleAbs(s, alpha=1.3, beta=10)
        
        hsv_enhanced = cv2.merge([h, s_boosted, v])
        enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Method 3: Slight sharpening to enhance edges
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        print("🎨 Applied color-agnostic preprocessing (CLAHE + saturation boost + sharpening)")
        return enhanced
        
    except Exception as e:
        print(f"⚠️ Preprocessing error (using original): {e}")
        return frame_bgr


def detect_grid_static_approach(
    frame_bgr: np.ndarray,
    aoi_absolute: Optional[Tuple[int, int, int, int]] = None,
    conf_thresh: float = 0.15,  # Lowered default threshold for better recall
    imgsz: int = 640
) -> Dict:
    """
    🔥 STATIC APPROACH - Exactly like test.py
    
    This function:
    1. Crops AOI from frame (if AOI provided)
    2. Applies color-agnostic preprocessing for better detection
    3. Saves cropped region to temp file (just like test.py)
    4. Runs YOLO on temp file (identical to test.py)
    5. Translates coordinates back to full frame space
    
    Args:
        frame_bgr: Full frame in BGR format
        aoi_absolute: AOI coordinates in full frame space (x1, y1, x2, y2) or None
        conf_thresh: YOLO confidence threshold
        imgsz: YOLO image size
        
    Returns:
        Dictionary with:
        - success: bool
        - num_cells: int
        - cells: List of grid cells with coordinates in FULL FRAME space
        - cells_in_aoi: List of grid cells with coordinates in AOI space
        - annotated_frame: Base64 encoded annotated image
        - aoi_used: AOI coordinates if provided
    """
    global YOLO_MODEL, YOLO_LOADED
    
    if not YOLO_LOADED or YOLO_MODEL is None:
        return {
            "success": False,
            "message": "YOLO model not loaded",
            "num_cells": 0,
            "cells": [],
            "cells_in_aoi": []
        }
    
    try:
        # Step 1: Crop AOI if provided
        offset_x, offset_y = 0, 0
        detection_frame = frame_bgr
        
        if aoi_absolute:
            x1, y1, x2, y2 = aoi_absolute
            # Ensure coordinates are within frame bounds
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Crop the AOI region
            detection_frame = frame_bgr[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
            
            print(f"📍 AOI cropped: ({x1}, {y1}) to ({x2}, {y2})")
            print(f"📐 AOI size: {detection_frame.shape[1]}x{detection_frame.shape[0]}")
        else:
            print(f"📐 Using full frame: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")
        
        # Step 2: Apply color-agnostic preprocessing for better detection
        detection_frame = preprocess_for_detection(detection_frame)
        
        # Step 3: Save to temp file (EXACTLY like test.py)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "yolo_grid_detect_frame.png")
        cv2.imwrite(temp_path, detection_frame)
        print(f"📸 Saved preprocessed frame to: {temp_path}")
        
        # Step 4: Run YOLO predict on file (EXACTLY like test.py)
        # Lower confidence threshold for better recall
        effective_conf = max(0.10, conf_thresh - 0.05)  # Slightly lower threshold
        print(f"🔍 Running YOLO detection (conf={effective_conf}, imgsz={imgsz})...")
        results = YOLO_MODEL.predict(
            source=temp_path,
            imgsz=imgsz,
            conf=effective_conf,
            verbose=False,
            save=False
        )
        
        # Step 5: Extract detections and translate coordinates
        cells_in_aoi = []  # Coordinates relative to AOI
        cells_full_frame = []  # Coordinates relative to full frame
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            print(f"📦 YOLO detected {len(results[0].boxes)} raw boxes")
            
            for i, box in enumerate(results[0].boxes):
                # Get coordinates in AOI space (or full frame if no AOI)
                x1_aoi, y1_aoi, x2_aoi, y2_aoi = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                w = x2_aoi - x1_aoi
                h = y2_aoi - y1_aoi
                
                print(f"   Box {i+1}: AOI coords ({x1_aoi:.0f},{y1_aoi:.0f}) -> ({x2_aoi:.0f},{y2_aoi:.0f}), conf={conf:.2f}, size={w:.0f}x{h:.0f}")
                
                # Basic sanity filter - reject tiny detections
                if w < 30 or h < 30:
                    print(f"   ❌ Rejected: too small")
                    continue
                
                # Store coordinates in AOI space
                cells_in_aoi.append({
                    "slot_number": len(cells_in_aoi) + 1,
                    "bbox": [int(x1_aoi), int(y1_aoi), int(x2_aoi), int(y2_aoi)],
                    "confidence": round(conf, 3)
                })
                
                # Translate to full frame space
                x1_full = int(x1_aoi + offset_x)
                y1_full = int(y1_aoi + offset_y)
                x2_full = int(x2_aoi + offset_x)
                y2_full = int(y2_aoi + offset_y)
                
                cells_full_frame.append({
                    "slot_number": len(cells_full_frame) + 1,
                    "bbox": [x1_full, y1_full, x2_full, y2_full],
                    "confidence": round(conf, 3)
                })
                
                print(f"   ✅ Full frame coords: ({x1_full}, {y1_full}) -> ({x2_full}, {y2_full})")
        
        # Step 5: Create annotated frame
        annotated = frame_bgr.copy()
        
        # Draw AOI border (if used)
        if aoi_absolute:
            x1, y1, x2, y2 = aoi_absolute
            # Darken area outside AOI
            mask = np.zeros_like(annotated)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
            mask_inv = cv2.bitwise_not(mask)
            darkened = cv2.addWeighted(annotated, 0.3, np.zeros_like(annotated), 0.7, 0)
            annotated = np.where(mask_inv > 0, darkened, annotated)
            
            # Draw thick yellow border for AOI
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 4)
            
            # Add AOI label
            label = "AOI (Area of Interest)"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 255), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw detected grids (using full frame coordinates)
        for cell in cells_full_frame:
            x1, y1, x2, y2 = cell["bbox"]
            slot_num = cell["slot_number"]
            conf = cell["confidence"]
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw label with background
            label = f"#{slot_num} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add summary info
        info_text = f"YOLO Detected: {len(cells_full_frame)} slots" + (" in AOI" if aoi_absolute else "")
        cv2.rectangle(annotated, (5, 5), (15 + len(info_text) * 12, 40), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Step 6: Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Step 7: Return results
        print(f"✅ Detection complete: {len(cells_full_frame)} slots found")
        
        return {
            "success": True,
            "num_cells": len(cells_full_frame),
            "cells": cells_full_frame,  # Full frame coordinates
            "cells_in_aoi": cells_in_aoi,  # AOI-relative coordinates (for debugging)
            "annotated_frame": encode_frame_to_base64(annotated),
            "aoi_used": aoi_absolute,
            "message": f"Detected {len(cells_full_frame)} parking slots"
        }
    
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "num_cells": 0,
            "cells": [],
            "cells_in_aoi": []
        }


def calculate_normalized_coordinates(
    cells: List[Dict],
    frame_width: int,
    frame_height: int
) -> List[Dict]:
    """
    Calculate normalized coordinates (0-1 range) for cells.
    This allows coordinates to scale with different frame sizes.
    
    Args:
        cells: List of cells with absolute coordinates
        frame_width: Width of the frame
        frame_height: Height of the frame
        
    Returns:
        List of cells with normalized coordinates added
    """
    normalized_cells = []
    
    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        
        normalized_cell = {
            **cell,  # Copy all existing fields
            "bbox_normalized": [
                x1 / frame_width,
                y1 / frame_height,
                x2 / frame_width,
                y2 / frame_height
            ]
        }
        
        normalized_cells.append(normalized_cell)
    
    return normalized_cells


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("YOLO Grid Detector - Test Mode")
    print("="*60)
    
    # Path to your trained model
    MODEL_PATH = "best.pt"  # Update this path
    
    # Load model
    if not load_yolo_model(MODEL_PATH):
        print("❌ Failed to load model. Update MODEL_PATH and try again.")
        exit(1)
    
    # Test image path
    TEST_IMAGE = "test_frame.jpg"  # Update this path
    
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Test image not found: {TEST_IMAGE}")
        exit(1)
    
    # Load test image
    frame = cv2.imread(TEST_IMAGE)
    h, w = frame.shape[:2]
    print(f"📸 Loaded test image: {w}x{h}")
    
    # Test 1: Full frame detection
    print("\n" + "="*60)
    print("Test 1: Full Frame Detection")
    print("="*60)
    result1 = detect_grid_static_approach(frame, aoi_absolute=None)
    print(f"✅ Detected {result1['num_cells']} slots")
    
    # Save annotated image
    if result1["annotated_frame"]:
        annotated_b64 = result1["annotated_frame"].split(",")[1]
        annotated_data = base64.b64decode(annotated_b64)
        with open("test_output_fullframe.jpg", "wb") as f:
            f.write(annotated_data)
        print("💾 Saved: test_output_fullframe.jpg")
    
    # Test 2: AOI detection
    print("\n" + "="*60)
    print("Test 2: AOI Detection (center region)")
    print("="*60)
    # Define AOI as center 50% of frame
    aoi = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))
    result2 = detect_grid_static_approach(frame, aoi_absolute=aoi)
    print(f"✅ Detected {result2['num_cells']} slots in AOI")
    
    # Save annotated image
    if result2["annotated_frame"]:
        annotated_b64 = result2["annotated_frame"].split(",")[1]
        annotated_data = base64.b64decode(annotated_b64)
        with open("test_output_aoi.jpg", "wb") as f:
            f.write(annotated_data)
        print("💾 Saved: test_output_aoi.jpg")
    
    # Test 3: Normalized coordinates
    print("\n" + "="*60)
    print("Test 3: Normalized Coordinates")
    print("="*60)
    normalized = calculate_normalized_coordinates(result1["cells"], w, h)
    print(f"✅ Generated normalized coordinates for {len(normalized)} cells")
    for cell in normalized[:3]:  # Show first 3
        print(f"   Slot {cell['slot_number']}: {cell['bbox_normalized']}")
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60)