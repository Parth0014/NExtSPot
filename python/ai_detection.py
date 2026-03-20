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
from concurrent.futures import ThreadPoolExecutor

# Import our new YOLO grid detector
from yolo_grid_detector import (
    load_yolo_model,
    detect_grid_static_approach,
    decode_base64_image,
    encode_frame_to_base64,
    calculate_normalized_coordinates
)

app = Flask(__name__)
CORS(app)

# Thread pool for parallel slot processing
SLOT_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ============================================================
# GLOBAL MODELS - Load ONCE at startup for efficiency
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
MODEL = None
PROCESSOR = None
MODEL_LOADED = False

# YOLO model path - UPDATE THIS PATH
YOLO_MODEL_PATH = r"C:\Users\jaypa\OneDrive\Desktop\shape_training\runs\detect\train4\weights\best.pt"


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
        MODEL_LOADED = True
        
        # Load YOLO model for grid detection
        print("\n" + "=" * 60)
        print("🎯 Loading Custom YOLO Model (best.pt)")
        print("=" * 60)
        load_yolo_model(YOLO_MODEL_PATH)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        traceback.print_exc()
        return False


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
        
        # Structural white-object detection using edges
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # If significant hard edges exist, it's a real object (not shadow)
        if edge_density > 0.04:
            return True, 0.75
        
        # SHADOW DETECTION
        is_likely_shadow = (mean_saturation < 25 and 40 < mean_value < 160)
        
        if is_likely_shadow:
            return False, 0.6
        
        # Check for color presence
        if mean_saturation > 45 and std_saturation > 15:
            return True, 0.7
        
        # Check for dark objects
        if mean_value < 60:
            return True, 0.6
        
        return False, 0.5
    
    except Exception as e:
        print(f"⚠️ Color detection error: {e}")
        return False, 0.5


def detect_vehicle_texture_based(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Detect vehicle based on texture/edge density.
    SHADOW-ROBUST: Filters out soft shadow edges, focuses on hard object edges.
    """
    try:
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        edges_strong = cv2.Canny(gray_blurred, 90, 220)
        edges_weak = cv2.Canny(gray_blurred, 20, 60)
        
        strong_edge_density = np.sum(edges_strong > 0) / edges_strong.size
        weak_edge_density = np.sum(edges_weak > 0) / edges_weak.size
        
        edge_ratio = strong_edge_density / (weak_edge_density + 0.001)
        
        laplacian = cv2.Laplacian(gray_blurred, cv2.CV_64F)
        texture_var = float(np.var(laplacian))  # type: ignore
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_std = float(np.std(gradient_magnitude))  # type: ignore
        
        # Circular object detection
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
            return True, 0.75
        
        is_likely_shadow = (edge_ratio < 0.3 and gradient_std < 30)
        
        if is_likely_shadow:
            return False, 0.55
        
        if strong_edge_density > 0.06 or (texture_var > 200 and gradient_std > 35):
            return True, 0.65
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            total_area = gray.shape[0] * gray.shape[1]
            if 0.15 < area / total_area < 0.9:
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
    """
    if reference_region_bgr is None:
        return False, 0.5
    
    try:
        if slot_region_bgr.shape != reference_region_bgr.shape:
            reference_region_bgr = cv2.resize(reference_region_bgr, 
                                             (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))
        
        diff_bgr = cv2.absdiff(slot_region_bgr, reference_region_bgr)
        gray_diff = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)
        
        hsv_current = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        hsv_reference = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2HSV)
        
        hue_diff = cv2.absdiff(hsv_current[:, :, 0], hsv_reference[:, :, 0])
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        
        sat_diff = cv2.absdiff(hsv_current[:, :, 1], hsv_reference[:, :, 1])
        val_diff = cv2.absdiff(hsv_current[:, :, 2], hsv_reference[:, :, 2])
        
        mean_hue_diff = float(np.mean(hue_diff))  # type: ignore
        mean_sat_diff = float(np.mean(sat_diff))  # type: ignore
        mean_val_diff = float(np.mean(val_diff))  # type: ignore
        
        is_likely_shadow = (
            mean_val_diff > 30 and
            mean_hue_diff < 15 and
            mean_sat_diff < 40
        )
        
        gray_current = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2GRAY)
        
        edges_current = cv2.Canny(gray_current, 30, 100)
        edges_reference = cv2.Canny(gray_reference, 30, 100)
        
        edge_diff = cv2.absdiff(edges_current, edges_reference)
        new_edges_ratio = np.sum(edge_diff > 0) / edge_diff.size
        
        current_edge_density = np.sum(edges_current > 0) / edges_current.size
        reference_edge_density = np.sum(edges_reference > 0) / edges_reference.size
        edge_increase = current_edge_density - reference_edge_density
        
        if new_edges_ratio > 0.03 and edge_increase > 0.02:
            return True, 0.7
        
        if is_likely_shadow:
            return False, 0.65
        
        combined_hs_diff = (hue_diff.astype(np.float32) * 2 + sat_diff.astype(np.float32)) / 3
        _, thresh_hs = cv2.threshold(combined_hs_diff.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
        _, thresh_bgr = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        
        hs_change_ratio = np.sum(thresh_hs > 0) / thresh_hs.size
        bgr_change_ratio = np.sum(thresh_bgr > 0) / thresh_bgr.size
        
        if hs_change_ratio > 0.12 and bgr_change_ratio > 0.15:
            return True, min(0.95, 0.75 + hs_change_ratio)
        elif hs_change_ratio > 0.20:
            return True, min(0.90, 0.7 + hs_change_ratio)
        elif bgr_change_ratio > 0.35 and hs_change_ratio > 0.1:
            return True, 0.65
        
        std_current = float(np.std(gray_current))  # type: ignore
        std_reference = float(np.std(gray_reference))  # type: ignore
        mean_current = float(np.mean(gray_current))  # type: ignore
        mean_reference = float(np.mean(gray_reference))  # type: ignore
        
        if mean_current > mean_reference + 20 and std_current < std_reference - 10:
            return True, 0.65
        
        return False, 0.7
    
    except Exception as e:
        print(f"⚠️ Difference detection error: {e}")
        return False, 0.5


def detect_shadow(slot_region_bgr: np.ndarray, 
                  reference_region_bgr: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Dedicated shadow detection to filter out false positives.
    Returns: (is_shadow, confidence)
    """
    try:
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        
        mean_sat = float(np.mean(hsv[:, :, 1]))  # type: ignore
        mean_val = float(np.mean(hsv[:, :, 2]))  # type: ignore
        std_val = float(np.std(hsv[:, :, 2]))  # type: ignore
        
        sat_check = mean_sat < 30
        val_check = 40 < mean_val < 170
        uniformity_check = std_val < 35
        
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        edges_strong = cv2.Canny(gray, 90, 220)
        edges_weak = cv2.Canny(gray, 30, 80)
        
        strong_edge_ratio = np.sum(edges_strong > 0) / edges_strong.size
        weak_edge_ratio = np.sum(edges_weak > 0) / edges_weak.size
        
        soft_edge_check = (weak_edge_ratio > 0.02 and strong_edge_ratio < 0.03)
        
        hue_unchanged = False
        if reference_region_bgr is not None:
            try:
                if slot_region_bgr.shape != reference_region_bgr.shape:
                    reference_region_bgr = cv2.resize(reference_region_bgr, 
                                                     (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))
                
                hsv_ref = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2HSV)
                
                hue_diff = cv2.absdiff(hsv[:, :, 0], hsv_ref[:, :, 0])
                hue_diff = np.minimum(hue_diff, 180 - hue_diff)
                mean_hue_diff = float(np.mean(hue_diff))  # type: ignore
                
                val_diff = cv2.absdiff(hsv[:, :, 2], hsv_ref[:, :, 2])
                mean_val_diff = float(np.mean(val_diff))  # type: ignore
                
                hue_unchanged = (mean_hue_diff < 12 and mean_val_diff > 25)
            except:
                pass
        
        shadow_score = sum([
            sat_check,
            val_check,
            uniformity_check,
            soft_edge_check,
            hue_unchanged
        ])
        
        is_shadow = bool(shadow_score >= 3)
        confidence = min(0.95, 0.5 + shadow_score * 0.1)
        
        return is_shadow, confidence
    
    except Exception as e:
        return False, 0.5


def detect_rapid_motion(current_region_bgr: np.ndarray, 
                        previous_region_bgr: Optional[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect rapid motion that indicates a hand or moving object passing through.
    Returns: (is_rapid_motion, motion_level)
    """
    if previous_region_bgr is None:
        return False, 0.0
    
    try:
        if current_region_bgr.shape != previous_region_bgr.shape:
            return False, 0.0
        
        # Convert to grayscale
        gray_current = cv2.cvtColor(current_region_bgr, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_region_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray_current, gray_previous)
        motion_level = float(np.mean(diff))
        
        # Calculate motion distribution (hands have large connected motion areas)
        _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_ratio = np.sum(motion_mask > 0) / motion_mask.size
        
        # Rapid motion detection:
        # - High motion level (pixels changing significantly)
        # - Large area affected (hand covers significant portion)
        is_rapid = motion_level > 35.0 and motion_ratio > 0.25
        
        return is_rapid, motion_level
    except:
        return False, 0.0


def detect_skin_tone(region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Detect if region contains skin tones (likely a hand).
    Returns: (has_skin, skin_ratio)
    """
    try:
        # Convert to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2YCrCb)
        
        # Skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create mask for skin pixels
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Calculate ratio of skin pixels
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        
        # If significant skin tone present, likely a hand
        has_skin = skin_ratio > 0.15
        
        return has_skin, skin_ratio
    except:
        return False, 0.0


def detect_vehicle_ensemble(slot_region_bgr: np.ndarray, 
                           reference_region_bgr: Optional[np.ndarray] = None,
                           previous_region_bgr: Optional[np.ndarray] = None,
                           use_ai: bool = True) -> Tuple[bool, float, bool]:
    """
    🚀 OPTIMIZED FAST DETECTION - Single pass, minimal conversions
    Target: 30+ FPS by reducing redundant operations
    Returns: (is_occupied, confidence, is_shadow)
    """
    try:
        # === SINGLE COLOR CONVERSION (do once, reuse) ===
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # === FAST STATISTICS (vectorized) ===
        mean_sat = float(np.mean(s))
        mean_val = float(np.mean(v))
        std_val = float(np.std(v))
        
        # === SINGLE EDGE DETECTION ===
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # === RAPID MOTION CHECK (lightweight) ===
        if previous_region_bgr is not None:
            try:
                if slot_region_bgr.shape == previous_region_bgr.shape:
                    gray_prev = cv2.cvtColor(previous_region_bgr, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray, gray_prev)
                    motion_level = float(np.mean(diff))
                    motion_ratio = np.sum(diff > 30) / diff.size
                    
                    # Rapid motion = transient object (hand)
                    if motion_level > 35.0 and motion_ratio > 0.25:
                        return False, 0.8, False
            except:
                pass
        
        # === SHADOW DETECTION (fast check) ===
        is_shadow = False
        shadow_conf = 0.0
        
        # Shadows: low saturation, uniform, few edges
        if mean_sat < 30 and std_val < 30 and edge_density < 0.02:
            is_shadow = True
            shadow_conf = 0.85
            return False, shadow_conf, True
        
        # === FAST OCCUPANCY DETECTION ===
        score = 0.0
        
        # Edge-based (objects have structure)
        if edge_density > 0.03:
            score += 0.25
        if edge_density > 0.06:
            score += 0.25
        
        # Color saturation (colored objects)
        if mean_sat > 40:
            score += 0.2
        if mean_sat > 70:
            score += 0.15
        
        # Texture variance
        if std_val > 25:
            score += 0.15
        
        # Reference comparison (if available)
        if reference_region_bgr is not None:
            try:
                if slot_region_bgr.shape == reference_region_bgr.shape:
                    ref_gray = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray, ref_gray)
                    diff_ratio = np.sum(diff > 40) / diff.size
                    
                    if diff_ratio > 0.15:
                        score += 0.3
                    if diff_ratio > 0.30:
                        score += 0.2
            except:
                pass
        
        # Dark object detection
        if mean_val < 60 and edge_density > 0.02:
            score += 0.2
        
        # White/bright object detection
        if mean_val > 200 and edge_density > 0.02:
            score += 0.2
        
        is_occupied = score >= 0.45
        confidence = min(0.95, 0.5 + score * 0.5)
        
        return is_occupied, confidence, is_shadow
        
    except Exception as e:
        return False, 0.5, False


def detect_vehicle_ensemble_full(slot_region_bgr: np.ndarray, 
                           reference_region_bgr: Optional[np.ndarray] = None,
                           previous_region_bgr: Optional[np.ndarray] = None,
                           use_ai: bool = True) -> Tuple[bool, float, bool]:
    """
    FULL Ensemble detection - used only every Nth frame for accuracy validation.
    Returns: (is_occupied, confidence, is_shadow)
    """
    
    # First check for rapid motion (hand passing through)
    is_rapid_motion, motion_level = detect_rapid_motion(slot_region_bgr, previous_region_bgr)
    if is_rapid_motion:
        return False, 0.8, False
    
    # Check if this is likely a shadow (dark area with no internal structure)
    is_shadow_region, shadow_confidence = is_likely_shadow(slot_region_bgr, reference_region_bgr)
    
    if is_shadow_region and shadow_confidence > 0.75:
        return False, shadow_confidence, True
    
    votes = []
    confidences = []
    weights = []
    
    # Method 1: Color-based detection
    is_occ_color, conf_color = detect_vehicle_color_based(slot_region_bgr)
    votes.append(is_occ_color)
    confidences.append(conf_color)
    weights.append(1.5)
    
    # Method 2: Texture-based detection
    is_occ_texture, conf_texture = detect_vehicle_texture_based(slot_region_bgr)
    votes.append(is_occ_texture)
    confidences.append(conf_texture)
    weights.append(1.5)
    
    # Method 3: Simple object detection (edges, gradients, structure)
    is_occ_simple, conf_simple = detect_object_simple(slot_region_bgr)
    votes.append(is_occ_simple)
    confidences.append(conf_simple)
    weights.append(2.0)
    
    # Method 4: Internal structure check (real objects have internal edges)
    has_structure, structure_conf = has_internal_structure(slot_region_bgr)
    votes.append(has_structure)
    confidences.append(structure_conf)
    weights.append(2.5)  # High weight - this is key for shadow rejection
    
    # If it looks like a shadow, add negative vote
    if is_shadow_region:
        votes.append(False)
        confidences.append(shadow_confidence)
        weights.append(2.0)
    
    weighted_occupied_votes = sum(w for v, w in zip(votes, weights) if v)
    weighted_vacant_votes = sum(w for v, w in zip(votes, weights) if not v)
    total_weight = sum(weights)
    
    is_occupied = weighted_occupied_votes > weighted_vacant_votes
    
    weighted_conf = sum(c * w for c, w in zip(confidences, weights)) / total_weight
    vote_agreement = max(weighted_occupied_votes, weighted_vacant_votes) / total_weight
    final_confidence = float(weighted_conf * vote_agreement)
    
    return is_occupied, final_confidence, is_shadow_region


def is_likely_shadow(slot_region_bgr: np.ndarray, 
                     reference_region_bgr: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """
    Determine if a region is likely a shadow rather than an object.
    
    Shadows have these characteristics:
    - Darker than reference but same hue
    - Low internal edge density (uniform darkness)
    - Low color saturation variation
    - No distinct internal structure
    """
    try:
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        
        # Check 1: Internal edge density (shadows have few internal edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        low_edges = edge_density < 0.02  # Shadows have very few internal edges
        
        # Check 2: Color uniformity (shadows are uniformly dark)
        val_std = float(np.std(hsv[:, :, 2]))
        sat_std = float(np.std(hsv[:, :, 1]))
        uniform_color = val_std < 30 and sat_std < 25
        
        # Check 3: Low saturation overall (shadows desaturate colors)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        low_saturation = mean_sat < 50
        
        # Check 4: If we have reference, check if only brightness changed
        hue_preserved = False
        if reference_region_bgr is not None:
            try:
                if slot_region_bgr.shape != reference_region_bgr.shape:
                    reference_region_bgr = cv2.resize(reference_region_bgr, 
                                                     (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))
                
                hsv_ref = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2HSV)
                
                # Hue difference (shadows preserve hue)
                hue_diff = cv2.absdiff(hsv[:, :, 0], hsv_ref[:, :, 0])
                hue_diff = np.minimum(hue_diff, 180 - hue_diff)
                mean_hue_diff = float(np.mean(hue_diff))
                
                # Value difference (shadows are darker)
                val_diff = float(np.mean(hsv_ref[:, :, 2])) - float(np.mean(hsv[:, :, 2]))
                
                # Shadow: hue similar, but darker
                hue_preserved = mean_hue_diff < 15 and val_diff > 20
            except:
                pass
        
        # Calculate shadow score
        shadow_indicators = sum([low_edges, uniform_color, low_saturation, hue_preserved])
        
        # Need multiple indicators to call it a shadow
        is_shadow = shadow_indicators >= 3
        confidence = min(0.95, 0.5 + shadow_indicators * 0.15)
        
        return is_shadow, confidence
        
    except Exception as e:
        return False, 0.5


def has_internal_structure(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Check if the region has internal structure (edges, texture, patterns).
    Real objects have internal detail, shadows don't.
    
    This helps detect:
    - White objects (they still have edges/structure)
    - Colored objects (they have texture)
    
    And reject:
    - Shadows (uniform darkness, no internal edges)
    """
    try:
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        
        # Edge detection at multiple scales
        edges_fine = cv2.Canny(gray, 30, 100)
        edges_coarse = cv2.Canny(gray, 50, 150)
        
        edge_density_fine = np.sum(edges_fine > 0) / edges_fine.size
        edge_density_coarse = np.sum(edges_coarse > 0) / edges_coarse.size
        
        # Gradient magnitude (texture indicator)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = float(np.mean(gradient_mag))
        max_gradient = float(np.max(gradient_mag))
        
        # Local contrast (variance in small windows)
        kernel_size = max(5, min(gray.shape) // 10)
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, (kernel_size, kernel_size))
        mean_local_var = float(np.mean(np.sqrt(local_var)))
        
        # Scoring
        score = 0
        
        # Fine edges (detail)
        if edge_density_fine > 0.03:
            score += 1
        if edge_density_fine > 0.06:
            score += 1
            
        # Coarse edges (object boundaries)
        if edge_density_coarse > 0.02:
            score += 1
            
        # Gradient (texture)
        if mean_gradient > 12:
            score += 1
        if mean_gradient > 25:
            score += 1
            
        # Strong edges somewhere (object boundary)
        if max_gradient > 100:
            score += 1
            
        # Local contrast (internal detail)
        if mean_local_var > 10:
            score += 1
        if mean_local_var > 20:
            score += 1
        
        has_structure = score >= 3
        confidence = min(0.95, 0.4 + score * 0.08)
        
        return has_structure, confidence
        
    except Exception as e:
        return False, 0.5


def detect_object_simple(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Simple object detection based on visual complexity.
    Detects if there's any significant object in the region.
    """
    try:
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color variance
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        hue_std = float(np.std(hsv[:, :, 0]))
        sat_std = float(np.std(hsv[:, :, 1]))
        val_std = float(np.std(hsv[:, :, 2]))
        
        # Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = float(np.mean(gradient_mag))
        
        # Score based on multiple factors
        score = 0
        
        # Edge density check (objects have edges)
        if edge_density > 0.03:
            score += 1
        if edge_density > 0.06:
            score += 1
            
        # Hue variation (colored objects)
        if hue_std > 15:
            score += 1
        if hue_std > 30:
            score += 1
            
        # Saturation variation
        if sat_std > 30:
            score += 1
            
        # Value variation (contrast)
        if val_std > 25:
            score += 1
            
        # Gradient (sharp features)
        if mean_gradient > 15:
            score += 1
        if mean_gradient > 30:
            score += 1
        
        # Need at least 3 indicators to consider occupied
        is_occupied = score >= 3
        confidence = min(0.95, 0.5 + score * 0.08)
        
        return is_occupied, confidence
        
    except Exception as e:
        print(f"⚠️ Simple detection error: {e}")
        return False, 0.5


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


# ============================================================
# SLOT TRACKER CLASS
# ============================================================

class SlotTracker:
    """
    Tracks a single parking slot's state with ROBUST temporal persistence.
    
    Key Features:
    - Requires sustained detection over time (not just frame count)
    - Ignores transient changes (hands, brief shadows)
    - Uses stability timer to confirm status changes
    - Tracks motion to distinguish stationary vs moving objects
    """
    
    # Configuration constants for temporal persistence
    STABILITY_TIME_SECONDS = 1.5      # Object must be stable for this long to confirm occupied (reduced from 5.0)
    VACANCY_TIME_SECONDS = 0.5        # Must be empty for this long to confirm vacant (reduced from 1.0)
    MIN_CONSECUTIVE_FRAMES = 4        # Minimum consecutive same-state detections (reduced from 8)
    MOTION_THRESHOLD = 25.0           # Pixel difference threshold for motion detection
    HISTORY_SIZE = 15                 # Larger history for better temporal smoothing
    
    def __init__(self, slot_number: int, bbox: list):
        self.slot_number = slot_number
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.status = "vacant"          # Current confirmed status
        self.pending_status = None       # Status waiting to be confirmed
        self.confidence = 0.0
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.last_change_time = time.time()
        self.frames_since_change = 0
        self.reference_region = None
        self.previous_region = None
        self.shadow_lock_frames = 0
        
        # NEW: Temporal persistence tracking
        self.pending_start_time = None   # When pending status was first detected
        self.consecutive_same_state = 0  # Count of consecutive same-state detections
        self.last_detection_state = None # Last raw detection result
        self.motion_history = deque(maxlen=5)  # Track recent motion levels
        self.stable_region = None        # Region when object became stable
    
    def set_reference(self, reference_region: np.ndarray):
        """Set the reference (empty) image for this slot."""
        self.reference_region = reference_region.copy()
    
    def _calculate_motion(self, current_region: np.ndarray) -> float:
        """
        Calculate motion level between current and previous frame.
        High motion = transient object (hand, shadow moving)
        Low motion = stationary object (parked item)
        """
        if self.previous_region is None:
            return 0.0
        
        try:
            if current_region.shape != self.previous_region.shape:
                return 0.0
            
            # Convert to grayscale for comparison
            gray_current = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(self.previous_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(gray_current, gray_previous)
            motion_level = float(np.mean(diff))
            
            return motion_level
        except:
            return 0.0
    
    def _is_object_stationary(self) -> bool:
        """
        Check if the detected object is stationary (not moving).
        Returns True if object has been stable across recent frames.
        """
        if len(self.motion_history) < 3:
            return False
        
        # Average motion should be low for stationary objects
        avg_motion = sum(self.motion_history) / len(self.motion_history)
        return avg_motion < self.MOTION_THRESHOLD
    
    def _check_region_stability(self, current_region: np.ndarray) -> bool:
        """
        Check if the current region is similar to when object was first detected.
        This helps confirm it's the same object, not different transient objects.
        """
        if self.stable_region is None:
            return True  # No baseline yet
        
        try:
            if current_region.shape != self.stable_region.shape:
                return True  # Can't compare, assume stable
            
            gray_current = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            gray_stable = cv2.cvtColor(self.stable_region, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray_current, gray_stable)
            diff_level = float(np.mean(diff))
            
            # If very different from stable region, it might be a different object
            return diff_level < 40.0
        except:
            return True
    
    def update(self, is_occupied: bool, confidence: float, current_region: Optional[np.ndarray] = None):
        """
        Update slot state with TEMPORAL PERSISTENCE logic.
        
        Status changes require:
        1. Consistent detection over multiple frames
        2. Stability over time (not just frame count)
        3. Low motion (object must be stationary)
        """
        current_time = time.time()
        
        # Calculate motion if we have current region
        if current_region is not None:
            motion_level = self._calculate_motion(current_region)
            self.motion_history.append(motion_level)
            self.previous_region = current_region.copy()
        
        # Add to history
        self.history.append(is_occupied)
        self.confidence = confidence
        self.frames_since_change += 1
        
        # Track consecutive same-state detections
        if is_occupied == self.last_detection_state:
            self.consecutive_same_state += 1
        else:
            self.consecutive_same_state = 1
            self.pending_start_time = None  # Reset pending timer
            self.stable_region = None
        
        self.last_detection_state = is_occupied
        
        # Need minimum history before making decisions
        if len(self.history) < 5:
            return None
        
        # Calculate detection ratios
        occupied_count = sum(self.history)
        total = len(self.history)
        occupied_ratio = occupied_count / total
        
        old_status = self.status
        new_status = old_status  # Default: no change
        
        # Determine what status the detection suggests
        if occupied_ratio >= 0.70:
            suggested_status = "occupied"
        elif occupied_ratio <= 0.30:
            suggested_status = "vacant"
        else:
            suggested_status = old_status  # Uncertain, keep current
        
        # Check if we should change status
        if suggested_status != old_status:
            # Start pending state if not already
            if self.pending_status != suggested_status:
                self.pending_status = suggested_status
                self.pending_start_time = current_time
                if current_region is not None:
                    self.stable_region = current_region.copy()
                return None  # Wait for confirmation
            
            # Check if enough time has passed
            time_in_pending = current_time - (self.pending_start_time or current_time)
            
            # Different thresholds for occupied vs vacant
            if suggested_status == "occupied":
                required_time = self.STABILITY_TIME_SECONDS
                required_frames = self.MIN_CONSECUTIVE_FRAMES
                
                # Additional check: object must be stationary
                is_stationary = self._is_object_stationary()
                
                # Check region stability (same object throughout)
                region_stable = True
                if current_region is not None:
                    region_stable = self._check_region_stability(current_region)
                
                # All conditions must be met
                if (time_in_pending >= required_time and 
                    self.consecutive_same_state >= required_frames and
                    is_stationary and region_stable):
                    new_status = "occupied"
                    
            else:  # vacant
                required_time = self.VACANCY_TIME_SECONDS
                required_frames = self.MIN_CONSECUTIVE_FRAMES // 2  # Faster vacancy detection
                
                if (time_in_pending >= required_time and 
                    self.consecutive_same_state >= required_frames):
                    new_status = "vacant"
        else:
            # Status matches, reset pending
            self.pending_status = None
            self.pending_start_time = None
        
        # Apply status change
        if new_status != old_status:
            self.status = new_status
            self.last_change_time = current_time
            self.frames_since_change = 0
            self.pending_status = None
            self.pending_start_time = None
            self.stable_region = None
            
            print(f"🔄 Slot #{self.slot_number}: {old_status} → {new_status} "
                  f"(confidence: {confidence:.2f}, consecutive: {self.consecutive_same_state})")
            
            return {
                "slot_number": self.slot_number,
                "old_status": old_status,
                "new_status": new_status,
                "confidence": confidence,
                "timestamp": current_time
            }
        
        return None


# ============================================================
# DETECTION SESSION CLASS
# ============================================================

class DetectionSession:
    """Manages detection for a single parking spot."""
    
    def __init__(self, spot_id, grid_config: Optional[dict] = None):
        self.spot_id = spot_id
        self.slots = {}
        self.frame_count = 0
        self.started_at = time.time()
        self.reference_frame = None
        self.grid_locked = False
        self.grid_config = grid_config
        self.reference_frame_size = None
        
        # Initialize slot trackers from config
        if grid_config and "cells" in grid_config:
            cells = grid_config["cells"]
            
            # Check if we have frame dimensions to scale normalized coords immediately
            if cells and cells[0].get("bbox_normalized"):
                # If grid_config has frame dimensions, scale immediately
                if "frame_width" in grid_config and "frame_height" in grid_config:
                    width = grid_config["frame_width"]
                    height = grid_config["frame_height"]
                    print(f"📐 Scaling normalized coordinates to configured size: {width}x{height}")
                    
                    for cell in cells:
                        slot_num = cell["slot_number"]
                        norm_bbox = cell["bbox_normalized"]
                        
                        x1 = int(norm_bbox[0] * width)
                        y1 = int(norm_bbox[1] * height)
                        x2 = int(norm_bbox[2] * width)
                        y2 = int(norm_bbox[3] * height)
                        
                        bbox = [x1, y1, x2, y2]
                        self.slots[slot_num] = SlotTracker(slot_num, bbox)
                    
                    print(f"📊 Session created with {len(self.slots)} pre-scaled slots")
                else:
                    print(f"📊 Session created with normalized grid (will scale on first frame)")
            else:
                # Has absolute coordinates
                for cell in cells:
                    slot_num = cell["slot_number"]
                    bbox = cell["bbox"]
                    self.slots[slot_num] = SlotTracker(slot_num, bbox)
                print(f"📊 Session created with {len(self.slots)} pre-defined slots")
            
            self.grid_locked = True
        else:
            print(f"📊 Session created - waiting for grid configuration")
    
    def set_reference_frame(self, frame_bgr: np.ndarray):
        """Set reference frame (empty parking lot)."""
        self.reference_frame = frame_bgr.copy()
        
        height, width = frame_bgr.shape[:2]
        for slot_num, tracker in self.slots.items():
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)
            ref_region = frame_bgr[y1:y2, x1:x2]
            tracker.set_reference(ref_region)
        
        print(f"✅ Reference frame set for {len(self.slots)} slots")
    
    def process_frame(self, frame_bgr: np.ndarray, use_ai: bool = True):
        """
        🚀 OPTIMIZED: Process a frame and detect occupancy for all slots.
        Target: 30+ FPS with frame skipping and minimal annotation.
        """
        self.frame_count += 1
        
        if frame_bgr is None:
            return None, {}, None
        
        height, width = frame_bgr.shape[:2]
        
        # On first frame, scale normalized coordinates to actual frame size
        if self.frame_count == 1 and len(self.slots) == 0 and self.grid_config:
            if "cells" in self.grid_config:
                cells = self.grid_config["cells"]
                
                if cells and cells[0].get("bbox_normalized"):
                    print(f"📐 Scaling normalized coordinates to frame size: {width}x{height}")
                    for cell in cells:
                        slot_num = cell["slot_number"]
                        norm_bbox = cell["bbox_normalized"]
                        
                        x1 = int(norm_bbox[0] * width)
                        y1 = int(norm_bbox[1] * height)
                        x2 = int(norm_bbox[2] * width)
                        y2 = int(norm_bbox[3] * height)
                        
                        bbox = [x1, y1, x2, y2]
                        self.slots[slot_num] = SlotTracker(slot_num, bbox)
                    
                    print(f"✅ Created {len(self.slots)} slots from normalized coordinates")
        
        # Store first frame as reference
        if self.reference_frame is None and self.frame_count == 1:
            print("📸 Capturing first frame as reference")
            self.set_reference_frame(frame_bgr)
        
        # 🚀 OPTIMIZATION: Skip detection on some frames, just use cached results
        # Process detection every 2nd frame for speed
        run_detection = (self.frame_count % 2 == 0) or self.frame_count <= 5
        
        occupancy = {}
        state_change = None
        
        # If no slots, return early with simple message
        if len(self.slots) == 0:
            annotated = frame_bgr.copy()
            message = "⚠️ NO GRID CONFIGURED"
            cv2.putText(annotated, message, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return annotated, {}, None
        
        # 🚀 OPTIMIZATION: Process all slots with minimal overhead
        for slot_num, tracker in self.slots.items():
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                occupancy[str(slot_num)] = {
                    "status": tracker.status,
                    "confidence": round(tracker.confidence, 2)
                }
                continue
            
            slot_region = frame_bgr[y1:y2, x1:x2]
            
            if slot_region is None or slot_region.size == 0:
                occupancy[str(slot_num)] = {
                    "status": tracker.status,
                    "confidence": round(tracker.confidence, 2)
                }
                continue
            
            # 🚀 Only run detection on specific frames
            if run_detection:
                result = detect_vehicle_ensemble(
                    slot_region, 
                    tracker.reference_region,
                    tracker.previous_region,
                    use_ai=False  # Disabled AI for speed
                )
                
                is_occupied, confidence, is_shadow = result
                
                # Shadow handling
                if is_shadow:
                    tracker.shadow_lock_frames = 8
                
                if tracker.shadow_lock_frames > 0:
                    tracker.shadow_lock_frames -= 1
                    is_occupied = False
                    confidence = max(confidence, 0.85)
                
                # Update tracker
                change = tracker.update(is_occupied, confidence, slot_region)
                if change:
                    state_change = change
            
            occupancy[str(slot_num)] = {
                "status": tracker.status,
                "confidence": round(tracker.confidence, 2)
            }
        
        # 🚀 OPTIMIZATION: Lightweight annotation (only draw boxes, minimal text)
        annotated = frame_bgr.copy()
        
        for slot_num, tracker in self.slots.items():
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)
            
            # Simple color based on status
            if tracker.status == "occupied":
                color = (0, 0, 255)  # Red
            elif tracker.pending_status == "occupied":
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw rectangle (fast)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Simple label (minimal text rendering)
            label = f"#{slot_num}"
            cv2.putText(annotated, label, (x1 + 3, y1 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Summary (draw once)
        total = len(self.slots)
        occupied = sum(1 for s in self.slots.values() if s.status == "occupied")
        summary = f"O:{occupied}/{total}"
        cv2.putText(annotated, summary, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return annotated, occupancy, state_change


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


@app.route('/detect-grid', methods=['POST'])
def detect_grid():
    """
    🔥 NEW ENDPOINT: Auto-detect grid using YOLO (static approach)
    
    Request body:
    {
        "frame": "base64_image_data",
        "aoi": {
            "bbox": [x1, y1, x2, y2],  // Optional: absolute coordinates
            "bbox_normalized": [x1, y1, x2, y2]  // Optional: normalized (0-1)
        }
    }
    
    Returns grid with BOTH absolute and normalized coordinates
    """
    try:
        data = request.json or {}
        frame_b64 = data.get('frame')
        aoi_data = data.get('aoi')
        
        if not frame_b64:
            return jsonify({"error": "Missing frame"}), 400
        
        # Decode frame
        frame_bgr = decode_base64_image(frame_b64)
        if frame_bgr is None:
            return jsonify({"error": "Invalid image"}), 400
        
        height, width = frame_bgr.shape[:2]
        print(f"📸 Frame size: {width}x{height}")
        
        # Process AOI
        aoi_absolute = None
        aoi_normalized = None
        
        if aoi_data:
            if isinstance(aoi_data, dict):
                # Check for normalized coordinates
                if 'bbox_normalized' in aoi_data:
                    aoi_normalized = aoi_data['bbox_normalized']
                    # Scale to absolute
                    x1 = int(aoi_normalized[0] * width)
                    y1 = int(aoi_normalized[1] * height)
                    x2 = int(aoi_normalized[2] * width)
                    y2 = int(aoi_normalized[3] * height)
                    aoi_absolute = (x1, y1, x2, y2)
                    print(f"🎯 AOI (from normalized): {aoi_absolute}")
                
                # Check for absolute coordinates
                elif 'bbox' in aoi_data:
                    bbox = aoi_data['bbox']
                    aoi_absolute = tuple(bbox) if isinstance(bbox, list) else bbox
                    # Calculate normalized
                    aoi_normalized = [
                        aoi_absolute[0] / width,
                        aoi_absolute[1] / height,
                        aoi_absolute[2] / width,
                        aoi_absolute[3] / height
                    ]
                    print(f"🎯 AOI (from absolute): {aoi_absolute}")
        
        # Run YOLO detection (static approach)
        result = detect_grid_static_approach(
            frame_bgr,
            aoi_absolute=aoi_absolute,
            conf_thresh=0.20,
            imgsz=640
        )
        
        if not result["success"]:
            return jsonify(result), 500
        
        # Add normalized coordinates to cells
        cells_with_normalized = calculate_normalized_coordinates(
            result["cells"],
            width,
            height
        )
        
        # Build response
        response = {
            "success": True,
            "num_cells": result["num_cells"],
            "cells": cells_with_normalized,  # Contains both absolute and normalized
            "annotated_frame": result["annotated_frame"],
            "frame_width": width,
            "frame_height": height,
            "message": result["message"]
        }
        
        # Add AOI info if used
        if aoi_absolute:
            response["aoi"] = {
                "bbox": list(aoi_absolute),
                "bbox_normalized": aoi_normalized
            }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Grid detection error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/start-detection', methods=['POST'])
def start_detection():
    """Start detection session for a parking spot."""
    try:
        data = request.json or {}
        spot_id = data.get('parking_spot_id')
        grid_config = data.get('grid_config')
        
        if not spot_id:
            return jsonify({
                "success": False,
                "message": "Missing parking_spot_id"
            }), 400
        
        # Create session
        session = DetectionSession(spot_id, grid_config)
        active_sessions[spot_id] = session
        
        print(f"✅ Detection started for spot {spot_id}")
        return jsonify({
            "success": True,
            "message": "Detection started",
            "spot_id": spot_id,
            "num_slots": len(session.slots)
        })
    
    except Exception as e:
        print(f"❌ Start detection error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


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
        
        # Include annotated frame
        if annotated_frame is not None:
            encoded = encode_frame_to_base64(annotated_frame, quality=75)
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
    
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)  