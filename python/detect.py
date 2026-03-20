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

def detect_with_ai(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    IMPROVED: Use Florence-2 VQA task for direct binary classification.
    This is far more accurate than caption-based keyword hunting.
    Returns: (is_occupied, confidence)
    """
    global MODEL, PROCESSOR

    try:
        image = Image.fromarray(cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2RGB))

        # IMPROVEMENT #1: Use VQA prompt for direct yes/no answer
        # Florence-2 VQA gives a direct answer instead of a caption we must parse
        prompt = "<VQA>"
        question = "Is there a vehicle, car, toy car, or any object occupying this parking slot?"

        inputs = PROCESSOR(
            text=prompt,
            images=image,
            return_tensors="pt",
            text_kwargs={"text_pair": question}
        ).to(DEVICE)

        generated_ids = MODEL.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

        result = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].lower().strip()

        print(f"  🤖 VQA result: '{result}'")

        # Direct yes/no parsing — much more reliable than keyword hunting
        if result.startswith("yes"):
            return True, 0.85
        elif result.startswith("no"):
            return False, 0.85
        else:
            # Ambiguous answer — fall back to keyword check with lower confidence
            keywords = ["car", "vehicle", "toy", "object", "block", "occupied"]
            is_object = any(k in result for k in keywords)
            return is_object, 0.60

    except Exception as e:
        print(f"⚠️ AI detection error: {e}")
        return False, 0.5


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
    IMPROVED: Detect vehicle by comparing with reference using SSIM index
    for a truly independent signal, plus the original color-diff logic.
    SHADOW-ROBUST: Uses normalized color space and hue comparison.
    """
    if reference_region_bgr is None:
        return False, 0.5

    try:
        if slot_region_bgr.shape != reference_region_bgr.shape:
            reference_region_bgr = cv2.resize(reference_region_bgr,
                                             (slot_region_bgr.shape[1], slot_region_bgr.shape[0]))

        # IMPROVEMENT #7: Add SSIM as an orthogonal signal
        # SSIM captures structural similarity — a fundamentally different measure
        # from pixel diff, making the ensemble truly independent.
        gray_current = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2GRAY)

        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score, _ = ssim(gray_current, gray_reference, full=True)
            # Low SSIM = structure has changed = likely occupied
            ssim_occupied = ssim_score < 0.75
            ssim_conf = min(0.90, 0.5 + abs(0.75 - ssim_score) * 2.0)
        except ImportError:
            # skimage not available — fall back gracefully
            ssim_occupied = None
            ssim_conf = 0.5

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

        edges_current = cv2.Canny(gray_current, 30, 100)
        edges_reference = cv2.Canny(gray_reference, 30, 100)

        edge_diff = cv2.absdiff(edges_current, edges_reference)
        new_edges_ratio = np.sum(edge_diff > 0) / edge_diff.size

        current_edge_density = np.sum(edges_current > 0) / edges_current.size
        reference_edge_density = np.sum(edges_reference > 0) / edges_reference.size
        edge_increase = current_edge_density - reference_edge_density

        if new_edges_ratio > 0.03 and edge_increase > 0.02:
            base_result = True
            base_conf = 0.7
        elif is_likely_shadow:
            base_result = False
            base_conf = 0.65
        else:
            combined_hs_diff = (hue_diff.astype(np.float32) * 2 + sat_diff.astype(np.float32)) / 3
            _, thresh_hs = cv2.threshold(combined_hs_diff.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
            _, thresh_bgr = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)

            hs_change_ratio = np.sum(thresh_hs > 0) / thresh_hs.size
            bgr_change_ratio = np.sum(thresh_bgr > 0) / thresh_bgr.size

            if hs_change_ratio > 0.12 and bgr_change_ratio > 0.15:
                base_result = True
                base_conf = min(0.95, 0.75 + hs_change_ratio)
            elif hs_change_ratio > 0.20:
                base_result = True
                base_conf = min(0.90, 0.7 + hs_change_ratio)
            elif bgr_change_ratio > 0.35 and hs_change_ratio > 0.1:
                base_result = True
                base_conf = 0.65
            else:
                std_current = float(np.std(gray_current))  # type: ignore
                std_reference = float(np.std(gray_reference))  # type: ignore
                mean_current = float(np.mean(gray_current))  # type: ignore
                mean_reference = float(np.mean(gray_reference))  # type: ignore

                if mean_current > mean_reference + 20 and std_current < std_reference - 10:
                    base_result = True
                    base_conf = 0.65
                else:
                    base_result = False
                    base_conf = 0.7

        # Combine SSIM signal with base result if available
        if ssim_occupied is not None:
            if ssim_occupied == base_result:
                # Agreement — boost confidence
                final_conf = min(0.95, (base_conf + ssim_conf) / 2 + 0.05)
                return base_result, final_conf
            else:
                # Disagreement — use higher-confidence result
                if ssim_conf > base_conf:
                    return ssim_occupied, ssim_conf * 0.9
                else:
                    return base_result, base_conf * 0.9

        return base_result, base_conf

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
    IMPROVED: Detect rapid motion using Farneback optical flow instead of
    raw frame difference. Optical flow is immune to global lighting flicker
    and correctly handles slow-moving hands.
    Returns: (is_rapid_motion, motion_magnitude)
    """
    if previous_region_bgr is None:
        return False, 0.0

    try:
        if current_region_bgr.shape != previous_region_bgr.shape:
            return False, 0.0

        gray_current = cv2.cvtColor(current_region_bgr, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_region_bgr, cv2.COLOR_BGR2GRAY)

        # IMPROVEMENT #2: Farneback optical flow
        # Unlike absdiff, optical flow captures COHERENT directional motion.
        # A hand moving through a slot produces large coherent vectors;
        # a lighting flicker produces small random ones.
        flow = cv2.calcOpticalFlowFarneback(
            gray_previous,
            gray_current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        # Ratio of pixels with significant motion
        motion_coverage = float(np.sum(mag > 1.5) / mag.size)

        # Fast coherent motion = hand; requires BOTH high magnitude AND large coverage
        # This rejects lighting flicker (low coverage) and tiny vibrations (low magnitude)
        is_rapid = mean_mag > 2.5 and motion_coverage > 0.20

        return is_rapid, mean_mag

    except Exception as e:
        # Fall back to original frame-diff approach if optical flow fails
        try:
            gray_c = cv2.cvtColor(current_region_bgr, cv2.COLOR_BGR2GRAY)
            gray_p = cv2.cvtColor(previous_region_bgr, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_c, gray_p)
            motion_level = float(np.mean(diff))
            motion_ratio = np.sum(diff > 30) / diff.size
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


def detect_vehicle_ensemble(
    slot_region_bgr: np.ndarray,
    reference_region_bgr: Optional[np.ndarray] = None,
    previous_region_bgr: Optional[np.ndarray] = None,
    use_ai: bool = True
) -> Tuple[bool, float, bool]:

    try:
        gray = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)

        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        mean_sat = float(np.mean(s))
        mean_val = float(np.mean(v))
        std_val = float(np.std(v))

        edges = cv2.Canny(gray, 60, 160)
        edge_density = np.sum(edges > 0) / edges.size

        # ============================================================
        # 🚨 STEP 1: SHADOW REJECTION
        # IMPROVEMENT #1: Adaptive thresholds based on scene statistics
        # instead of hard-coded values that break under different lighting.
        # ============================================================

        # Compute scene-adaptive saturation threshold
        # If the whole frame region is low-saturation (overcast day),
        # the fixed threshold of 35 would wrongly reject real objects.
        scene_sat_thresh = max(20, mean_sat * 0.6)   # adaptive lower bound
        scene_val_low    = max(20, mean_val * 0.3)   # adaptive dark threshold
        scene_val_high   = min(230, mean_val * 1.8)  # adaptive bright threshold

        is_shadow = (
            mean_sat < scene_sat_thresh and
            scene_val_low < mean_val < scene_val_high and
            std_val < 35 and
            edge_density < 0.025
        )

        if reference_region_bgr is not None:
            try:
                ref_hsv = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2HSV)

                hue_diff = cv2.absdiff(h, ref_hsv[:, :, 0])
                hue_diff = np.minimum(hue_diff, 180 - hue_diff)

                mean_hue_diff = float(np.mean(hue_diff))
                val_diff = float(np.mean(ref_hsv[:, :, 2])) - mean_val

                if mean_hue_diff < 12 and val_diff > 20:
                    is_shadow = True
            except:
                pass

        # LBP-inspired texture check: real objects have local binary variation
        # even when HSV looks like a shadow. Compute local std in small patches.
        patch_h = max(4, gray.shape[0] // 6)
        patch_w = max(4, gray.shape[1] // 6)
        local_stds = []
        for py in range(0, gray.shape[0] - patch_h, patch_h):
            for px in range(0, gray.shape[1] - patch_w, patch_w):
                patch = gray[py:py+patch_h, px:px+patch_w]
                local_stds.append(float(np.std(patch)))
        mean_local_std = float(np.mean(local_stds)) if local_stds else 0.0

        # If significant local texture exists, it is NOT a plain shadow
        if is_shadow and mean_local_std > 18.0:
            is_shadow = False

        if is_shadow:
            return False, 0.9, True

        # ============================================================
        # ⚡ STEP 2: MOTION FILTER (optical flow)
        # IMPROVEMENT #2: Uses detect_rapid_motion which now uses
        # Farneback optical flow instead of raw frame diff.
        # ============================================================

        if previous_region_bgr is not None:
            try:
                is_rapid, motion_mag = detect_rapid_motion(slot_region_bgr, previous_region_bgr)
                if is_rapid:
                    return False, 0.85, False
            except:
                pass

        # ============================================================
        # 🧠 STEP 3: CV DETECTION
        # IMPROVEMENT #3: Add contour solidity check to distinguish
        # a solid object blob from scattered noise edges.
        # ============================================================

        score = 0.0

        if edge_density > 0.04:
            score += 0.35
        if edge_density > 0.07:
            score += 0.25

        # Contour solidity: real objects form compact, solid blobs
        _, binary_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            total_area = gray.shape[0] * gray.shape[1]

            if hull_area > 0:
                solidity = area / hull_area
                coverage = area / total_area

                # High solidity + meaningful coverage = compact object, not noise
                if solidity > 0.60 and 0.08 < coverage < 0.95:
                    score += 0.30
                elif solidity > 0.40 and coverage > 0.05:
                    score += 0.15

        if mean_sat > 45:
            score += 0.2

        if std_val > 30:
            score += 0.2

        if reference_region_bgr is not None:
            try:
                ref_gray = cv2.cvtColor(reference_region_bgr, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, ref_gray)
                diff_ratio = np.sum(diff > 40) / diff.size

                if diff_ratio > 0.12:
                    score += 0.3
                if diff_ratio > 0.25:
                    score += 0.25
            except:
                pass

        is_occupied = score >= 0.5
        confidence = min(0.95, 0.5 + score * 0.5)

        if edge_density > 0.09:
            return True, 0.85, False

        # ============================================================
        # 🤖 STEP 4: AI FALLBACK (ONLY IF UNCERTAIN)
        # Uses the improved VQA-based detect_with_ai
        # ============================================================

        if use_ai and 0.48 < score < 0.60:
            ai_result, ai_conf = detect_with_ai(slot_region_bgr)

            if ai_result:
                is_occupied = True
                confidence = max(confidence, ai_conf)
            else:
                is_occupied = False
                confidence = max(confidence, ai_conf)

        return is_occupied, confidence, False

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

    # First check for rapid motion (optical flow-based)
    is_rapid_motion, motion_level = detect_rapid_motion(slot_region_bgr, previous_region_bgr)
    if is_rapid_motion:
        return False, 0.8, False

    # Check if this is likely a shadow
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
    weights.append(2.5)

    # Method 5: SSIM-based difference (orthogonal signal)
    # IMPROVEMENT #7: Include SSIM as independent method in full ensemble
    if reference_region_bgr is not None:
        is_occ_diff, conf_diff = detect_vehicle_difference_based(slot_region_bgr, reference_region_bgr)
        votes.append(is_occ_diff)
        confidences.append(conf_diff)
        weights.append(2.0)

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
        low_edges = edge_density < 0.02

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

                hue_diff = cv2.absdiff(hsv[:, :, 0], hsv_ref[:, :, 0])
                hue_diff = np.minimum(hue_diff, 180 - hue_diff)
                mean_hue_diff = float(np.mean(hue_diff))

                val_diff = float(np.mean(hsv_ref[:, :, 2])) - float(np.mean(hsv[:, :, 2]))

                hue_preserved = mean_hue_diff < 15 and val_diff > 20
            except:
                pass

        shadow_indicators = sum([low_edges, uniform_color, low_saturation, hue_preserved])

        is_shadow = shadow_indicators >= 3
        confidence = min(0.95, 0.5 + shadow_indicators * 0.15)

        return is_shadow, confidence

    except Exception as e:
        return False, 0.5


def has_internal_structure(slot_region_bgr: np.ndarray) -> Tuple[bool, float]:
    """
    Check if the region has internal structure (edges, texture, patterns).
    Real objects have internal detail, shadows don't.
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

        score = 0

        if edge_density_fine > 0.03:
            score += 1
        if edge_density_fine > 0.06:
            score += 1

        if edge_density_coarse > 0.02:
            score += 1

        if mean_gradient > 12:
            score += 1
        if mean_gradient > 25:
            score += 1

        if max_gradient > 100:
            score += 1

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

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        hsv = cv2.cvtColor(slot_region_bgr, cv2.COLOR_BGR2HSV)
        hue_std = float(np.std(hsv[:, :, 0]))
        sat_std = float(np.std(hsv[:, :, 1]))
        val_std = float(np.std(hsv[:, :, 2]))

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = float(np.mean(gradient_mag))

        score = 0

        if edge_density > 0.03:
            score += 1
        if edge_density > 0.06:
            score += 1

        if hue_std > 15:
            score += 1
        if hue_std > 30:
            score += 1

        if sat_std > 30:
            score += 1

        if val_std > 25:
            score += 1

        if mean_gradient > 15:
            score += 1
        if mean_gradient > 30:
            score += 1

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

    IMPROVEMENTS APPLIED:
    - Confidence-weighted history (IMPROVEMENT #4): each detection is weighted
      by its confidence before voting, so a 0.9-confidence "vacant" outweighs
      a 0.52-confidence "occupied".
    - Exponential decay on history so recent frames matter more than old ones.
    """

    STABILITY_TIME_SECONDS = 1.5
    VACANCY_TIME_SECONDS = 0.5
    MIN_CONSECUTIVE_FRAMES = 4
    MOTION_THRESHOLD = 25.0
    HISTORY_SIZE = 15
    DECAY = 0.92   # Weight decay per frame — older frames matter less

    def __init__(self, slot_number: int, bbox: list):
        self.slot_number = slot_number
        self.bbox = bbox
        self.status = "vacant"
        self.pending_status = None
        self.confidence = 0.0
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.last_change_time = time.time()
        self.frames_since_change = 0
        self.reference_region = None
        self.previous_region = None
        self.shadow_lock_frames = 0

        # Temporal persistence tracking
        self.pending_start_time = None
        self.consecutive_same_state = 0
        self.last_detection_state = None
        self.motion_history = deque(maxlen=5)
        self.stable_region = None

        # IMPROVEMENT #4: Confidence-weighted accumulators with exponential decay
        # weighted_occupied accumulates confidence × vote for "occupied"
        # weighted_total  accumulates confidence for all votes
        self.weighted_occupied = 0.0
        self.weighted_total = 0.0

    def set_reference(self, reference_region: np.ndarray):
        """Set the reference (empty) image for this slot."""
        self.reference_region = reference_region.copy()

    def _calculate_motion(self, current_region: np.ndarray) -> float:
        """
        Calculate motion level between current and previous frame.
        Uses optical flow magnitude if previous frame is available.
        """
        if self.previous_region is None:
            return 0.0

        try:
            if current_region.shape != self.previous_region.shape:
                return 0.0

            _, motion_mag = detect_rapid_motion(current_region, self.previous_region)
            return motion_mag
        except:
            return 0.0

    def _is_object_stationary(self) -> bool:
        """
        Check if the detected object is stationary (not moving).
        Returns True if object has been stable across recent frames.
        """
        if len(self.motion_history) < 3:
            return False

        avg_motion = sum(self.motion_history) / len(self.motion_history)
        return avg_motion < self.MOTION_THRESHOLD

    def _check_region_stability(self, current_region: np.ndarray) -> bool:
        """
        Check if the current region is similar to when object was first detected.
        """
        if self.stable_region is None:
            return True

        try:
            if current_region.shape != self.stable_region.shape:
                return True

            gray_current = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            gray_stable = cv2.cvtColor(self.stable_region, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray_current, gray_stable)
            diff_level = float(np.mean(diff))

            return diff_level < 40.0
        except:
            return True

    def update(self, is_occupied: bool, confidence: float, current_region: Optional[np.ndarray] = None):
        """
        Update slot state with TEMPORAL PERSISTENCE logic.
        IMPROVEMENT #4: Uses confidence-weighted exponential decay instead of
        simple boolean history, so high-confidence detections dominate.
        """
        current_time = time.time()

        # Track motion if we have current region
        if current_region is not None:
            motion_level = self._calculate_motion(current_region)
            self.motion_history.append(motion_level)
            self.previous_region = current_region.copy()

        # IMPROVEMENT #4: Update exponential decay accumulators
        self.weighted_occupied = self.weighted_occupied * self.DECAY + (confidence if is_occupied else 0.0)
        self.weighted_total    = self.weighted_total    * self.DECAY + confidence

        # Also keep raw boolean history for consecutive-state counting
        self.history.append(is_occupied)
        self.confidence = confidence
        self.frames_since_change += 1

        # Track consecutive same-state detections
        if is_occupied == self.last_detection_state:
            self.consecutive_same_state += 1
        else:
            self.consecutive_same_state = 1
            self.pending_start_time = None
            self.stable_region = None

        self.last_detection_state = is_occupied

        # Need minimum history before making decisions
        if len(self.history) < 5:
            return None

        # IMPROVEMENT #4: Use confidence-weighted ratio instead of raw count ratio
        if self.weighted_total > 0:
            occupied_ratio = self.weighted_occupied / self.weighted_total
        else:
            occupied_count = sum(self.history)
            total = len(self.history)
            occupied_ratio = occupied_count / total

        old_status = self.status
        new_status = old_status

        # Determine what status the detection suggests
        if occupied_ratio >= 0.70:
            suggested_status = "occupied"
        elif occupied_ratio <= 0.30:
            suggested_status = "vacant"
        else:
            suggested_status = old_status

        if suggested_status != old_status:
            if self.pending_status != suggested_status:
                self.pending_status = suggested_status
                self.pending_start_time = current_time
                if current_region is not None:
                    self.stable_region = current_region.copy()
                return None

            time_in_pending = current_time - (self.pending_start_time or current_time)

            if suggested_status == "occupied":
                required_time = self.STABILITY_TIME_SECONDS
                required_frames = self.MIN_CONSECUTIVE_FRAMES

                is_stationary = self._is_object_stationary()

                region_stable = True
                if current_region is not None:
                    region_stable = self._check_region_stability(current_region)

                if (time_in_pending >= required_time and
                    self.consecutive_same_state >= required_frames and
                    is_stationary and region_stable):
                    new_status = "occupied"

            else:  # vacant
                required_time = self.VACANCY_TIME_SECONDS
                required_frames = self.MIN_CONSECUTIVE_FRAMES // 2

                if (time_in_pending >= required_time and
                    self.consecutive_same_state >= required_frames):
                    new_status = "vacant"
        else:
            self.pending_status = None
            self.pending_start_time = None

        if new_status != old_status:
            self.status = new_status
            self.last_change_time = current_time
            self.frames_since_change = 0
            self.pending_status = None
            self.pending_start_time = None
            self.stable_region = None

            print(f"🔄 Slot #{self.slot_number}: {old_status} → {new_status} "
                  f"(conf: {confidence:.2f}, weighted_ratio: {occupied_ratio:.2f}, "
                  f"consecutive: {self.consecutive_same_state})")

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
    """
    Manages detection for a single parking spot.

    IMPROVEMENTS:
    - Rolling background model (IMPROVEMENT #2): MOG2 subtractor updates
      the background continuously, handling lighting drift across the day.
    - Adaptive frame skipping (IMPROVEMENT #5): processes every frame when
      motion is detected; throttles to every 6th frame when the lot is quiet.
    """

    def __init__(self, spot_id, grid_config: Optional[dict] = None):
        self.spot_id = spot_id
        self.slots = {}
        self.frame_count = 0
        self.started_at = time.time()
        self.reference_frame = None
        self.grid_locked = False
        self.grid_config = grid_config
        self.reference_frame_size = None

        # IMPROVEMENT #2: Per-slot rolling background subtractors (MOG2)
        # MOG2 learns the background over time so it handles lighting changes,
        # slow shadows shifting across the day, and camera exposure drift.
        # detectShadows=True means shadow pixels are labelled 127 (not 255),
        # giving us free shadow detection on every frame.
        self.bg_subtractors: dict = {}

        # IMPROVEMENT #5: Adaptive frame skipping — track whole-scene motion
        self.last_frame_gray: Optional[np.ndarray] = None
        self.scene_motion_level: float = 0.0

        # Initialize slot trackers from config
        if grid_config and "cells" in grid_config:
            cells = grid_config["cells"]

            if cells and cells[0].get("bbox_normalized"):
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
                        self._init_bg_subtractor(slot_num)

                    print(f"📊 Session created with {len(self.slots)} pre-scaled slots")
                else:
                    print(f"📊 Session created with normalized grid (will scale on first frame)")
            else:
                for cell in cells:
                    slot_num = cell["slot_number"]
                    bbox = cell["bbox"]
                    self.slots[slot_num] = SlotTracker(slot_num, bbox)
                    self._init_bg_subtractor(slot_num)
                print(f"📊 Session created with {len(self.slots)} pre-defined slots")

            self.grid_locked = True
        else:
            print(f"📊 Session created - waiting for grid configuration")

    def _init_bg_subtractor(self, slot_num: int):
        """Create a fresh MOG2 background subtractor for one slot."""
        self.bg_subtractors[slot_num] = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=40,
            detectShadows=True   # Shadow pixels → value 127 in the fg mask
        )

    def set_reference_frame(self, frame_bgr: np.ndarray):
        """Set reference frame (empty parking lot) and warm up MOG2 models."""
        self.reference_frame = frame_bgr.copy()

        height, width = frame_bgr.shape[:2]
        for slot_num, tracker in self.slots.items():
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)
            ref_region = frame_bgr[y1:y2, x1:x2]
            tracker.set_reference(ref_region)

            # IMPROVEMENT #2: Warm up MOG2 with the reference frame so it
            # starts with a valid background model immediately.
            if slot_num in self.bg_subtractors:
                for _ in range(5):   # feed reference several times to stabilise
                    self.bg_subtractors[slot_num].apply(ref_region, learningRate=0.5)

        print(f"✅ Reference frame set for {len(self.slots)} slots")

    def _get_bg_diff_ratio(self, slot_num: int, slot_region: np.ndarray) -> Tuple[float, float]:
        """
        Apply MOG2 subtractor and return (fg_ratio, shadow_ratio).
        fg_ratio    — fraction of pixels confidently in the foreground (value 255)
        shadow_ratio — fraction of pixels identified as shadow (value 127)
        A slot with high fg_ratio is occupied; high shadow_ratio alone means shadow.
        """
        if slot_num not in self.bg_subtractors:
            self._init_bg_subtractor(slot_num)

        fg_mask = self.bg_subtractors[slot_num].apply(slot_region, learningRate=0.005)

        total = fg_mask.size
        fg_ratio     = float(np.sum(fg_mask == 255) / total)
        shadow_ratio = float(np.sum(fg_mask == 127) / total)

        return fg_ratio, shadow_ratio

    def process_frame(self, frame_bgr: np.ndarray, use_ai: bool = True):
        """
        🚀 OPTIMIZED: Process a frame and detect occupancy for all slots.

        IMPROVEMENT #5 — Adaptive frame skipping:
        - Compute whole-scene motion on every frame (very cheap).
        - If scene_motion > threshold: run detection this frame (active lot).
        - Otherwise: run detection only every 6th frame (quiet lot).
        This keeps CPU usage low while ensuring no events are missed.
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
                        self._init_bg_subtractor(slot_num)

                    print(f"✅ Created {len(self.slots)} slots from normalized coordinates")

        # Store first frame as reference
        if self.reference_frame is None and self.frame_count == 1:
            print("📸 Capturing first frame as reference")
            self.set_reference_frame(frame_bgr)

        # IMPROVEMENT #5: Compute whole-scene motion (cheap grayscale diff)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray_small = cv2.resize(frame_gray, (160, 120))  # tiny for speed

        if self.last_frame_gray is not None:
            scene_diff = cv2.absdiff(frame_gray_small, self.last_frame_gray)
            self.scene_motion_level = float(np.mean(scene_diff))
        else:
            self.scene_motion_level = 0.0

        self.last_frame_gray = frame_gray_small.copy()

        # Adaptive skip decision:
        # Motion detected → process now; quiet → every 2nd frame for smoother playback
        MOTION_TRIGGER = 3.0   # mean pixel diff threshold on the small image
        QUIET_SKIP = 2         # process every Nth frame when quiet (reduced from 6 for smoother video)

        run_detection = (
            self.scene_motion_level > MOTION_TRIGGER or
            self.frame_count % QUIET_SKIP == 0 or
            self.frame_count <= 5   # always process first few frames
        )

        occupancy = {}
        state_change = None

        if len(self.slots) == 0:
            annotated = frame_bgr.copy()
            message = "⚠️ NO GRID CONFIGURED"
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

            slot_region = frame_bgr[y1:y2, x1:x2]

            if slot_region is None or slot_region.size == 0:
                occupancy[str(slot_num)] = {
                    "status": tracker.status,
                    "confidence": round(tracker.confidence, 2)
                }
                continue

            if run_detection:
                # IMPROVEMENT #2: Get MOG2 background-subtraction signal
                fg_ratio, shadow_ratio = self._get_bg_diff_ratio(slot_num, slot_region)

                # If MOG2 sees only shadow (no foreground), short-circuit to vacant
                if shadow_ratio > 0.25 and fg_ratio < 0.05:
                    change = tracker.update(False, 0.80, slot_region)
                    if change:
                        state_change = change
                    occupancy[str(slot_num)] = {
                        "status": tracker.status,
                        "confidence": round(tracker.confidence, 2),
                        "bg_fg_ratio": round(fg_ratio, 3),
                        "bg_shadow_ratio": round(shadow_ratio, 3)
                    }
                    continue

                # If MOG2 confidently sees foreground, can skip expensive CV
                if fg_ratio > 0.20:
                    change = tracker.update(True, min(0.90, 0.65 + fg_ratio), slot_region)
                    if change:
                        state_change = change
                    occupancy[str(slot_num)] = {
                        "status": tracker.status,
                        "confidence": round(tracker.confidence, 2),
                        "bg_fg_ratio": round(fg_ratio, 3),
                        "bg_shadow_ratio": round(shadow_ratio, 3)
                    }
                    continue

                # Uncertain MOG2 result → run full ensemble
                result = detect_vehicle_ensemble(
                    slot_region,
                    tracker.reference_region,
                    tracker.previous_region,
                    use_ai=use_ai
                )

                is_occupied, confidence, is_shadow = result

                change = tracker.update(is_occupied, confidence, slot_region)
                if change:
                    state_change = change

            occupancy[str(slot_num)] = {
                "status": tracker.status,
                "confidence": round(tracker.confidence, 2)
            }

        # Lightweight annotation
        annotated = frame_bgr.copy()

        for slot_num, tracker in self.slots.items():
            x1, y1, x2, y2 = clamp_bbox(tracker.bbox, width, height)

            if tracker.status == "occupied":
                color = (0, 0, 255)      # Red
            elif tracker.pending_status == "occupied":
                color = (0, 165, 255)    # Orange
            else:
                color = (0, 255, 0)      # Green

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"#{slot_num}"
            cv2.putText(annotated, label, (x1 + 3, y1 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        total = len(self.slots)
        occupied = sum(1 for s in self.slots.values() if s.status == "occupied")
        # Show motion level in summary to help with debugging
        summary = f"O:{occupied}/{total} M:{self.scene_motion_level:.1f}"
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
    Auto-detect grid using YOLO (static approach).

    Request body:
    {
        "frame": "base64_image_data",
        "aoi": {
            "bbox": [x1, y1, x2, y2],
            "bbox_normalized": [x1, y1, x2, y2]
        }
    }
    """
    try:
        data = request.json or {}
        frame_b64 = data.get('frame')
        aoi_data = data.get('aoi')

        if not frame_b64:
            return jsonify({"error": "Missing frame"}), 400

        frame_bgr = decode_base64_image(frame_b64)
        if frame_bgr is None:
            return jsonify({"error": "Invalid image"}), 400

        height, width = frame_bgr.shape[:2]
        print(f"📸 Frame size: {width}x{height}")

        aoi_absolute = None
        aoi_normalized = None

        if aoi_data:
            if isinstance(aoi_data, dict):
                if 'bbox_normalized' in aoi_data:
                    aoi_normalized = aoi_data['bbox_normalized']
                    x1 = int(aoi_normalized[0] * width)
                    y1 = int(aoi_normalized[1] * height)
                    x2 = int(aoi_normalized[2] * width)
                    y2 = int(aoi_normalized[3] * height)
                    aoi_absolute = (x1, y1, x2, y2)
                    print(f"🎯 AOI (from normalized): {aoi_absolute}")

                elif 'bbox' in aoi_data:
                    bbox = aoi_data['bbox']
                    aoi_absolute = tuple(bbox) if isinstance(bbox, list) else bbox
                    aoi_normalized = [
                        aoi_absolute[0] / width,
                        aoi_absolute[1] / height,
                        aoi_absolute[2] / width,
                        aoi_absolute[3] / height
                    ]
                    print(f"🎯 AOI (from absolute): {aoi_absolute}")

        result = detect_grid_static_approach(
            frame_bgr,
            aoi_absolute=aoi_absolute,
            conf_thresh=0.20,
            imgsz=640
        )

        if not result["success"]:
            return jsonify(result), 500

        cells_with_normalized = calculate_normalized_coordinates(
            result["cells"],
            width,
            height
        )

        response = {
            "success": True,
            "num_cells": result["num_cells"],
            "cells": cells_with_normalized,
            "annotated_frame": result["annotated_frame"],
            "frame_width": width,
            "frame_height": height,
            "message": result["message"]
        }

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

        if not spot_id:
            return jsonify({"error": "Missing spot_id"}), 400

        if spot_id not in active_sessions:
            return jsonify({"error": "No active session for this spot"}), 400

        if not frame_b64:
            return jsonify({"error": "Missing frame data"}), 400

        frame_bgr = decode_base64_image(frame_b64)

        if frame_bgr is None:
            return jsonify({
                "error": "Failed to decode image",
                "occupancy": {"slots": {}},
                "state_change": None
            }), 400

        session = active_sessions[spot_id]

        client_wants_ai = data.get('use_ai', True)

        next_frame = session.frame_count + 1

        # Run AI every 2nd frame (instead of every 3rd) for better responsiveness
        use_ai = client_wants_ai and (next_frame % 2 == 0)

        annotated_frame, occupancy, state_change = session.process_frame(frame_bgr, use_ai)

        response = {
            "occupancy": {"slots": occupancy},
            "state_change": state_change,
            "timestamp": timestamp,
            "frame_count": session.frame_count,
            "num_slots": len(session.slots),
            "scene_motion": round(session.scene_motion_level, 2)
        }

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