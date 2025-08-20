import os
from typing import Tuple, Dict
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tempfile import TemporaryDirectory
import base64
import time

router = APIRouter()


# --- All Helper Functions (No Changes Here) ---

def remove_bg(input_path: str, output_folder: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.png")
    output_image.save(output_path)
    return output_path


def extract_torso_component(mask: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if center_y >= labels.shape[0] or center_x >= labels.shape[1]:
        raise RuntimeError("Center point is outside the image bounds after processing.")
    label_at_center = labels[center_y, center_x]
    torso_mask = (labels == label_at_center).astype(np.uint8) * 255
    return torso_mask


def get_torso_edges(mask: np.ndarray, y: int, center_x: float) -> Tuple[int, int]:
    cols = np.where(mask[y] > 0)[0]
    if len(cols) < 2: return -1, -1
    runs = []
    start = cols[0]
    for i in range(1, len(cols)):
        if cols[i] != cols[i - 1] + 1:
            runs.append((start, cols[i - 1]))
            start = cols[i]
    runs.append((start, cols[-1]))
    for x0, x1 in runs:
        if x0 <= center_x <= x1: return x0, x1
    return cols[0], cols[-1]


def elliptical_circumference(a: float, b: float) -> float:
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))


def process_image(path: str):
    import cv2
    import mediapipe as mp
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(f"Could not load image: {path}")
    if img.shape[2] == 4:
        alpha = img[:, :, 3];
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY);
        rgb = img[:, :, :3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY);
        rgb = img
    h, w = mask.shape
    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        res = pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: raise RuntimeError("No pose landmarks detected")
        lm = res.pose_landmarks.landmark
        shoulder_y = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y + lm[
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) * 0.5) * h)
        hip_y = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_HIP].y + lm[
            mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) * 0.5) * h)
        shoulder_x = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x + lm[
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x) * 0.5) * w)
    return mask, shoulder_y, hip_y, shoulder_x, h


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    return f"data:image/{ext};base64,{encoded_string}"


## MODIFIED: This function now draws all four lines AND ADDS TEXT LABELS.
def draw_measurement_lines(
        original_image_path: str,
        shoulder_y: int, shoulder_x0: int, shoulder_x1: int,
        chest_y: int, chest_x0: int, chest_x1: int,
        waist_y: int, waist_x0: int, waist_x1: int,
        hip_y: int, hip_x0: int, hip_x1: int,
        output_path: str
):
    img = cv2.imread(original_image_path)
    red_color = (0, 0, 255);
    white_color = (255, 255, 255);
    black_color = (0, 0, 0)
    thickness = max(2, int(img.shape[1] / 300));
    font_scale = max(0.5, img.shape[1] / 1000)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_text_with_outline(text, x, y):
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_pos = (x - w - 15, y + (h // 2))  # Position text to the left of the line's start
        cv2.putText(img, text, text_pos, font, font_scale, black_color, thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, text_pos, font, font_scale, white_color, thickness, cv2.LINE_AA)

    if shoulder_x0 > 0:
        cv2.line(img, (shoulder_x0, shoulder_y), (shoulder_x1, shoulder_y), red_color, thickness)
        draw_text_with_outline("Shoulder", shoulder_x0, shoulder_y)
    if chest_x0 > 0:
        cv2.line(img, (chest_x0, chest_y), (chest_x1, chest_y), red_color, thickness)
        draw_text_with_outline("Chest", chest_x0, chest_y)
    if waist_x0 > 0:
        cv2.line(img, (waist_x0, waist_y), (waist_x1, waist_y), red_color, thickness)
        draw_text_with_outline("Waist", waist_x0, waist_y)
    if hip_x0 > 0:
        cv2.line(img, (hip_x0, hip_y), (hip_x1, hip_y), red_color, thickness)
        draw_text_with_outline("Hip", hip_x0, hip_y)

    cv2.imwrite(output_path, img)


@router.post("/measure")
async def measure(
        front_image: UploadFile = File(...),
        side_image: UploadFile = File(...),
        height_cm: float = File(...)
):
    output_dir = "output";
    os.makedirs(output_dir, exist_ok=True)

    with TemporaryDirectory() as tmpdir:
        try:
            # 1. Save and process images
            front_path = os.path.join(tmpdir, front_image.filename);
            side_path = os.path.join(tmpdir, side_image.filename)
            with open(front_path, "wb") as f:
                f.write(await front_image.read())
            with open(side_path, "wb") as f:
                f.write(await side_image.read())

            front_clean = remove_bg(front_path, tmpdir);
            side_clean = remove_bg(side_path, tmpdir)
            raw_f_mask, f_shoulder_y, f_hip_y, f_center_x, pixel_h = process_image(front_clean)
            s_mask, s_shoulder_y, s_hip_y, s_center_x, s_pixel_h = process_image(side_clean)

            f_mask = extract_torso_component(raw_f_mask, f_center_x, f_shoulder_y)

            # --- 2. Calculate Measurements (IMPROVED LOGIC) ---
            torso_height = f_hip_y - f_shoulder_y

            ## Shoulder Width
            f_shoulder_x0, f_shoulder_x1 = get_torso_edges(raw_f_mask, f_shoulder_y, f_center_x)
            shoulder_w_px = f_shoulder_x1 - f_shoulder_x0

            ## Chest Circumference
            f_chest_y = f_shoulder_y + int(torso_height * 0.20)
            s_chest_y = s_shoulder_y + int((s_hip_y - s_shoulder_y) * 0.20)
            f_chest_x0, f_chest_x1 = get_torso_edges(f_mask, f_chest_y, f_center_x)
            s_chest_x0, s_chest_x1 = get_torso_edges(s_mask, s_chest_y, s_center_x)
            chest_c_px = elliptical_circumference((f_chest_x1 - f_chest_x0) / 2, (s_chest_x1 - s_chest_x0) / 2)

            ## Waist Circumference (Search for narrowest part)
            waist_search_start_y = f_shoulder_y + int(torso_height * 0.50)
            waist_search_end_y = f_shoulder_y + int(torso_height * 0.85)
            f_best_y, f_best_w, f_waist_x0, f_waist_x1 = -1, float('inf'), -1, -1
            s_best_y, s_best_w, s_waist_x0, s_waist_x1 = -1, float('inf'), -1, -1
            for yf in range(waist_search_start_y, waist_search_end_y):
                x0f, x1f = get_torso_edges(f_mask, yf, f_center_x)
                if x0f > 0 and (x1f - x0f) < f_best_w: f_best_w, f_best_y, f_waist_x0, f_waist_x1 = (
                            x1f - x0f), yf, x0f, x1f
            s_search_start_y = int(waist_search_start_y * (s_pixel_h / pixel_h));
            s_search_end_y = int(waist_search_end_y * (s_pixel_h / pixel_h))
            for ys in range(s_search_start_y, s_search_end_y):
                x0s, x1s = get_torso_edges(s_mask, ys, s_center_x)
                if x0s > 0 and (x1s - x0s) < s_best_w: s_best_w, s_best_y, s_waist_x0, s_waist_x1 = (
                            x1s - x0s), ys, x0s, x1s
            if f_best_y == -1 or s_best_y == -1: raise RuntimeError("Could not determine waist.")
            waist_c_px = elliptical_circumference(f_best_w / 2, s_best_w / 2)

            ## Hip Circumference (Search for WIDEST part)
            hip_search_start_y = f_hip_y
            hip_search_end_y = f_hip_y + int(torso_height * 0.15)
            f_hip_best_y, f_hip_max_w, f_hip_x0, f_hip_x1 = -1, 0, -1, -1
            s_hip_best_y, s_hip_max_w, s_hip_x0, s_hip_x1 = -1, 0, -1, -1
            for yf in range(hip_search_start_y, hip_search_end_y):
                x0f, x1f = get_torso_edges(f_mask, yf, f_center_x)
                if x0f > 0 and (x1f - x0f) > f_hip_max_w: f_hip_max_w, f_hip_best_y, f_hip_x0, f_hip_x1 = (
                            x1f - x0f), yf, x0f, x1f
            s_search_start_y = int(hip_search_start_y * (s_pixel_h / pixel_h));
            s_search_end_y = int(hip_search_end_y * (s_pixel_h / pixel_h))
            for ys in range(s_search_start_y, s_search_end_y):
                x0s, x1s = get_torso_edges(s_mask, ys, s_center_x)
                if x0s > 0 and (x1s - x0s) > s_hip_max_w: s_hip_max_w, s_hip_best_y, s_hip_x0, s_hip_x1 = (
                            x1s - x0s), ys, x0s, x1s
            if f_hip_best_y == -1 or s_hip_best_y == -1: raise RuntimeError("Could not determine hip.")
            hip_c_px = elliptical_circumference(f_hip_max_w / 2, s_hip_max_w / 2)

            # 3. Convert Pixels to Final Units
            scale = height_cm / pixel_h
            shoulder_inches = (shoulder_w_px * scale) * 0.3937;
            chest_inches = (chest_c_px * scale) * 0.3937
            waist_inches = (waist_c_px * scale) * 0.3937;
            hip_inches = (hip_c_px * scale) * 0.3937

            # 4. Create and Save Highlighted Images
            timestamp = int(time.time())
            front_highlighted_path = os.path.join(output_dir, f"front_highlighted_{timestamp}.jpg")
            side_highlighted_path = os.path.join(output_dir, f"side_highlighted_{timestamp}.jpg")
            draw_measurement_lines(
                front_path, f_shoulder_y, f_shoulder_x0, f_shoulder_x1,
                f_chest_y, f_chest_x0, f_chest_x1, f_best_y, f_waist_x0, f_waist_x1,
                f_hip_best_y, f_hip_x0, f_hip_x1, front_highlighted_path
            )
            draw_measurement_lines(
                side_path, s_shoulder_y, s_chest_x0, s_chest_x1, s_chest_y, s_chest_x0, s_chest_x1,
                s_best_y, s_waist_x0, s_waist_x1, s_hip_best_y, s_hip_x0, s_hip_x1, side_highlighted_path
            )

            # 5. Format and Return Response
            front_image_b64 = image_to_base64(front_highlighted_path);
            side_image_b64 = image_to_base64(side_highlighted_path)
            return JSONResponse({
                "measurements": {
                    "shoulder_width_in_approx": round(shoulder_inches, 1),
                    "chest_circumference_in_approx": round(chest_inches * 80 / 100, 1),
                    "waist_circumference_in_approx": round(waist_inches, 1),
                    "hip_circumference_in_approx": round(hip_inches, 1)
                },
                "highlighted_images": {"front": front_image_b64, "side": side_image_b64}
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))