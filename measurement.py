import os
from typing import Tuple
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tempfile import TemporaryDirectory

router = APIRouter()
@router.get("/")
def home():
    return{"Hello": "World"}

half_band = 5  # pixels

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
    label_at_center = labels[center_y, center_x]
    torso_mask = (labels == label_at_center).astype(np.uint8) * 255
    return torso_mask

def get_torso_edges(mask: np.ndarray, y: int, center_x: float) -> Tuple[int, int]:
    cols = np.where(mask[y] > 0)[0]
    if len(cols) < 2:
        raise RuntimeError(f"No silhouette pixels found at row {y}")
    runs = []
    start = cols[0]
    for i in range(1, len(cols)):
        if cols[i] != cols[i-1] + 1:
            runs.append((start, cols[i-1]))
            start = cols[i]
    runs.append((start, cols[-1]))
    for x0, x1 in runs:
        if x0 <= center_x <= x1:
            return x0, x1
    return cols[0], cols[-1]

def elliptical_circumference(a: float, b: float) -> float:
    h = ((a - b)**2) / ((a + b)**2)
    return np.pi * (a + b) * (1 + (3*h) / (10 + np.sqrt(4 - 3*h)))

def process_image(path: str):
    import cv2
    import mediapipe as mp
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        rgb = img[:, :, :3]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        rgb = img

    h, w = mask.shape
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            raise RuntimeError("No pose landmarks detected")
        lm = res.pose_landmarks.landmark
        shoulder_y = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y +
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) * 0.5) * h)
        hip_y      = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_HIP].y +
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y) * 0.5) * h)
        shoulder_x = int(((lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x +
                           lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x) * 0.5) * w)
    return mask, shoulder_y, hip_y, shoulder_x, h

@router.post("/measure")
async def measure(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    height_cm: float = File(...)
):
    with TemporaryDirectory() as tmpdir:
        try:
            front_path = os.path.join(tmpdir, front_image.filename)
            side_path = os.path.join(tmpdir, side_image.filename)
            with open(front_path, "wb") as f:
                f.write(await front_image.read())
            with open(side_path, "wb") as f:
                f.write(await side_image.read())

            front_clean = remove_bg(front_path, tmpdir)
            side_clean = remove_bg(side_path, tmpdir)

            raw_f_mask, f_chest_y, f_waist_center, f_chest_cx, pixel_h = process_image(front_clean)
            s_mask, s_chest_y, s_waist_center, s_chest_cx, _ = process_image(side_clean)

            f_mask = extract_torso_component(raw_f_mask, f_chest_cx, f_chest_y)

            f_chest_x0, f_chest_x1 = get_torso_edges(f_mask, f_chest_y, f_chest_cx)
            s_chest_x0, s_chest_x1 = get_torso_edges(s_mask, s_chest_y, s_chest_cx)
            f_chest_w = f_chest_x1 - f_chest_x0
            s_chest_w = s_chest_x1 - s_chest_x0

            f_best_y, f_best_w = None, float('inf')
            s_best_y, s_best_w = None, float('inf')
            for dy in range(-half_band, half_band+1):
                yf = np.clip(f_waist_center + dy, 0, pixel_h-1)
                ys = np.clip(s_waist_center + dy, 0, s_mask.shape[0]-1)

                x0f, x1f = get_torso_edges(f_mask, yf, f_chest_cx)
                x0s, x1s = get_torso_edges(s_mask, ys, s_chest_cx)
                wf, ws = x1f - x0f, x1s - x0s

                if wf < f_best_w:
                    f_best_w, f_best_y, f_waist_x0, f_waist_x1 = wf, yf, x0f, x1f
                if ws < s_best_w:
                    s_best_w, s_best_y, s_waist_x0, s_waist_x1 = ws, ys, x0s, x1s

            chest_c_px = elliptical_circumference(f_chest_w/2, s_chest_w/2)
            waist_c_px = elliptical_circumference(f_best_w/2, s_best_w/2)
            scale = height_cm / pixel_h
            chest_cm = chest_c_px * scale
            waist_cm = waist_c_px * scale

            #chest_inches = chest_cm * 0.3937
            #waist_inches = waist_cm * 0.3937

            return JSONResponse({
                "chest_circumference_in_inches": round(chest_cm, 1),
                "waist_circumference_in_inches": round(waist_cm, 1)
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
