import os
import base64
import time
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

router = APIRouter()
load_dotenv() 

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set")

def to_base64(image_bytes: bytes, content_type: str) -> str:
    return f"data:{content_type};base64," + base64.b64encode(image_bytes).decode("utf-8")

@router.post("/tryon")
async def tryon(
    model_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("auto"),
):
    valid_categories = {"auto", "tops", "bottoms", "one-pieces"}
    if category not in valid_categories:
        raise HTTPException(status_code=400, detail=f"Invalid category. Choose one of {valid_categories}")

    model_bytes = await model_image.read()
    garment_bytes = await garment_image.read()

    model_base64 = to_base64(model_bytes, model_image.content_type)
    garment_base64 = to_base64(garment_bytes, garment_image.content_type)

    try:
        response = requests.post(
            "https://api.fashn.ai/v1/run",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model_image": model_base64,
                "garment_image": garment_base64,
                "category": category
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error initiating try-on: {response.text}")

        prediction_id = response.json().get("id")
        if not prediction_id:
            raise HTTPException(status_code=500, detail="No prediction ID returned.")

        status_url = f"https://api.fashn.ai/v1/status/{prediction_id}"
        while True:
            status_response = requests.get(
                status_url,
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            status_data = status_response.json()

            if status_data.get("status") == "completed":
                output_url = status_data.get("output", [None])[0]
                if output_url:
                    return JSONResponse(content={"tryon_image_url": output_url})
                else:
                    raise HTTPException(status_code=500, detail="No output image URL found.")
            elif status_data.get("status") == "failed":
                error_msg = status_data.get('error', {}).get('message', 'Unknown error')
                raise HTTPException(status_code=500, detail=f"Try-on process failed: {error_msg}")
            else:
                time.sleep(5)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


