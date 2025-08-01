import os
import requests
from pprint import pprint
from fastapi import APIRouter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
username: str = os.getenv("OXYLABS_USERNAME")
password: str = os.getenv("OXYLABS_PASSWORD")

#Payload schema for amazon_data retrieve and matching endpoint
class amazon_product_payload(BaseModel):
    asin: str = Field(...,
                      title="Amazon Standard Identification Number",
                      description="Enter the Amazon Standard Identification Number",
                      min_length=3,
                      max_length=10)
    user_chest_size: float = Field(..., title="Chest Size",
                                   description="User Chest size in inches",
                                   gt=0)
    user_waist_size: float = Field(..., title="Waist Size",
                                   description="User Waist size in inches",
                                   gt=0)


#Pydantic schema for gemini output
class user_fit(BaseModel):
    result: str

def get_sizes(user_measurements: dict, product_sizes: str):
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"The following are the product's size: {product_sizes},"
                 f"The following are the user measurements in centimeters: {user_measurements}"
                 f"Using the given information tell me wether the appareal matches anything from the list, "
                 f"in terms of body fit. The options are [good fit, tall, short, loose, tight]"
                 f"If the measurement unit of the apparel is not given consider it in inches.",
        config={
            "response_mime_type": "application/json",
            "response_schema": user_fit
        }
    ).parsed
    return response

def mapper(chest_size: float, waist_size: float) -> dict:
    user_size = {}
    chest_size = int(chest_size)
    waist_size = int(waist_size)

    if 0 < chest_size <= 74:
        user_size["chest_size"] = (chest_size, "XXS")
    if 74 < chest_size <= 81:
        user_size["chest_size"] = (chest_size, "XS")
    if 81 < chest_size <= 89:
        user_size["chest_size"] = (chest_size, "S")
    if 89 < chest_size <= 97:
        user_size["chest_size"] = (chest_size, "M")
    if 97 < chest_size <= 107:
        user_size["chest_size"] = (chest_size, "L")
    if 107 < chest_size <= 119:
        user_size["chest_size"] = (chest_size, "XL")
    if 119 < chest_size <= 131:
        user_size["chest_size"] = (chest_size, "XXL")

    if waist_size <= 58:
        user_size["waist_size"] = [waist_size, "XXS"]
    if 58 < waist_size <= 64:
        user_size["waist_size"] = [waist_size, "XS"]
    if 64 < waist_size <= 72:
        user_size["waist_size"] = [waist_size, "S"]
    if 72 < waist_size <= 81:
        user_size["waist_size"] = [waist_size, "M"]
    if 81 < waist_size <= 90:
        user_size["waist_size"] = [waist_size, "L"]
    if 90 < waist_size <= 102:
        user_size["waist_size"] = [waist_size, "XL"]
    if 102 < waist_size <- 114:
        user_size["waist_size"] = (waist_size, "XXL")
    return user_size

router = APIRouter()

@router.post("/get-amazon-data")
def get_amazon_data(product: amazon_product_payload):
    payload = {'source': 'amazon_product','query': product.asin ,'parse': True} # Get response.
    response = requests.request('POST','https://realtime.oxylabs.io/v1/queries',
                                auth=(username, password),json=payload,).json()
    response = response['results'][0]['content']["variation"]
    for product_data in response:
        if product_data["selected"] is True:
            measurements = mapper(product.user_chest_size, product.user_waist_size)
            size_for_user = get_sizes(user_measurements=measurements,
                                      product_sizes=product_data["dimensions"]["Size"])

            return {"response": size_for_user}
    return {"response": "Exception"}

"""
{
  "chest_circumference_in_inches": 116.6,
  "waist_circumference_in_inches": 87.1
}
"""
