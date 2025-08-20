import os
from fastapi import APIRouter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import requests
import pandas as pd
from io import StringIO
import json
from typing import Optional

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
username: str = os.getenv("OXYLABS_USERNAME")
password: str = os.getenv("OXYLABS_PASSWORD")

#Payload schema for amazon_data retrieve and matching endpoint
class amazon_product_payload(BaseModel):
    url: str = Field(...,
                      title="Product URL",
                      description="Enter the Amazon Product URL",
                      min_length=3,
                      max_length=1000)
    user_chest_size: float = Field(..., title="Chest Size",
                                   description="User Chest size in inches",
                                   gt=0)
    user_waist_size: float = Field(..., title="Waist Size",
                                   description="User Waist size in inches",
                                   gt=0)
    user_hips_size: float = Field(..., title="Waist Size",
                                   description="User Waist size in inches",
                                   gt=0)
    user_shoulder_size: float = Field(..., title="Waist Size",
                                   description="User Waist size in inches",
                                   gt=0)


#Pydantic schema for gemini output
class user_fit(BaseModel):
    result: str


def get_sizes_gemini(product: str, user_measurements: dict, product_sizes: str):
    client = genai.Client(api_key=gemini_api_key)
    prompt = (f"The following is the product {product} The following are the product's size: {product_sizes}, "
              f"The following are the user measurements in centimeters: {user_measurements}. "
              f"Using the given information tell me wether the appareal matches anything from the list, "
              f"in terms of body fit. The options are [good fit, tall, short, loose, tight]."
              f"Give the response only in one word.")

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents= prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": user_fit
        }
    ).parsed
    return response, prompt

def get_size_chart_tables(size_chart):
    prompt = f"I have scrapped these tables from Amazon product webpage {size_chart}. The product is a apparel. These tables contain size charts of the apparel and its reviews. Your job is to only return me the tables and discard the reviews. Only give me the tables in the response. dont give me anything else in the response"
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    ).choices[0].message.content
    return response

class SizeRecommendationResponse(BaseModel):
    size: str
    reason: Optional[str]

def get_user_size(size_chart, user_measurements, product):
    prompt = (f"The following is the product: {product}"
              f"This is the size chart of a product: {size_chart} "
              f"The following are the measurements of a person :{user_measurements}."
              f" Your job is to determine which size from the size chart is the closest to the "
              f"user measurements. The size you give as the response should be a size from the size chart."
              f"If the product covers only the top of the body (shirt, tshirt, blazer, hoodie, etc) consider only the chest size. "
              f"If the product covers only the bottom of the body (pants, trackpants, boxers, etc)consider only the waist size."
              f"If the product covers the entire body (sundress, gowns) consider both the chest and waist size."
              f"Strictly respond in JSON format as follows: "
              f"{{'size': 'the closest size', 'reason': 'the reason for this fit'}}")



    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    ).choices[0].message.content
    size, reason = response.split(", Reason:")  # Splitting by 'Reason:'
    size = size.replace("Size: ", "").strip()  # Extract the size
    reason = reason.strip()  # Extract the reason

    # Return the response as a JSON-like dictionary using the Pydantic model
    response = SizeRecommendationResponse(size=size, reason=reason)
    return response

def get_sizes_openai(product: str, user_measurements: dict, product_sizes: str, size_chart):
    client = OpenAI(api_key=openai_api_key)
    prompt = f"Consider the product {product} with size {product_sizes} being worn by a person with the following chest and waist measurements in inches: {user_measurements}. The following is the size chart of the product {size_chart}. Based on the measurements provided use the size chart and please decide whether the product will fit as loose, tight, or good fit. Here good fit refers to whether the product is wearable. A tolerence limit of +-2inches for the chest and waist measurements is allowed to be a good fit. Respond with a single word."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    ).choices[0].message.content
    return response, prompt

def mapper(chest_size: float, waist_size: float) -> dict:
    user_size = {}
    chest_size = int(chest_size)
    waist_size = int(waist_size)

    if 0 < chest_size <= 74:
        user_size["chest_size"] = (chest_size, "XXS")
    elif 74 < chest_size <= 81:
        user_size["chest_size"] = (chest_size, "XS")
    elif 81 < chest_size <= 89:
        user_size["chest_size"] = (chest_size, "S")
    elif 89 < chest_size <= 97:
        user_size["chest_size"] = (chest_size, "M")
    elif 97 < chest_size <= 107:
        user_size["chest_size"] = (chest_size, "L")
    elif 107 < chest_size <= 119:
        user_size["chest_size"] = (chest_size, "XL")
    elif 119 < chest_size <= 131:
        user_size["chest_size"] = (chest_size, "XXL")
    elif 131 < chest_size <= 140:
        user_size["chest_size"] = (chest_size, "XXXL")
    elif chest_size < 140:
        user_size["chest_size"] = (chest_size, "Too Large")
    else:
        user_size["chest_size"] = (chest_size, "Invalid Size")


    if 0 < waist_size <= 58:
        user_size["waist_size"] = (waist_size, "XXS")
    elif 58 < waist_size <= 64:
        user_size["waist_size"] = (waist_size, "XS")
    elif 64 < waist_size <= 72:
        user_size["waist_size"] = (waist_size, "S")
    elif 72 < waist_size <= 81:
        user_size["waist_size"] = (waist_size, "M")
    elif 81 < waist_size <= 90:
        user_size["waist_size"] = (waist_size, "L")
    elif 90 < waist_size <= 102:
        user_size["waist_size"] = (waist_size, "XL")
    elif 102 < waist_size <= 114:
        user_size["waist_size"] = (waist_size, "XXL")
    elif 114 < waist_size <= 117:
        user_size["waist_size"] = (waist_size, "XXXL")
    elif waist_size < 117:
        user_size["waist_size"] = (waist_size, "Too Large")
    else:
        user_size["waist_size"] = (waist_size, "Invalid Size")
    return user_size

router = APIRouter()

@router.post("/get-amazon-data")
def get_amazon_data(product: amazon_product_payload):
    payload = {'source': 'amazon_product','query': product.asin ,'parse': True} # Get response.
    response = requests.request('POST','https://realtime.oxylabs.io/v1/queries',
                                auth=(username, password),json=payload,).json()
    product_sizes = response['results'][0]['content']["variation"]
    product_name = str(response['results'][0]['content']["title"])
    for product_data in product_sizes:
        if product_data["selected"] is True:
            measurements = mapper(product.user_chest_size, product.user_waist_size)
            #size_for_user, prompt = get_sizes(product=str(product_name), user_measurements=measurements,
            #                          product_sizes=product_data["dimensions"]["Size"])
            size_for_user, prompt = get_sizes_openai(product=str(product_name), user_measurements=measurements,
                                      product_sizes=product_data["dimensions"]["Size"])

            return {"response": size_for_user, "prompt": prompt}
    return {"response": "Exception"}

@router.post("/get-size-chart")
def get_size_chart(product: amazon_product_payload):
    payload = {
        'source': 'amazon',
        'url': product.url
    }

    response = requests.request(
        'POST',
        'https://realtime.oxylabs.io/v1/queries',
        auth=(username, password),
        json=payload,
    ).text

    response = StringIO(response)

    #Get size charts from the Product URL using
    size_chart = pd.read_html(io=response, flavor="html5lib")
    size_chart = get_size_chart_tables(size_chart)                          #OPEN AI call

    #Extract ASIN from the product URl
    asin = product.url.split("dp/")[1].split("/")[0].split("?")[0]
    payload = {'source': 'amazon_product','query': asin ,'parse': True} # Get response.
    response = requests.request('POST','https://realtime.oxylabs.io/v1/queries',
                                auth=(username, password),json=payload,).json()
    #Extract available product sizes list from the response
    product_sizes = response['results'][0]['content']["variation"]
    #Extract product name from the response
    product_name = str(response['results'][0]['content']["title"])
    #Extract the exact product size
    for product_data in product_sizes:
        if product_data["selected"] is True:
            #Configure the user measurements
            measurements = {"chest_in_inches": product.user_chest_size, "waist_in_inches": product.user_waist_size,
                            "shoulders_in_inches": product.user_shoulder_size, "hips_in_inches": product.user_hips_size}
            size_for_user, prompt = get_sizes_openai(product=str(product_name), user_measurements=measurements,
                                    product_sizes=product_data["dimensions"]["Size"], size_chart=size_chart)
            #user_size = get_user_size(size_chart, measurements, product_name)       #OPEN AI call
            return {"response": size_for_user, "prompt": prompt}
                #, "user_measurements": measurements, "product_sizes": product_data["dimensions"]["Size"], "size_chart": size_chart}
    return None


"""
{
    "shoulder_width_in_approx": 16,
    "chest_circumference_in_approx": 43.5,
    "waist_circumference_in_approx": 32.1,
    "hip_circumference_in_approx": 34.6
}
"""