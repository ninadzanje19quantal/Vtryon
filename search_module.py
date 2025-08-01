from fastapi import FastAPI
from dotenv import load_dotenv
import os
import requests
import openai

load_dotenv()
def link_scrapper_perplexity(link: str):
    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "Give me the sizes this product is outfit is available in."
            },
            {
                "role": "user",
                "content": link
            }
        ]
    }
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers).text
    return response

api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

def link_scrapper_openai(link: str):
    completion = client.chat.completions.create(
        model="gpt-4o", # Or other suitable model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts product size information from a website."},
            {"role": "user", "content": f"{link}\nWhat size is this product available in on Amazon."}
        ]
    )
    return completion.choices

#print(link_scrapper_openai("https://www.amazon.com/dp/B07MQPRDS1?th=1&psc=1&language=en_US"))
app = FastAPI()

@app.get("/")
async def home():
    return {"Hello": "World"}

@app.post("/search")
async def search(link: str):
    description = link_scrapper_perplexity(link)
    return description

"""url_list = ["https://www.amazon.com/dp/B07MQPRDS1?th=1&psc=1&language=en_US",
"https://www.amazon.com/dp/B0DK411RVZ?th=1&psc=1&language=en_US",
"https://www.amazon.com/dp/B0B31H865L?th=1&psc=1&language=en_US"
]

for i in url_list:
    response = requests.get(i)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option("detach", True)

    # load and open the webpage
    driver = webdriver.Chrome()
    driver.get(i)
    full_html = driver.page_source
    print(full_html)
    time.sleep(10)
    driver.close()

    break"""

"""
import requests


scraper_api_key = os.getenv("SCRAPER_API")
payload = { 'api_key': scraper_api_key, 'url': 'https://httpbin.org/' }
r = requests.get('https://en.wikipedia.org/wiki/India', params=payload)
print(r.text)
"""