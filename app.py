from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import re
from requests.exceptions import RequestException
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API key
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not set. Please add it to the .env file.")
    raise ValueError("GROQ_API_KEY is not set. Please add it to the .env file and restart the application.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/plotter", response_class=HTMLResponse)
async def read_plotter(request: Request):
    return templates.TemplateResponse("plotter.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(query: str = Form(...), model: str = Form(...), image: UploadFile = File(None)):
    try:
        # Handle image if provided
        encoded_image = None
        mime_type = None
        if image:
            image_content = await image.read()
            if not image_content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            
            # Check file size (max 5MB)
            if len(image_content) > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image size exceeds 5MB")
            
            # Validate image format
            try:
                img = Image.open(io.BytesIO(image_content))
                img.verify()
                if img.format not in ['JPEG', 'PNG']:
                    raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported")
                mime_type = 'image/jpeg' if img.format == 'JPEG' else 'image/png'
            except Exception as e:
                logger.error(f"Invalid image format: {str(e)}")
                raise HTTPException(status_code=400, detail="Invalid image format. Please upload a JPEG or PNG image")
            
            encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Prepare the message for the API
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add text query
        messages[0]["content"].append({"type": "text", "text": query})
        
        # Add image if available
        if encoded_image:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}
            })

        def make_api_request(model_name, retries=3, backoff_factor=1):
            for attempt in range(retries):
                try:
                    response = requests.post(
                        GROQ_API_URL,
                        json={
                            "model": model_name,
                            "messages": messages,
                            "max_tokens": 1000
                        },
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        timeout=30
                    )
                    if response.status_code == 429:  # Rate limit
                        logger.warning(f"Rate limit hit for {model_name}, retrying in {backoff_factor * 2**attempt}s...")
                        time.sleep(backoff_factor * 2**attempt)
                        continue
                    return response
                except RequestException as e:
                    logger.error(f"API request failed for {model_name}: {str(e)}")
                    if attempt == retries - 1:
                        raise HTTPException(status_code=503, detail=f"Failed to connect to the Groq API for {model_name} after retries")
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later")

        # Validate model parameter
        if model not in ["llama", "llava"]:
            raise HTTPException(status_code=400, detail="Invalid model. Choose 'llama' or 'llava'.")

        # Use the same model for the API request
        api_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        response = make_api_request(api_model)
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            # Remove LaTeX-style $...$ patterns
            cleaned_answer = re.sub(r'\$(.*?)\$', r'\1', answer)
            logger.info(f"Processed response for {model} from {api_model}: {cleaned_answer[:100]}...")
        else:
            logger.error(f"Error from API for {api_model}: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Error from API for {api_model}: {response.status_code}")

        # Return a single response in the format expected by the frontend
        return JSONResponse(status_code=200, content={"response": cleaned_answer})

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)