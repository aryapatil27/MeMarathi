from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add the app folder to Python path
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from predict import predict_news

app = FastAPI()

# Project root
BASE_DIR = APP_DIR.parent

# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class News(BaseModel):
    text: str

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "templates" / "index.html")

@app.post("/predict")
def predict(data: News):
    category = predict_news(data.text)
    return {"category": category}