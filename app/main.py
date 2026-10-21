
from pathlib import Path

from app.predict import predict_news
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()

# Project root
APP_DIR = Path(__file__).resolve().parent
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