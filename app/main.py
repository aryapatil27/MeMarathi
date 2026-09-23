from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from predict import predict_news


app = FastAPI()


# Path to the project root
BASE_DIR = Path(__file__).resolve().parent.parent


# Serve frontend files (CSS and JS)
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class News(BaseModel):
    text: str


# Serve the HTML page
@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "templates" / "index.html"
    )


# Prediction API
@app.post("/predict")
def predict(data: News):
    category = predict_news(data.text)

    return {
        "category": category
    }