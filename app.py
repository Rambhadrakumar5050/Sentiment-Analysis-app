from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# Load our own trained sentiment model
classifier = pipeline(
    "sentiment-analysis",
    model="./sentiment-model",
    tokenizer="./sentiment-model"
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form = await request.form()

    text = form["text"]

    # Run our trained model
    result = classifier(text)[0]

    label = result["label"]
    confidence = result["score"]

    # IMDb labels:
    # LABEL_0 = Negative
    # LABEL_1 = Positive
    if label == "LABEL_0":
        label = "NEGATIVE"
    elif label == "LABEL_1":
        label = "POSITIVE"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {
                "text": text,
                "label": label,
                "confidence": confidence
            }
        }
    )