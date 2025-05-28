from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from thai_lottery_project.backend.quantum_utilities import analyze_lottery
from backend.stripe_api import create_checkout_session

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
def analyze(request: Request, number: str = Form(...)):
    result = analyze_lottery(number)
    return templates.TemplateResponse("result.html", {"request": request, "result": result})

@app.post("/pay")
def pay():
    return create_checkout_session()
