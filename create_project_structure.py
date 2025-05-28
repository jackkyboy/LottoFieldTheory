import os

# โครงสร้างโฟลเดอร์และไฟล์พร้อมเนื้อหาเริ่มต้น
structure = {
    "backend/main.py": """from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.lottery_utils import analyze_lottery
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
""",

    "backend/lottery_utils.py": """import pandas as pd

def analyze_lottery(number: str):
    df = pd.read_csv("backend/data/sample_lottery_data.csv")
    matched = df[df['เลขท้าย 2 ตัว'] == number]
    return f"เลข {number} เคยออกทั้งหมด {len(matched)} ครั้ง"
""",

    "backend/stripe_api.py": """import stripe

stripe.api_key = "your-secret-key"

def create_checkout_session():
    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[{
            'price_data': {
                'currency': 'thb',
                'product_data': {'name': 'วิเคราะห์หวย'},
                'unit_amount': 10000,
            },
            'quantity': 1,
        }],
        mode='payment',
        success_url='http://localhost:8000/',
        cancel_url='http://localhost:8000/',
    )
    return {"url": session.url}
""",

    "backend/data/sample_lottery_data.csv": "วันที่,รางวัลที่ 1,เลขท้าย 2 ตัว\n2023-01-01,123456,89\n2023-02-01,654321,35\n",

    "frontend/index.html": """<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>วิเคราะห์หวยไทย</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container mt-5">
    <h2>วิเคราะห์เลขหวยย้อนหลัง</h2>
    <form action="/analyze" method="post">
        <div class="mb-3">
            <label for="number" class="form-label">เลข 2 ตัวท้าย:</label>
            <input type="text" class="form-control" id="number" name="number" maxlength="2">
        </div>
        <button type="submit" class="btn btn-primary">วิเคราะห์</button>
    </form>
    <form action="/pay" method="post" class="mt-3">
        <button class="btn btn-success">ชำระเงินเพื่อสนับสนุน</button>
    </form>
</body>
</html>
""",

    "frontend/result.html": """<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>ผลวิเคราะห์</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container mt-5">
    <h2>ผลการวิเคราะห์</h2>
    <p>{{ result }}</p>
    <a href="/" class="btn btn-secondary">กลับ</a>
</body>
</html>
""",

    "frontend/css/style.css": "/* CSS custom ถ้าต้องการ */",

    "static/js/script.js": "// JavaScript custom ถ้าต้องการ",

    "requirements.txt": "fastapi\nuvicorn\nstripe\npandas\njinja2\npython-multipart\n",

    "README.md": "# Thai Lottery Project\n\nวิเคราะห์ข้อมูลหวยย้อนหลังแบบเบาๆ ด้วย FastAPI + Bootstrap\n"
}


def create_project_structure(base_dir="thai_lottery_project"):
    for path, content in structure.items():
        full_path = os.path.join(base_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ Project '{base_dir}' created successfully!")


if __name__ == "__main__":
    create_project_structure()
