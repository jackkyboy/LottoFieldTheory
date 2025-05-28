# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/clean_lotto_data.py

import pandas as pd
from pathlib import Path

# 🔧 ตั้งค่าทั่วไป (Global Variables)
BASE_DIR = Path("/Users/apichet/Downloads")
INPUT_FILE = BASE_DIR / "lotto_110year.csv"
OUTPUT_FILE = BASE_DIR / "lotto_110year_cleaned.csv"

# 🗓️ ฟังก์ชันแปลงชื่อเดือน + พ.ศ. → ค.ศ.
THAI_MONTHS = {
    'มกราคม': 'January',
    'กุมภาพันธ์': 'February',
    'มีนาคม': 'March',
    'เมษายน': 'April',
    'พฤษภาคม': 'May',
    'มิถุนายน': 'June',
    'กรกฎาคม': 'July',
    'สิงหาคม': 'August',
    'กันยายน': 'September',
    'ตุลาคม': 'October',
    'พฤศจิกายน': 'November',
    'ธันวาคม': 'December'
}

def convert_thai_date(thai_date_str):
    try:
        for month_th, month_en in THAI_MONTHS.items():
            if month_th in thai_date_str:
                parts = thai_date_str.split(month_th)
                day = parts[0].strip()
                year_th = parts[1].strip()
                year_ad = int(year_th) - 543
                return f"{day} {month_en} {year_ad}"
        return thai_date_str
    except Exception as e:
        print(f"❌ Error converting date: {thai_date_str} → {e}")
        return thai_date_str

# 🚀 โหลดและแปลงข้อมูล
def clean_lotto_data(input_path=INPUT_FILE, output_path=OUTPUT_FILE):
    try:
        df = pd.read_csv(input_path)

        # แปลงวันที่ไทย → อังกฤษ + ค.ศ.
        df['date'] = df['date'].apply(convert_thai_date)

        # เติม 0 ซ้ายให้เลขท้าย 2 ตัว
        if 'last2' in df.columns:
            df['last2'] = df['last2'].astype(str).str.zfill(2)

        print("✅ ข้อมูลหลังแปลงวันที่และเลขท้าย 2 ตัว:")
        print(df.head())

        df.to_csv(output_path, index=False)
        print(f"📁 บันทึกไฟล์ที่: {output_path}")

    except FileNotFoundError:
        print(f"❌ ไม่พบไฟล์: {input_path}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

# ✅ เรียกใช้เมื่อรันไฟล์โดยตรง
if __name__ == "__main__":
    clean_lotto_data()
