import streamlit as st
import requests
import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from duckduckgo_search import DDGS
from datetime import datetime, timedelta

# =========================================================
# 🔑 [필수] API Key 설정
# =========================================================
USER_API_KEY = "up_tBmfMapBCD79mdkpNxYzgbWnOpKf2"

DATA_FILE = "users_v15_stable.json"
CASH_EQUIVALENTS = ["GLD", "IAU", "TLT", "IEF", "SHY", "BIL", "SGOV", "USDKRW=X"]

# =========================================================
# 1. UI 스타일링
# =========================================================
st.set_page_config(page_title="어시스트먼트 파이널", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; background-color: #f4f6f9; }
    
    .card { background: white; padding: 15px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 10px; border: 1px solid #e1e4e8; }
    .header-box { background: linear-gradient(135deg, #000428, #004e92); color: white; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 15px; }
    
    .impact-pos { border-left: 4px solid #28a745; background: #e8f5e9; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    .impact-neg { border-left: 4px solid #dc3545; background: #ffebee; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    .impact-neu { border-left: 4px solid #6c757d; background: #f8f9fa; padding: 10px; margin-bottom: 8px; border-radius: 4px; }
    
    .score-badge { font-size: 1.5em; font-weight: bold; color: #004e92; text-align: center; display: block;}
    .benchmark-box { background-color: #fff3e0; border: 1px solid #ffe0b2; padding: 10px; border-radius: 8px; font-size: 0.85em; margin-top: 5px; color: #e65100; }
    
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. 데이터 엔진 & 유틸리티
# =========================================================
def translate_query_to_ticker(query):
    mapping = {
        "구글": "GOOGL", "마이크로소프트": "MSFT", "마소": "MSFT", "애플": "AAPL",
        "테슬라": "TSLA", "엔비디아": "NVDA", "아마존": "AMZN", "메타": "META",
        "넷플릭스": "NFLX", "코카콜라": "KO", "스타벅스": "SBUX", "amd": "AMD",
        "비트코인": "BTC-USD", "이더리움": "ETH-USD", "삼전": "005930.KS", "삼성전자": "005930.KS"
    }
    return mapping.get(query.replace(" ", ""), query)

def load_user_data(user_id):
    if not os.path.exists(DATA_FILE): return {}
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try: return json.load(f).get(user_id, {})
        except: return {}

def save_user_data(user_id, pf):
    try:
        with open(DATA_FILE, "r") as f: all_data = json.load(f)
    except: all_data = {}
    all_data[user_id] = pf
    with open(DATA_FILE, "w") as f: json.dump(all_data, f)

def reset_user_data(user_id):
    all_data = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            try: all_data = json.load(f)
            except: pass
    if user_id in all_data:
        del all_data[user_id]
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
    return {}

def get_exchange_rate():
    try: return yf.Ticker("KRW=X").fast_info['last_price'] or 1450.0
    except: return 1450.0

def search_symbol_yahoo(query):
    real_query = translate_query_to_ticker(query)
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={real_query}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers).json()
        cands = []
        for q in res.get('quotes', []):
            s = q['symbol']
            n = q.get('shortname', q.get('longname', s))
            if s.endswith(".KS") or s.endswith(".KQ"): cands.insert(0, f"{s} | {n} (한국)")
            else: cands.append(f"{s} | {n}")
        return cands
    except: return []

# --- OCR ---
def extract_text_from_image_upstage(api_key, image_bytes):
    url = "https://api.upstage.ai/v1/document-ai/layout-analysis"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"document": image_bytes}
    try:
        response = requests.post(url, headers=headers, files=files).json()
        elements = response.get("elements", [])
        return " ".join([e.get("text", "") for e in elements if e.get("category") in ["text", "table"]])
    except: return "Error"

def parse_portfolio_from_text(api_key, raw_text):
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1/solar")
    prompt = f"텍스트에서 '종목명'과 '수량' 추출 후 JSON 반환. 한국=.KS, 미국=티커. \n{raw_text[:3000]}"
    try:
        res = client.chat.completions.create(model="solar-1-mini-chat", messages=[{"role": "user", "content": prompt}])
        return json.loads(res.choices[0].message.content.replace("```json", "").replace("```", "").strip())
    except: return {}

# --- 데이터 번들링 ---
def generate_smart_ticker(info):
    if not info: return "N/A"
    codes = []
    if info.get('marketCap', 0) > 10000000000: codes.append("L")
    else: codes.append("M/S")
    if info.get('beta', 1.0) > 1.3: codes.append("HiVol")
    else: codes.append("LoVol")
    if info.get('trailingPE', 0) < 15: codes.append("Value")
    else: codes.append("Growth")
    return "-".join