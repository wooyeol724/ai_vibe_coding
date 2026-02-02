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
    return "-".join(codes)

@st.cache_data(ttl=600) # 10분 캐싱으로 성능 최적화
def get_full_stock_data(ticker, exchange_rate=1450.0):
    if ticker == "CASH":
        return {"price": 1, "krw": 1, "name": "현금", "is_cash": True, "smart_ticker": "CASH"}
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        full = stock.info
        
        price = info['last_price']
        if price is None: return None
        is_krw = ticker.endswith(".KS") or ticker.endswith(".KQ")
        krw_price = price if is_krw else price * exchange_rate
        prev = info['previous_close']
        change = ((price - prev)/prev)*100 if prev else 0
        
        hist = stock.history(period="6mo")
        techs = {}
        if not hist.empty:
            close = hist['Close']
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            techs['RSI'] = 100 - (100 / (1 + rs)).iloc[-1]
            techs['History'] = hist

        targets = {
            "low": full.get('targetLowPrice'), "mean": full.get('targetMeanPrice'),
            "high": full.get('targetHighPrice'), "rec": full.get('recommendationKey', '-').upper()
        }
        
        ddgs = DDGS()
        try:
            news_raw = ddgs.news(keywords=f"{ticker} stock analyst ratings price target forecast", max_results=6)
            news_list = [{"id": i+1, "title": n['title'], "source": n['source'], "url": n['url']} for i, n in enumerate(news_raw)]
        except: news_list = []

        return {
            "ticker": ticker, "name": full.get('shortName', ticker),
            "price": price, "krw_price": krw_price, "currency": "KRW" if is_krw else "USD",
            "change": change, "smart_ticker": generate_smart_ticker(full),
            "financials": {
                "PER": full.get('trailingPE'), "PEG": full.get('pegRatio'),
                "ROE": full.get('returnOnEquity', 0), "RevGrowth": full.get('revenueGrowth', 0),
                "Margin": full.get('profitMargins', 0)
            },
            "techs": techs, "targets": targets, "news": news_list,
            "sector": full.get('sector', 'Other'), "beta": full.get('beta', 1.0),
            "is_cash": ticker in CASH_EQUIVALENTS
        }
    except: return None

# =========================================================
# 3. AI 분석 엔진
# =========================================================
def analyze_integrated_report(api_key, data):
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1/solar")
    
    f = data['financials']
    hard_data = f"PER: {f['PER']}, ROE: {f['ROE']}, Margin: {f['Margin']}, RSI: {data['techs'].get('RSI')}"
    news_text = "\n".join([f"News {n['id']}: {n['title']} ({n['source']})" for n in data['news']])
    
    prompt = f"""
    당신은 톱티어 분석가 'Assistment'입니다. '{data['name']}' 분석.
    
    [데이터] {hard_data}
    [뉴스] {news_text}
    
    **필수 과제 (변경 불가):**
    1. **5대 기준 평가 (10점 만점) & 벤치마크:**
       - 각 기준별로 점수를 매기고, **"해당 기준에서 10점 만점을 받는 대표적인 글로벌 기업(롤모델)"**과 **"그 이유"**를 함께 제시하세요.
       - 예: Valuation 10점 기업 = 'KO (안정적 배당과 저평가)', Growth 10점 기업 = 'NVDA (폭발적 AI 성장)'
       
    2. **3-Way 예측 (변화율 % 포함):** 가이던스, 월가, AI Agent.
    3. **Top Analysts (실명):** 주요 애널리스트 의견.
    4. **뉴스 임팩트:** 호재/악재 분류.
    
    **JSON 출력:**
    {{
        "scores": {{
            "Valuation": {{ "score": 0, "reason": "...", "benchmark_corp": "...", "benchmark_reason": "..." }},
            "Profitability": {{ "score": 0, "reason": "...", "benchmark_corp": "...", "benchmark_reason": "..." }},
            "Moat_Tech": {{ "score": 0, "reason": "...", "benchmark_corp": "...", "benchmark_reason": "..." }},
            "Growth": {{ "score": 0, "reason": "...", "benchmark_corp": "...", "benchmark_reason": "..." }},
            "Momentum": {{ "score": 0, "reason": "...", "benchmark_corp": "...", "benchmark_reason": "..." }}
        }},
        "forecast": {{
            "guidance_view": "...", "guidance_change": "+5% ~ +8%",
            "market_view": "...", "market_change": "...",
            "ai_view": "...", "ai_change": "...",
            "ai_target_price": 150.0
        }},
        "top_analysts": [ {{ "name": "...", "view": "..." }} ],
        "news_impact": [ {{ "ref_id": 1, "type": "Positive", "reason": "..." }} ],
        "summary": "요약"
    }}
    """
    try:
        res = client.chat.completions.create(model="solar-1-mini-chat", messages=[{"role": "user", "content": prompt}], temperature=0.1)
        return json.loads(res.choices[0].message.content.replace("```json","").replace("```","").strip())
    except: return {"error": "분석 실패"}

def analyze_guru_portfolio(api_key, pf_text, guru):
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1/solar")
    prompt = f"당신은 {guru}입니다. 포트폴리오 평가:\n{pf_text}"
    try:
        res = client.chat.completions.create(model="solar-1-mini-chat", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content
    except: return "오류"

def calculate_portfolio_score(pf_data, total_val):
    if total_val == 0: return 0, {}
    
    w_roe = sum([d['roe'] * (d['val']/total_val) for d in pf_data if not d['is_cash']])
    fund = min(20, (w_roe * 100))
    
    secs = {}
    for d in pf_data: secs[d['sector']] = secs.get(d['sector'], 0) + d['val']
    max_sec = max(secs.values())/total_val if secs else 1.0
    sec = 20 * (1 - max_sec) + 5
    
    avg_beta = sum([d['beta'] * (d['val']/total_val) for d in pf_data])
    beta = max(0, 20 - (abs(1.0 - avg_beta) * 20))
    
    c_val = sum([d['val'] for d in pf_data if d['is_cash']])
    c_r = c_val / total_val
    cash = (c_r/0.2)*20 if c_r <= 0.2 else max(0, 20-((c_r-0.2)*20))
    
    return min(100, fund + sec + beta + cash + 20), {
        "Fundamental": fund, "Diversification": sec, "Stability": beta, "Cash": cash
    }

def plot_prediction_chart(current, targets, ai_target):
    fig = go.Figure()
    l, m, h = targets.get('low'), targets.get('mean'), targets.get('high')
    if l and h: fig.add_trace(go.Scatter(x=[l, h], y=["Target", "Target"], mode='lines', line=dict(color='gray'), name='Range'))
    if m: fig.add_trace(go.Scatter(x=[m], y=["Target"], mode='markers', marker=dict(color='blue', size=10), name='Wall St Avg'))
    fig.add_trace(go.Scatter(x=[current], y=["Target"], mode='markers', marker=dict(color='black', size=10, symbol='x'), name='Now'))
    if ai_target: fig.add_trace(go.Scatter(x=[ai_target], y=["Target"], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='AI Pick'))
    fig.update_layout(title="주가 예측", xaxis_title="가격", yaxis_visible=False, height=200, margin=dict(t=30,b=20,l=10,r=10))
    return fig

# =========================================================
# 4. 메인 UI
# =========================================================
with st.sidebar:
    st.header("👤 내 지갑")
    user_id = st.text_input("ID", "user1")
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_user_data(user_id)
    
    c1, c2 = st.columns(2)
    if c1.button("📂 로드"): 
        st.session_state['portfolio'] = load_user_data(user_id)
        st.rerun()
    if c2.button("🗑️ 초기화"):
        st.session_state['portfolio'] = reset_user_data(user_id)
        st.rerun()
    
    st.divider()
    st.subheader("📸 스크린샷 업로드")
    uploaded_file = st.file_uploader("잔고 화면", type=['png', 'jpg'])
    if uploaded_file and "API_KEY" not in USER_API_KEY:
        if st.button("분석 및 추가"):
            with st.spinner("OCR 분석 중..."):
                raw_text = extract_text_from_image_upstage(USER_API_KEY, uploaded_file.getvalue())
                parsed = parse_portfolio_from_text(USER_API_KEY, raw_text)
                if parsed:
                    for t, q in parsed.items():
                        if t not in st.session_state['portfolio']:
                            st.session_state['portfolio'][t] = {'qty': q, 'avg_price': 0.0, 'target_price': 0.0}
                        else:
                            st.session_state['portfolio'][t]['qty'] += q
                    save_user_data(user_id, st.session_state['portfolio'])
                    st.success("완료")
                    st.rerun()
                else: st.error("실패")

st.markdown("<div class='header-box'><h1>🏛️ Assistment Final</h1><p>완벽한 분석 기준과 통합 솔루션</p></div>", unsafe_allow_html=True)

# [개선] 세션 상태 관리 (분석 리포트 초기화 로직)
def on_target_change():
    if 'rep' in st.session_state:
        del st.session_state['rep'] # 타겟 변경 시 리포트 삭제 (초기화)

# 통합 검색창
query = st.text_input("🔍 종목 검색 (구글, 삼성전자...)", key="search_bar")
candidates = search_symbol_yahoo(query) if query else []
target_sel = st.selectbox("선택", ["선택..."] + candidates + ["CASH (현금)"], on_change=on_target_change)
target = "CASH" if "CASH" in target_sel else (target_sel.split(" | ")[0] if target_sel != "선택..." else None)

t_an, t_pf = st.tabs(["📊 종목 분석 & 매매", "💼 포트폴리오 정밀 진단"])

# [Tab 1] 종목 분석 & 매매
with t_an:
    if target:
        with st.spinner("데이터 로딩..."):
            data = get_full_stock_data(target, get_exchange_rate())
        
        if data:
            # 포트폴리오 데이터
            my_asset = st.session_state['portfolio'].get(target, {'qty': 0.0, 'avg_price': 0.0, 'target_price': 0.0})
            mq, ma, mt = my_asset['qty'], my_asset['avg_price'], my_asset['target_price']
            
            # ROI
            cur_val = data['krw_price'] * mq
            buy_val = (ma * mq * get_exchange_rate()) if data['currency'] == "USD" else (ma * mq)
            if data['currency'] == "USD": buy_val = ma * mq * get_exchange_rate()
            else: buy_val = ma * mq
            profit = cur_val - buy_val
            roi = (profit / buy_val * 100) if buy_val > 0 else 0

            # 상단 메트릭
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(data['name'], f"{data['price']:,.2f}", f"{data['change']:.2f}%")
            c2.metric("보유/평단", f"{mq}주", f"@{ma:,.0f}")
            c3.metric("손익(ROI)", f"₩{profit:,.0f}", f"{roi:+.2f}%")
            if mt > 0: c4.progress(min(data['price']/mt, 1.0), text=f"목표(${mt}) 달성률")
            else: c4.info("목표가 미설정")

            sub_t1, sub_t2, sub_t3 = st.tabs(["5대 기준 분석", "예측/전문가", "매매/설정"])
            
            with sub_t1:
                # [개선] 분석 실행 버튼 상태 관리
                # 'rep'에 데이터가 있고, 그 데이터가 현재 target에 대한 것이면 다시 돌리지 않음
                is_analyzed = False
                if 'rep' in st.session_state and st.session_state['rep'].get('ticker') == target:
                    is_analyzed = True
                
                if not is_analyzed:
                    if st.button("🚀 AI 분석 실행", key="gen_rep"):
                        if "API_KEY" in USER_API_KEY: st.error("키 필요")
                        else:
                            with st.spinner("분석 중..."):
                                result = analyze_integrated_report(USER_API_KEY, data)
                                if "error" not in result:
                                    result['ticker'] = target # 현재 타겟 정보 추가
                                    st.session_state['rep'] = result
                                    st.rerun() # 데이터 저장 후 리로드하여 UI 갱신
                                else:
                                    st.error("분석 실패")
                
                # 분석 결과 표시
                if 'rep' in st.session_state and st.session_state['rep'].get('ticker') == target:
                    rep = st.session_state['rep']
                    
                    st.info(f"💡 {rep.get('summary')}")
                    sc = rep.get('scores', {})
                    
                    # [5대 기준 점수 + 벤치마크]
                    total_score = sum([v.get('score', 0) for v in sc.values()])
                    st.metric("종합 점수 (50점 만점)", f"{total_score}점")
                    
                    col_5 = st.columns(5)
                    keys_5 = ["Valuation", "Profitability", "Moat_Tech", "Growth", "Momentum"]
                    labels_5 = ["밸류에이션", "실익/수익성", "대체불가/기술", "성장성", "모멘텀"]
                    
                    for i, k in enumerate(keys_5):
                        item = sc.get(k, {'score':0, 'reason':'-'})
                        with col_5[i]:
                            st.markdown(f"<div class='score-badge'>{item['score']}/10</div>", unsafe_allow_html=True)
                            st.caption(labels_5[i])
                            with st.expander("상세 보기"):
                                st.write(item['reason'])
                                st.markdown(f"<div class='benchmark-box'>🏆 <b>10점 모델: {item.get('benchmark_corp', 'N/A')}</b><br>{item.get('benchmark_reason', 'N/A')}</div>", unsafe_allow_html=True)

                    st.markdown("#### 📰 뉴스 영향")
                    for i in rep.get('news_impact', []):
                        try: rid = int(str(i.get('ref_id', 0)))
                        except: rid = 0
                        title = data['news'][rid-1]['title'] if 0 < rid <= len(data['news']) else "뉴스"
                        itype = i.get('type', 'Neutral')
                        cls = "impact-pos" if "Positive" in itype else ("impact-neg" if "Negative" in itype else "impact-neu")
                        st.markdown(f"<div class='{cls}'><b>{itype}</b> | {title}<br>{i.get('reason')}</div>", unsafe_allow_html=True)

            with sub_t2:
                if 'rep' in st.session_state and st.session_state['rep'].get('ticker') == target:
                    rep = st.session_state['rep']
                    fc = rep.get('forecast', {})
                    st.markdown("#### 🔮 3-Way 예측")
                    c_f1, c_f2, c_f3 = st.columns(3)
                    with c_f1: st.markdown(f"<div class='card'><b>📢 가이던스</b><br>{fc.get('guidance_view')}<br><b>{fc.get('guidance_change', '-')}</b></div>", unsafe_allow_html=True)
                    with c_f2: st.markdown(f"<div class='card'><b>🏙️ 월가</b><br>{fc.get('market_view')}<br><b>{fc.get('market_change', '-')}</b></div>", unsafe_allow_html=True)
                    with c_f3: st.markdown(f"<div class='card'><b>🤖 AI</b><br>{fc.get('ai_view')}<br><b>{fc.get('ai_change', '-')}</b></div>", unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown("#### 🧑‍🏫 Top Analysts")
                    for analyst in rep.get('top_analysts', []):
                        st.markdown(f"<div class='analyst-box'><b>{analyst.get('name', 'Unknown')}</b>: {analyst.get('view')}</div>", unsafe_allow_html=True)

                    st.divider()
                    try: ai_p = float(fc.get('ai_target_price', 0))
                    except: ai_p = 0
                    st.plotly_chart(plot_prediction_chart(data['price'], data['targets'], ai_p), use_container_width=True)
                else: st.info("분석 리포트를 생성하세요.")

            with sub_t3:
                c_e1, c_e2, c_e3 = st.columns(3)
                nq = c_e1.number_input("수량", value=float(mq))
                navg = c_e2.number_input("평단가", value=float(ma))
                ntgt = c_e3.number_input("목표가", value=float(mt))
                if st.button("저장"):
                    if nq > 0: st.session_state['portfolio'][target] = {'qty': nq, 'avg_price': navg, 'target_price': ntgt}
                    else: 
                        if target in st.session_state['portfolio']: del st.session_state['portfolio'][target]
                    save_user_data(user_id, st.session_state['portfolio'])
                    st.success("저장됨")
                    st.rerun()
    else: st.info("종목을 검색하세요.")

# [Tab 2] 포트폴리오 진단
with t_pf:
    if st.session_state['portfolio']:
        if st.button("🚀 전체 진단 실행", type="primary"):
            with st.spinner("진단 중..."):
                items, tot, txt = [], 0, ""
                for t, info in st.session_state['portfolio'].items():
                    d = get_full_stock_data(t, get_exchange_rate())
                    if d:
                        v = d['krw_price'] * info['qty']
                        tot += v
                        items.append({"ticker":t, "val":v, "sector":d['sector'], "beta":d['beta'], "roe":d['financials']['ROE'], "is_cash":d['is_cash']})
                        txt += f"{t}: {v:.0f}원, "
                
                sc, dt = calculate_portfolio_score(items, tot)
                st.session_state['pf_res'] = (sc, dt, tot, txt)
        
        if 'pf_res' in st.session_state:
            sc, dt, tot, txt = st.session_state['pf_res']
            c1, c2 = st.columns([1, 1])
            c1.metric("총 자산", f"₩{tot:,.0f}")
            c2.metric("종합 점수", f"{sc}점")
            
            cols = st.columns(4)
            cols[0].metric("기본", f"{dt['Fundamental']:.0f}")
            cols[1].metric("분산", f"{dt['Diversification']:.0f}")
            cols[2].metric("안정", f"{dt['Stability']:.0f}")
            cols[3].metric("현금", f"{dt['Cash']:.0f}")
            st.progress(sc/100)
            
            guru = st.selectbox("조언자", ["Ray Dalio", "Peter Lynch"])
            if st.button("조언 듣기"):
                with st.spinner("생성 중..."):
                    st.info(analyze_guru_portfolio(USER_API_KEY, txt, guru))
    else:
        st.warning("포트폴리오가 비었습니다.")
