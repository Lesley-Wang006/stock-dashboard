# ─────────────────────────────────────────────────────
# Stock Signal Dashboard
# Run locally: streamlit run app.py
# Install: pip install streamlit yfinance ta anthropic
#          newsapi-python vaderSentiment plotly feedparser
#          openai requests
# ─────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import feedparser
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Signal Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

LANG = {
    "English": {
        "title":"Stock Signal Dashboard","theme":"Theme","dark":"Dark","pink":"Pink",
        "stock_tab":"Single Stock","portfolio_tab":"Portfolio","chat_tab":"AI Chat",
        "stock_label":"Enter any ticker (e.g. AAPL, NVDA, RY.TO, BTC-USD)",
        "date_label":"Date","ai_provider":"AI Provider",
        "gemini_key":"Google Gemini Key (free at aistudio.google.com)",
        "claude_key":"Anthropic Claude Key (console.anthropic.com)",
        "deepseek_key":"DeepSeek Key (platform.deepseek.com)",
        "news_key":"NewsAPI Key (optional — newsapi.org)",
        "not_advice":"⚠️ Not financial advice",
        "loading":"Loading data...","fetching_news":"Fetching news...",
        "close":"Close","rsi":"RSI (14)","macd":"MACD","pe":"P/E",
        "mktcap":"Market Cap","beta":"Beta","dividend":"Dividend",
        "week52":"52-Week","next_earnings":"Earnings","signal_score":"Score",
        "signal_votes":"Signal Breakdown","ai_headlines":"Headlines",
        "price_chart":"Price Chart","ask_ai":"Ask AI","quick_q":"Quick Questions",
        "buy_today":"Buy today?","explain_rsi":"Explain RSI",
        "news_impact":"News impact","risk_check":"Risk check","ma_cross":"MA cross?",
        "type_q":"Type your question...","clear_chat":"Clear chat",
        "portfolio":"Portfolio Builder","add_stock":"Add stock","weight":"Weight %",
        "portfolio_score":"Portfolio Score","diversification":"Diversification",
        "bullish":"BULLISH","bearish":"BEARISH","neutral":"NEUTRAL",
        "overbought":"Overbought","oversold":"Oversold",
        "bull_momentum":"Bullish momentum","bear_momentum":"Bearish momentum",
        "falling_rates":"Falling rates","rising_rates":"Rising rates",
        "stable_rates":"Stable rates","pos_news":"Positive news",
        "neg_news":"Negative news","neut_news":"Neutral news",
        "more_calls":"More calls — bullish","heavy_puts":"Heavy puts — bearish",
        "balanced":"Balanced flow","above_50ma":"Above 50-day MA",
        "below_50ma":"Below 50-day MA","golden_cross":"Golden cross",
        "death_cross":"Death cross","high_vol":"High volume","low_vol":"Low volume",
        "morning_brief":"AI Morning Brief","days_to_earn":"days to earnings",
    },
    "Français": {
        "title":"Tableau de Bord Signal","theme":"Thème","dark":"Sombre","pink":"Rose",
        "stock_tab":"Action Unique","portfolio_tab":"Portefeuille","chat_tab":"Chat IA",
        "stock_label":"Entrez un symbole (ex: AAPL, NVDA, RY.TO)",
        "date_label":"Date","ai_provider":"Fournisseur IA",
        "gemini_key":"Clé Google Gemini (gratuit — aistudio.google.com)",
        "claude_key":"Clé Claude (console.anthropic.com)",
        "deepseek_key":"Clé DeepSeek (platform.deepseek.com)",
        "news_key":"Clé NewsAPI (optionnel)",
        "not_advice":"⚠️ Pas de conseil financier",
        "loading":"Chargement...","fetching_news":"Récupération...",
        "close":"Clôture","rsi":"RSI (14)","macd":"MACD","pe":"P/E",
        "mktcap":"Capitalisation","beta":"Bêta","dividend":"Dividende",
        "week52":"52 Sem","next_earnings":"Résultats","signal_score":"Score",
        "signal_votes":"Détail Signaux","ai_headlines":"Actualités",
        "price_chart":"Graphique","ask_ai":"Demander à l'IA","quick_q":"Questions Rapides",
        "buy_today":"Acheter?","explain_rsi":"RSI?","news_impact":"Impact nouvelles",
        "risk_check":"Risques","ma_cross":"Croisement MA?",
        "type_q":"Votre question...","clear_chat":"Effacer",
        "portfolio":"Portefeuille","add_stock":"Ajouter","weight":"Poids %",
        "portfolio_score":"Score Portefeuille","diversification":"Diversification",
        "bullish":"HAUSSIER","bearish":"BAISSIER","neutral":"NEUTRE",
        "overbought":"Suracheté","oversold":"Survendu",
        "bull_momentum":"Dynamique haussière","bear_momentum":"Dynamique baissière",
        "falling_rates":"Taux en baisse","rising_rates":"Taux en hausse",
        "stable_rates":"Taux stables","pos_news":"Actualités positives",
        "neg_news":"Actualités négatives","neut_news":"Neutres",
        "more_calls":"Plus de calls","heavy_puts":"Puts importants",
        "balanced":"Équilibré","above_50ma":"Au-dessus MA50",
        "below_50ma":"Sous MA50","golden_cross":"Croix dorée",
        "death_cross":"Croix de mort","high_vol":"Volume élevé","low_vol":"Volume faible",
        "morning_brief":"Résumé Matinal IA","days_to_earn":"jours avant résultats",
    },
    "简体中文": {
        "title":"股票信号仪表板","theme":"主题","dark":"深色","pink":"粉色",
        "stock_tab":"单股分析","portfolio_tab":"投资组合","chat_tab":"AI 对话",
        "stock_label":"输入股票代码（如 AAPL、NVDA、RY.TO）",
        "date_label":"日期","ai_provider":"AI 提供商",
        "gemini_key":"Google Gemini 密钥（免费）","claude_key":"Claude 密钥",
        "deepseek_key":"DeepSeek 密钥","news_key":"NewsAPI 密钥（可选）",
        "not_advice":"⚠️ 非投资建议",
        "loading":"加载中...","fetching_news":"获取新闻...",
        "close":"收盘价","rsi":"RSI","macd":"MACD","pe":"市盈率",
        "mktcap":"市值","beta":"贝塔","dividend":"股息","week52":"52周",
        "next_earnings":"财报","signal_score":"评分",
        "signal_votes":"信号详情","ai_headlines":"新闻",
        "price_chart":"价格图表","ask_ai":"咨询 AI","quick_q":"快速提问",
        "buy_today":"买入？","explain_rsi":"RSI解释","news_impact":"新闻影响",
        "risk_check":"风险检查","ma_cross":"均线交叉？",
        "type_q":"输入问题...","clear_chat":"清除",
        "portfolio":"投资组合","add_stock":"添加","weight":"权重%",
        "portfolio_score":"组合评分","diversification":"分散度",
        "bullish":"看涨","bearish":"看跌","neutral":"中性",
        "overbought":"超买","oversold":"超卖",
        "bull_momentum":"看涨动能","bear_momentum":"看跌动能",
        "falling_rates":"利率下降","rising_rates":"利率上升",
        "stable_rates":"利率稳定","pos_news":"正面新闻",
        "neg_news":"负面新闻","neut_news":"中性新闻",
        "more_calls":"看涨期权多","heavy_puts":"看跌期权多",
        "balanced":"期权平衡","above_50ma":"高于50日均线",
        "below_50ma":"低于50日均线","golden_cross":"金叉",
        "death_cross":"死叉","high_vol":"成交量高","low_vol":"成交量低",
        "morning_brief":"AI 早间简报","days_to_earn":"天后财报",
    }
}

# ── Session state ─────────────────────────────────────
for k,v in [("messages",[]),("theme","Dark"),("lang","English"),
            ("ai_provider","Google Gemini (Free)"),("portfolio",[]),("last_q","")]:
    if k not in st.session_state: st.session_state[k]=v

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    lang = st.selectbox("🌐 Language",["English","Français","简体中文"],
                        index=["English","Français","简体中文"].index(st.session_state.lang))
    st.session_state.lang = lang
    T = LANG[lang]

    st.markdown(f"### {T['theme']}")
    theme = st.radio("Theme",options=[T['dark'],T['pink']],
                     index=0 if st.session_state.theme==T['dark'] else 1,
                     horizontal=True, label_visibility="collapsed")
    st.session_state.theme = theme
    is_pink = theme == T['pink']

    st.markdown("---")
    st.markdown(f"### {T['ai_provider']}")
    AI_OPTIONS = ["Google Gemini (Free)","Claude (Anthropic)","DeepSeek"]
    ai_choice = st.radio("AI Provider", AI_OPTIONS,
                          index=AI_OPTIONS.index(st.session_state.ai_provider)
                                if st.session_state.ai_provider in AI_OPTIONS else 0,
                          label_visibility="collapsed")
    st.session_state.ai_provider = ai_choice

    GEMINI_KEY = ANTHROPIC_KEY = DEEPSEEK_KEY = ""
    if ai_choice == "Google Gemini (Free)":
        GEMINI_KEY = st.text_input(T['gemini_key'], type="password", placeholder="AIzaSy...")
    elif ai_choice == "Claude (Anthropic)":
        ANTHROPIC_KEY = st.text_input(T['claude_key'], type="password", placeholder="sk-ant-...")
    else:
        DEEPSEEK_KEY = st.text_input(T['deepseek_key'], type="password", placeholder="sk-...")

    st.markdown("---")
    NEWSAPI_KEY = st.text_input(T['news_key'], type="password", placeholder="leave empty = RSS only")
    st.markdown("---")
    st.caption(T['not_advice'])

T       = LANG[st.session_state.lang]
is_pink = st.session_state.theme == T['pink']

# ── Theme colors ──────────────────────────────────────
if is_pink:
    BG,BG2,BG3 = "#fff0f5","#ffe4ef","#ffd6e7"
    BORDER = "#ffb3d1"
    TEXT,TEXT2,TEXT3 = "#2d0a1a","#6b2d45","#c47a95"
    ACCENT = "#e91e8c"
    BULL,BEAR,NEUT = "#00a86b","#e91e8c","#ff8c42"
    FONT,RADIUS,CHART_BG = "Georgia,serif","14px","#ffe4ef"
else:
    BG,BG2,BG3 = "#0a0c0f","#111318","#181c22"
    BORDER = "#1e2530"
    TEXT,TEXT2,TEXT3 = "#ffffff","#a0aab8","#4a5568"
    ACCENT = "#ffffff"
    BULL,BEAR,NEUT = "#00d084","#ff4d6a","#ffc940"
    FONT,RADIUS,CHART_BG = "'IBM Plex Mono',monospace","6px","#111318"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
html,body,[class*="css"],.stApp,.main,.block-container{{background:{BG}!important;color:{TEXT}!important;font-family:{FONT};}}
section[data-testid="stSidebar"]{{background:{BG2}!important;border-right:1px solid {BORDER};}}
.mc{{background:{BG2};border:1px solid {BORDER};border-radius:{RADIUS};padding:12px 14px;margin:3px 0;}}
.ml{{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:4px;}}
.mv{{font-size:16px;font-weight:700;}}.ms{{font-size:11px;color:{TEXT3};margin-top:2px;}}
.cb{{background:rgba(0,208,132,.1);border:1px solid rgba(0,208,132,.3);border-radius:{RADIUS};padding:16px 20px;margin:10px 0;}}
.cr{{background:rgba(255,77,106,.1);border:1px solid rgba(255,77,106,.3);border-radius:{RADIUS};padding:16px 20px;margin:10px 0;}}
.cn{{background:rgba(255,140,66,.08);border:1px solid rgba(255,140,66,.3);border-radius:{RADIUS};padding:16px 20px;margin:10px 0;}}
.vr{{font-size:12px;padding:6px 0;border-bottom:1px solid {BORDER};color:{TEXT2};}}
.ni{{padding:8px 0;border-bottom:1px solid {BORDER};}}
.cu{{background:{BG3};border-radius:{RADIUS};padding:10px 14px;margin:6px 0;color:{TEXT};font-size:13px;}}
.ca{{background:{BG2};border:1px solid {BORDER};border-radius:{RADIUS};padding:10px 14px;margin:6px 0;color:{TEXT2};font-size:13px;line-height:1.7;}}
.sl{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin:14px 0 7px;}}
.brief{{background:{BG2};border-left:3px solid {ACCENT};padding:12px 16px;margin:8px 0;font-size:13px;color:{TEXT2};line-height:1.8;border-radius:0 {RADIUS} {RADIUS} 0;}}
h1,h2,h3{{color:{TEXT}!important;}}p,li,span{{color:{TEXT2};}}
.stButton>button{{background:{ACCENT}!important;color:{'#fff' if is_pink else '#000'}!important;border:none!important;border-radius:{'20px' if is_pink else '4px'}!important;font-size:12px!important;padding:6px 12px!important;}}
.stTextInput>div>div>input{{background:{BG2}!important;border:1px solid {BORDER}!important;color:{TEXT}!important;border-radius:{RADIUS}!important;}}
.stSelectbox>div>div{{background:{BG2}!important;color:{TEXT}!important;}}
.stTabs [data-baseweb="tab"]{{background:{BG2};color:{TEXT2};}}
.stTabs [aria-selected="true"]{{background:{BG3}!important;color:{TEXT}!important;border-bottom:2px solid {ACCENT}!important;}}
</style>""", unsafe_allow_html=True)

# ── AI call functions ─────────────────────────────────
def call_gemini(prompt, key):
    try:
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
        res  = requests.post(url, json={"contents":[{"parts":[{"text":prompt}]}]}, timeout=30)
        data = res.json()
        if 'error' in data: return f"Gemini error: {data['error']['message']}"
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e: return f"Gemini error: {e}"

def call_claude(prompt, key, system="You are a financial analyst. Keep answers concise. Not financial advice."):
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500,
                system=system, messages=[{"role":"user","content":prompt}])
        return msg.content[0].text
    except Exception as e: return f"Claude error: {e}"

def call_deepseek(prompt, key):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        r = client.chat.completions.create(model="deepseek-chat", max_tokens=500,
                messages=[{"role":"user","content":prompt}])
        return r.choices[0].message.content
    except Exception as e: return f"DeepSeek error: {e}"

def ask_ai(prompt, provider, gemini_key, anthropic_key, deepseek_key, lang):
    note = {"Français":"Répondez en français.","简体中文":"请用简体中文回答。"}.get(lang,"")
    full_prompt = f"{note} {prompt}" if note else prompt
    if provider=="Google Gemini (Free)" and gemini_key.strip():
        return call_gemini(full_prompt, gemini_key)
    elif provider=="Claude (Anthropic)" and anthropic_key.strip():
        return call_claude(full_prompt, anthropic_key)
    elif provider=="DeepSeek" and deepseek_key.strip():
        return call_deepseek(full_prompt, deepseek_key)
    return {"Français":"Ajoutez une clé API.","简体中文":"请添加API密钥。"}.get(lang,"Add an API key in the sidebar.")

# ── Data loading (Optimized & Cached) ──────────────────
@st.cache_data(ttl=3600)
def load_stock(ticker):
    try:
        df = yf.download(ticker, start="2006-01-01", end=datetime.today(), progress=False)
        if len(df)==0: return None,None
        df.index = pd.to_datetime(df.index)
        if isinstance(df.columns,pd.MultiIndex): df.columns=[c[0] for c in df.columns]
        df['RSI']         = ta.momentum.RSIIndicator(df['Close'],window=14).rsi()
        macd              = ta.trend.MACD(df['Close'])
        df['MACD']        = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist']   = macd.macd_diff()
        df['ATR']         = ta.volatility.AverageTrueRange(df['High'],df['Low'],df['Close'],window=14).average_true_range()
        df['MA50']        = df['Close'].rolling(50).mean()
        df['MA200']       = df['Close'].rolling(200).mean()
        df['Vol_avg']     = df['Volume'].rolling(20).mean()
        df['rate_change'] = 0.0
        try:
            rates = yf.download("^TNX",start="2006-01-01",end=datetime.today(),progress=False)
            if isinstance(rates.columns,pd.MultiIndex): rates.columns=[c[0] for c in rates.columns]
            rates['rc'] = rates['Close'].diff()*100
            df = df.merge(rates[['rc']],left_index=True,right_index=True,how='left')
            df['rate_change'] = df['rc'].ffill().fillna(0)
            df.drop(columns=['rc'],errors='ignore',inplace=True)
        except: pass
        info={}
        try:
            t=yf.Ticker(ticker); i=t.info
            info={'pe':i.get('trailingPE','N/A'),'mktcap':i.get('marketCap',0),
                  'beta':i.get('beta','N/A'),'dividend':i.get('dividendYield',0),
                  'week_high':i.get('fiftyTwoWeekHigh',0),'week_low':i.get('fiftyTwoWeekLow',0)}
            try:
                ed=t.earnings_dates
                if ed is not None and len(ed)>0:
                    future=ed[ed.index>datetime.now(ed.index.tz)]
                    info['next_earnings']=future.index[-1].strftime('%Y-%m-%d') if len(future)>0 else 'N/A'
                else: info['next_earnings']='N/A'
            except: info['next_earnings']='N/A'
        except: pass
        return df.dropna(subset=['RSI','MACD']),info
    except: return None,None

@st.cache_data(ttl=1800)
def get_options(ticker):
    try:
        t=yf.Ticker(ticker)
        if not t.options: return 0.75,25.0
        chain=t.option_chain(t.options[0])
        pc=round(chain.puts['volume'].sum()/max(chain.calls['volume'].sum(),1),2)
        iv=round(((chain.calls['impliedVolatility'].mean()+chain.puts['impliedVolatility'].mean())/2)*100,1)
        return pc,iv
    except: return 0.75,25.0

@st.cache_data(ttl=1800)  # Caching news requests protects your API limits from refreshing continuously
def fetch_headlines(ticker, newsapi_key):
    headlines=[]
    company=ticker.replace('.TO','').replace('-USD','')
    for url in [f"https://finance.yahoo.com/rss/headline?s={ticker}","https://feeds.reuters.com/reuters/businessNews"]:
        try:
            feed=feedparser.parse(url)
            for e in feed.entries[:4]:
                title=e.get('title','')
                if any(k in title.lower() for k in [company.lower(),ticker.lower()]) or 'yahoo' in url:
                    headlines.append(title)
        except: pass
    if newsapi_key and newsapi_key.strip():
        try:
            from newsapi import NewsApiClient
            na=NewsApiClient(api_key=newsapi_key)
            today=datetime.today()
            arts=na.get_everything(q=f'{company} {ticker}',
                     from_param=(today-timedelta(days=1)).strftime("%Y-%m-%d"),
                     to=today.strftime("%Y-%m-%d"),language='en',sort_by='relevancy',page_size=4)
            for a in arts['articles']:
                if a['title'] not in headlines: headlines.append(a['title'])
        except: pass
    return headlines[:8]

@st.cache_data(ttl=1800)  # Caching structural scoring metrics to shield free tiers
def score_headlines(headlines):
    analyzer=SentimentIntensityAnalyzer()
    return [{'headline':h,'score':round(analyzer.polarity_scores(h)['compound']),
             'reason':'VADER sentiment score'} for h in headlines]

def get_conclusion(row, sentiment, pc_ratio, T):
    score,votes=0,[]
    rsi=float(row['RSI']); macd=float(row['MACD']); msig=float(row['MACD_signal'])
    rc=float(row.get('rate_change',0)); cl=float(row['Close'])
    ma50=float(row.get('MA50',cl)); ma200=float(row.get('MA200',cl))
    vol=float(row.get('Volume',0)); vavg=float(row.get('Vol_avg',1))

    if rsi<30:   score+=1;votes.append(('+1',f'RSI {rsi:.1f}',T['oversold']))
    elif rsi>70: score-=1;votes.append(('-1',f'RSI {rsi:.1f}',T['overbought']))
    else:                  votes.append((' 0',f'RSI {rsi:.1f}','Neutral'))
    if macd>msig: score+=1;votes.append(('+1',f'MACD {macd:.2f}',T['bull_momentum']))
    else:         score-=1;votes.append(('-1',f'MACD {macd:.2f}',T['bear_momentum']))
    if rc<-5:   score+=1;votes.append(('+1',f'Rates {rc:+.1f}bps',T['falling_rates']))
    elif rc>5:  score-=1;votes.append(('-1',f'Rates {rc:+.1f}bps',T['rising_rates']))
    else:                 votes.append((' 0',f'Rates {rc:+.1f}bps',T['stable_rates']))
    if sentiment>.2:    score+=1;votes.append(('+1',f'Sent {sentiment:+.2f}',T['pos_news']))
    elif sentiment<-.2: score-=1;votes.append(('-1',f'Sent {sentiment:+.2f}',T['neg_news']))
    else:                         votes.append((' 0',f'Sent {sentiment:+.2f}',T['neut_news']))
    if pc_ratio<.7:   score+=1;votes.append(('+1',f'P/C {pc_ratio}',T['more_calls']))
    elif pc_ratio>1.: score-=1;votes.append(('-1',f'P/C {pc_ratio}',T['heavy_puts']))
    else:                       votes.append((' 0',f'P/C {pc_ratio}',T['balanced']))
    if cl>ma50:    score+=1;votes.append(('+1',f'MA50 ${ma50:.1f}',T['above_50ma']))
    else:          score-=1;votes.append(('-1',f'MA50 ${ma50:.1f}',T['below_50ma']))
    if ma50>ma200: score+=1;votes.append(('+1','MA cross',T['golden_cross']))
    else:          score-=1;votes.append(('-1','MA cross',T['death_cross']))
    if vol>vavg*1.2: score+=1;votes.append(('+1',f'Vol {vol/1e6:.1f}M',T['high_vol']))
    else:             votes.append((' 0',f'Vol {vol/1e6:.1f}M',T['low_vol']))

    label=T['bullish'] if score>=3 else T['bearish'] if score<=-3 else T['neutral']
    return label,score,votes

def fmt_cap(v):
    if not v or v==0: return 'N/A'
    if v>=1e12: return f"${v/1e12:.2f}T"
    if v>=1e9:  return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"

# ── Main UI ───────────────────────────────────────────
st.markdown(f"# {'🌸' if is_pink else '📈'} {T['title']}")
tab1,tab2,tab3 = st.tabs([T['stock_tab'],T['portfolio_tab'],T['chat_tab']])

# ═══════ TAB 1 — SINGLE STOCK ═══════
with tab1:
    c1,c2 = st.columns([2,1])
    with c1:
        TICKER = st.text_input(T['stock_label'],value="AAPL",
                               placeholder="AAPL, NVDA, MSFT, RY.TO, BTC-USD").upper().strip()
    with c2:
        selected_date = st.date_input(T['date_label'],value=datetime.today(),
                                      min_value=datetime(2006,1,1),max_value=datetime.today())
    if not TICKER: st.info("Enter a ticker above"); st.stop()

    with st.spinner(T['loading']): df,info = load_stock(TICKER)
    if df is None: st.error(f"Could not load {TICKER}. Check the symbol."); st.stop()

    available = df.index[df.index<=pd.Timestamp(selected_date)]
    if len(available)==0: st.error("No data for this date"); st.stop()
    row         = df.loc[available[-1]]
    actual_date = available[-1].strftime("%Y-%m-%d")
    pc_ratio,impl_vol = get_options(TICKER)

    with st.spinner(T['fetching_news']):
        headlines  = fetch_headlines(TICKER,NEWSAPI_KEY)
        ai_results = score_headlines(headlines)
        sentiment  = sum(h['score'] for h in ai_results)/max(len(ai_results),1)

    close = float(row['Close'])
    label,score,votes = get_conclusion(row,sentiment,pc_ratio,T)
    prev  = float(df['Close'].iloc[-2]) if len(df)>1 else close
    pct   = (close-prev)/prev*100
    pe    = info.get('pe','N/A')
    ne    = info.get('next_earnings','N/A')
    ne_txt = ne
    if ne!='N/A':
        try:
            d=(datetime.strptime(ne,'%Y-%m-%d')-datetime.today()).days
            ne_txt=f"{ne} ({d}d)"
        except: pass

    # Metric cards
    cols = st.columns(8)
    mets = [
        (T['close'],f"${close:.2f}",f"{pct:+.2f}%",BULL if pct>=0 else BEAR),
        (T['rsi'],f"{float(row['RSI']):.1f}",T['overbought'] if float(row['RSI'])>70 else T['oversold'] if float(row['RSI'])<30 else 'Neutral',BEAR if float(row['RSI'])>70 else BULL if float(row['RSI'])<30 else NEUT),
        (T['macd'],f"{float(row['MACD']):.2f}",f"Sig:{float(row['MACD_signal']):.2f}",BULL if float(row['MACD'])>float(row['MACD_signal']) else BEAR),
        (T['signal_score'],f"{score:+d}/8",label,BULL if score>=3 else BEAR if score<=-3 else NEUT),
        (T['pe'],str(round(pe,1)) if isinstance(pe,(int,float)) else 'N/A',fmt_cap(info.get('mktcap',0)),TEXT2),
        (T['beta'],str(round(info.get('beta',0),2)) if isinstance(info.get('beta'),(int,float)) else 'N/A',f"{info.get('dividend',0)*100:.1f}%",TEXT2),
        (T['week52'],f"${info.get('week_low',0):.0f}–${info.get('week_high',0):.0f}",f"Now ${close:.0f}",TEXT2),
        (T['next_earnings'],ne_txt,'',TEXT2),
    ]
    for col,(lbl,val,sub,clr) in zip(cols,mets):
        with col:
            st.markdown(f'<div class="mc"><div class="ml">{lbl}</div><div class="mv" style="color:{clr}">{val}</div><div class="ms">{sub}</div></div>',unsafe_allow_html=True)

    # Conclusion banner
    css_c='cb' if T['bullish'] in label else 'cr' if T['bearish'] in label else 'cn'
    clr_c=BULL if T['bullish'] in label else BEAR if T['bearish'] in label else NEUT
    bv=[v[1] for v in votes if v[0]=='+1']; bev=[v[1] for v in votes if v[0]=='-1']
    rsn=(f"{T['bullish']}: "+', '.join(bv)) if T['bullish'] in label else \
        (f"{T['bearish']}: "+', '.join(bev)) if T['bearish'] in label else T['neutral']
    st.markdown(f'<div class="{css_c}"><div style="font-size:20px;font-weight:700;color:{clr_c}">{label} · {actual_date}</div><div style="font-size:12px;color:{clr_c};opacity:.85;margin-top:3px">{rsn}</div></div>',unsafe_allow_html=True)

    # Morning brief — button only, saves quota
    st.markdown(f'<div class="sl">{T["morning_brief"]}</div>',unsafe_allow_html=True)
    if GEMINI_KEY or ANTHROPIC_KEY or DEEPSEEK_KEY:
        if st.button("🤖 Generate Morning Brief"):
            with st.spinner("Generating..."):
                prompt=f"Write a 3-sentence morning brief for {TICKER} on {actual_date}. Price:${close:.2f}, Signal:{label}({score}/8), RSI:{float(row['RSI']):.1f}. What should investors watch today?"
                brief=ask_ai(prompt,st.session_state.ai_provider,GEMINI_KEY,ANTHROPIC_KEY,DEEPSEEK_KEY,st.session_state.lang)
                st.markdown(f'<div class="brief">{brief}</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="brief" style="color:{TEXT3}">Add an API key in the sidebar</div>',unsafe_allow_html=True)

    # Charts + breakdown
    col_l,col_r = st.columns([2,1])
    with col_l:
        st.markdown(f'<div class="sl">{T["price_chart"]} — 180 days</div>',unsafe_allow_html=True)
        recent = df.tail(180)
        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.6,0.2,0.2],vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(x=recent.index,open=recent['Open'],high=recent['High'],
            low=recent['Low'],close=recent['Close'],
            increasing=dict(line=dict(color=BULL,width=1),fillcolor=BULL),
            decreasing=dict(line=dict(color=BEAR,width=1),fillcolor=BEAR),name=TICKER),row=1,col=1)
        fig.add_trace(go.Scatter(x=recent.index,y=recent['MA50'],line=dict(color='#f59e0b',width=1.2,dash='dot'),name='MA50'),row=1,col=1)
        fig.add_trace(go.Scatter(x=recent.index,y=recent['MA200'],line=dict(color='#8b5cf6',width=1.2,dash='dot'),name='MA200'),row=1,col=1)
        vc=[BULL if c>=o else BEAR for c,o in zip(recent['Close'],recent['Open'])]
        fig.add_trace(go.Bar(x=recent.index,y=recent['Volume'],marker_color=vc,name='Vol',opacity=0.6),row=2,col=1)
        fig.add_trace(go.Scatter(x=recent.index,y=recent['Vol_avg'],line=dict(color=NEUT,width=1),name='AvgVol'),row=2,col=1)
        mc=[BULL if v>=0 else BEAR for v in recent['MACD_hist'].fillna(0)]
        fig.add_trace(go.Bar(x=recent.index,y=recent['MACD_hist'],marker_color=mc,name='Hist',opacity=0.7),row=3,col=1)
        fig.add_trace(go.Scatter(x=recent.index,y=recent['MACD'],line=dict(color='#3d9eff',width=1),name='MACD'),row=3,col=1)
        fig.add_trace(go.Scatter(x=recent.index,y=recent['MACD_signal'],line=dict(color='#f59e0b',width=1),name='Signal'),row=3,col=1)
        fig.update_layout(plot_bgcolor=CHART_BG,paper_bgcolor=BG,font=dict(color=TEXT2,size=11),
            xaxis=dict(gridcolor=BORDER,rangeslider_visible=False),
            xaxis2=dict(gridcolor=BORDER),xaxis3=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER),yaxis2=dict(gridcolor=BORDER),yaxis3=dict(gridcolor=BORDER),
            legend=dict(bgcolor='rgba(0,0,0,0)',font=dict(size=10)),
            margin=dict(l=0,r=0,t=10,b=0),height=460,hovermode='x unified')
        st.plotly_chart(fig,use_container_width=True)

    with col_r:
        st.markdown(f'<div class="sl">{T["signal_votes"]}</div>',unsafe_allow_html=True)
        for v,sig,rsn in votes:
            c=BULL if v=='+1' else BEAR if v=='-1' else TEXT3
            st.markdown(f'<div class="vr"><span style="color:{c};font-weight:700">[{v}]</span> <span style="color:{TEXT}">{sig}</span><br><span style="font-size:11px;color:{TEXT3}">{rsn}</span></div>',unsafe_allow_html=True)
        st.markdown(f'<div class="sl">{T["ai_headlines"]}</div>',unsafe_allow_html=True)
        for h in ai_results[:5]:
            c=BULL if h['score']>0 else BEAR if h['score']<0 else TEXT3
            a='▲' if h['score']>0 else '▼' if h['score']<0 else '◆'
            st.markdown(f'<div class="ni"><div style="font-size:12px;color:{c}">{a} {h["headline"][:55]}...</div><div style="font-size:11px;color:{TEXT3};font-style:italic">{h["reason"]}</div></div>',unsafe_allow_html=True)

# ═══════ TAB 2 — PORTFOLIO ═══════
with tab2:
    st.markdown(f"## {T['portfolio']}")
    pa,pw,pb = st.columns([2,1,1])
    with pa: new_t=st.text_input(T['add_stock'],placeholder="AAPL",key="pt").upper().strip()
    with pw: new_w=st.number_input(T['weight'],min_value=1,max_value=100,value=20,key="pw")
    with pb:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("+ Add"):
            if new_t:
                st.session_state.portfolio=[p for p in st.session_state.portfolio if p['ticker']!=new_t]
                st.session_state.portfolio.append({'ticker':new_t,'weight':new_w})
                st.rerun()

    if not st.session_state.portfolio:
        st.info("Add stocks above to build your portfolio")
    else:
        total_w=sum(p['weight'] for p in st.session_state.portfolio)
        results=[]; p_scores=[]
        for p in st.session_state.portfolio:
            tk=p['ticker']; w=p['weight']
            with st.spinner(f"Loading {tk}..."): df_p,info_p=load_stock(tk)
            if df_p is None: continue
            row_p=df_p.iloc[-1]; pc_p,_=get_options(tk)
            lbl_p,sc_p,_=get_conclusion(row_p,0,pc_p,T)
            p_scores.append(sc_p*(w/total_w))
            results.append({'ticker':tk,'weight':w,'pct':w/total_w*100,
                            'close':float(row_p['Close']),'label':lbl_p,'score':sc_p,'rsi':float(row_p['RSI'])})

        ts=sum(p_scores)
        pl=T['bullish'] if ts>=1.5 else T['bearish'] if ts<=-1.5 else T['neutral']
        pc_color=BULL if T['bullish'] in pl else BEAR if T['bearish'] in pl else NEUT
        css_pl='cb' if T['bullish'] in pl else 'cr' if T['bearish'] in pl else 'cn'
        st.markdown(f'<div class="{css_pl}"><div style="font-size:18px;font-weight:700;color:{pc_color}">{T["portfolio_score"]}: {pl} ({ts:+.2f})</div></div>',unsafe_allow_html=True)

        hcols=st.columns([1.5,1,1,1,1,1,0.8])
        for col,h in zip(hcols,['Ticker','Weight','Price','RSI','Signal','Score','Del']):
            col.markdown(f"<div style='font-size:10px;color:{TEXT3};text-transform:uppercase'>{h}</div>",unsafe_allow_html=True)
        for r in results:
            rc=BULL if T['bullish'] in r['label'] else BEAR if T['bearish'] in r['label'] else NEUT
            cs=st.columns([1.5,1,1,1,1,1,0.8])
            with cs[0]: st.markdown(f"<div style='color:{TEXT};font-weight:600'>{r['ticker']}</div>",unsafe_allow_html=True)
            with cs[1]: st.markdown(f"<div style='color:{TEXT2}'>{r['pct']:.1f}%</div>",unsafe_allow_html=True)
            with cs[2]: st.markdown(f"<div style='color:{TEXT2}'>${r['close']:.2f}</div>",unsafe_allow_html=True)
            with cs[3]: st.markdown(f"<div style='color:{BEAR if r['rsi']>70 else BULL if r['rsi']<30 else TEXT2}'>{r['rsi']:.1f}</div>",unsafe_allow_html=True)
            with cs[4]: st.markdown(f"<div style='color:{rc};font-weight:600'>{r['label']}</div>",unsafe_allow_html=True)
            with cs[5]: st.markdown(f"<div style='color:{rc}'>{r['score']:+d}/8</div>",unsafe_allow_html=True)
            with cs[6]:
                if st.button("✕",key=f"rm_{r['ticker']}"):
                    st.session_state.portfolio=[p for p in st.session_state.portfolio if p['ticker']!=r['ticker']]
                    st.rerun()

        if results:
            colors=['#00d084','#ff4d6a','#ffc940','#8b5cf6','#f59e0b','#3d9eff','#ec4899','#10b981']
            fig2=go.Figure(go.Pie(labels=[r['ticker'] for r in results],values=[r['weight'] for r in results],
                hole=0.5,marker=dict(colors=colors[:len(results)]),textfont=dict(color=TEXT)))
            fig2.update_layout(plot_bgcolor=BG,paper_bgcolor=BG,font=dict(color=TEXT2),
                legend=dict(bgcolor='rgba(0,0,0,0)'),margin=dict(l=0,r=0,t=20,b=0),height=260,
                title=dict(text=T['diversification'],font=dict(color=TEXT)))
            st.plotly_chart(fig2,use_container_width=True)

# ═══════ TAB 3 — AI CHAT ═══════
with tab3:
    ci='🌸' if is_pink else '🤖'
    st.markdown(f"## {ci} {T['ask_ai']}")
    st.caption(f"Using {st.session_state.ai_provider} · answers only when you send")

    for msg in st.session_state.messages:
        css_m="cu" if msg['role']=='user' else "ca"
        icon="👤" if msg['role']=='user' else ci
        st.markdown(f'<div class="{css_m}">{icon} {msg["content"]}</div>',unsafe_allow_html=True)

    st.markdown(f'<div class="sl">{T["quick_q"]}</div>',unsafe_allow_html=True)
    q1,q2,q3,q4,q5=st.columns(5)
    quick_q=None
    tk_ctx='AAPL' if 'TICKER' not in dir() else TICKER
    with q1:
        if st.button(T['buy_today'],key="qb1"): quick_q=f"Should I buy {tk_ctx} today?"
    with q2:
        if st.button(T['explain_rsi'],key="qb2"): quick_q=f"Explain RSI for {tk_ctx}"
    with q3:
        if st.button(T['news_impact'],key="qb3"): quick_q=f"News impact on {tk_ctx}?"
    with q4:
        if st.button(T['risk_check'],key="qb4"): quick_q=f"Main risks for {tk_ctx}?"
    with q5:
        if st.button(T['ma_cross'],key="qb5"): quick_q=f"MA crossover signal for {tk_ctx}?"

    with st.form(key="chat_form",clear_on_submit=True):
        user_input=st.text_input(T['type_q'],placeholder=f"e.g. Why is {tk_ctx} bearish?")
        send=st.form_submit_button("Send ↗")

    question=quick_q or (user_input if send and user_input else None)

    if question and st.session_state.last_q != question:
        st.session_state.last_q = question
        st.session_state.messages.append({'role':'user','content':question})
        with st.spinner("..."):
            ans=ask_ai(question,st.session_state.ai_provider,GEMINI_KEY,ANTHROPIC_KEY,DEEPSEEK_KEY,st.session_state.lang)
        st.session_state.messages.append({'role':'assistant','content':ans})
        st.rerun()

    if st.button(T['clear_chat']):
        st.session_state.messages=[]
        st.session_state.last_q=""
        st.rerun()

st.markdown("---")
st.caption(T['not_advice'])