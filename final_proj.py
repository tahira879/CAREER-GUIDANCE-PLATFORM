"""
PathFinder AI — Career Guidance Platform
Theme: PathFinder (Plus Jakarta Sans, #3D52A0/#7091E6/#EDE8F5)
Single file. Run: streamlit run main.py
pip install streamlit groq python-dotenv pandas numpy scikit-learn plotly joblib PyPDF2 docx2txt xgboost
Place career_guidance_dataset.csv and institutes_dataset.csv in the same folder.
"""

import os, io, json, time, re, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from groq import Groq
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib

try:    import PyPDF2;    HAS_PDF  = True
except: HAS_PDF  = False
try:    import docx2txt;  HAS_DOCX = True
except: HAS_DOCX = False

warnings.filterwarnings("ignore")
load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PathFinder AI | Career Compass",
    page_icon="🧭", layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
DEFAULTS = {
    "page":"landing","modal":None,
    "logged_in":False,"current_user":None,"accounts":{},
    "app_page":"home","user_profile":{},"career_matches":None,
    "selected_career":None,"roadmap":None,
    "chat_history":[],"resume_analysis":None,
    "persona_summary":None,"model_results":None,
    "custom_career_input":"",
}
for k,v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k]=v

# ═══════════════════════════════════════════════════════════════════
# CSS — PathFinder Theme (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Syne:wght@700;800&display=swap');
:root{
  --primary:#3D52A0;--secondary:#7091E6;--accent:#ADBBDA;
  --bg:#EDE8F5;--card:#ffffff;
  --text:#1a1a2e;--muted:#8697C4;--border:#d0d7f0;
  --success:#22c55e;--warning:#f59e0b;--danger:#ef4444;
}
*{box-sizing:border-box;}
html,body,[class*="css"]{
  font-family:'Plus Jakarta Sans',sans-serif!important;
  background-color:var(--bg)!important;color:var(--text)!important;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#1e2d6b 0%,#3D52A0 55%,#2a3c8a 100%)!important;
  border-right:3px solid var(--secondary)!important;
}
[data-testid="stSidebar"] *{color:#EDE8F5!important;}
[data-testid="stSidebar"] .stButton>button{
  background:rgba(255,255,255,0.10)!important;
  border:1px solid rgba(255,255,255,0.22)!important;
  color:#EDE8F5!important;border-radius:14px!important;
  font-weight:700!important;padding:10px 16px!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  transition:all 0.2s!important;width:100%!important;text-align:left!important;
}
[data-testid="stSidebar"] .stButton>button:hover{
  background:rgba(255,255,255,0.22)!important;transform:translateX(5px)!important;
}
.pf-nav{
  position:fixed;top:0;width:100%;height:70px;
  background:rgba(237,232,245,0.97);backdrop-filter:blur(20px);
  z-index:1000;display:flex;align-items:center;
  padding:0 5%;border-bottom:2px solid var(--primary);
  box-shadow:0 2px 24px rgba(61,82,160,0.13);
}
.pf-brand{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--primary);letter-spacing:-1px;}
.pf-brand span{color:var(--secondary);}
.hero-wrap{width:100%;height:80vh;overflow:hidden;position:relative;}
.hero-img{width:100%;height:100%;
  background:url('https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?q=80&w=2000') no-repeat center;
  background-size:cover;transition:transform 3s ease;}
.hero-wrap:hover .hero-img{transform:scale(1.07);}
.hero-overlay{
  position:absolute;top:0;left:0;width:100%;height:100%;
  background:linear-gradient(135deg,rgba(13,17,23,.72),rgba(61,82,160,.55));
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:white;text-align:center;padding:0 20px;
}
.hero-title{
  font-family:'Syne',sans-serif;font-size:82px;font-weight:900;streamlit run my.py
  letter-spacing:-4px;margin:0;color:white;
  text-shadow:0 4px 30px rgba(0,0,0,.4);
  animation:heroIn .8s ease forwards;
}
@keyframes heroIn{from{opacity:0;transform:translateY(40px);}to{opacity:1;transform:translateY(0);}}
.glass-card{
  background:rgba(255,255,255,.8);backdrop-filter:blur(12px);
  padding:36px;border-radius:28px;
  border:2px solid rgba(61,82,160,.1);
  transition:all .4s ease;text-align:left;margin-bottom:24px;
  min-height:260px;display:flex;flex-direction:column;justify-content:center;
}
.glass-card:hover{
  background:var(--primary);transform:translateY(-12px) scale(1.02);
  box-shadow:0 32px 64px rgba(61,82,160,.35);border-color:var(--secondary);
}
.glass-card h3{color:var(--primary);font-weight:800;font-size:22px;margin-bottom:14px;transition:.3s;}
.glass-card p{color:#2D3748;font-size:14px;line-height:1.75;transition:.3s;}
.glass-card:hover h3,.glass-card:hover p{color:white!important;}
.pf-card{
  background:white;border-radius:20px;padding:28px;
  border:2px solid rgba(61,82,160,.08);
  box-shadow:0 4px 24px rgba(61,82,160,.08);
  margin-bottom:20px;transition:border-color .2s;
}
.pf-card:hover{border-color:var(--secondary);}
.pf-card-title{
  font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
  color:var(--primary);margin-bottom:16px;display:flex;align-items:center;gap:8px;
}
.match-card{
  background:white;border:2px solid rgba(61,82,160,.1);
  border-radius:18px;padding:22px;margin-bottom:14px;
  position:relative;overflow:hidden;
  transition:all .3s;box-shadow:0 2px 16px rgba(61,82,160,.07);
}
.match-card:hover{box-shadow:0 8px 32px rgba(61,82,160,.18);transform:translateY(-3px);}
.match-card.top{border-color:var(--primary);}
.match-rank{
  font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:900;
  color:var(--primary);opacity:.18;position:absolute;top:10px;right:18px;
}
.match-title{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;margin:0 0 6px;color:var(--text);}
.score-bar-bg{background:#e8eaf6;border-radius:999px;height:6px;margin:6px 0;}
.score-bar-fill{height:6px;border-radius:999px;background:linear-gradient(90deg,var(--primary),var(--secondary));transition:width .8s;}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:700;margin:2px;}
.badge-blue{background:rgba(61,82,160,.12);color:var(--primary);border:1px solid rgba(61,82,160,.2);}
.badge-purple{background:rgba(112,145,230,.15);color:#5865a0;border:1px solid rgba(112,145,230,.3);}
.badge-green{background:rgba(34,197,94,.12);color:#16a34a;border:1px solid rgba(34,197,94,.25);}
.badge-yellow{background:rgba(245,158,11,.12);color:#b45309;border:1px solid rgba(245,158,11,.25);}
.badge-red{background:rgba(239,68,68,.12);color:#dc2626;border:1px solid rgba(239,68,68,.25);}
.badge-teal{background:rgba(20,184,166,.12);color:#0d9488;border:1px solid rgba(20,184,166,.25);}
.auth-wrap{
  background:white;border-radius:28px;padding:48px 52px;
  margin:30px auto;max-width:480px;
  box-shadow:0 8px 48px rgba(61,82,160,.13);
  border:2px solid rgba(61,82,160,.08);
  animation:fadeUp .5s ease;
}
.dash-wrap{
  background:white;border-radius:28px;padding:48px 52px;
  margin:30px auto;max-width:780px;
  box-shadow:0 8px 48px rgba(61,82,160,.13);
  border:2px solid rgba(61,82,160,.08);
  animation:fadeUp .5s ease;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(28px);}to{opacity:1;transform:translateY(0);}}
.dash-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
.dash-field{background:#f3f5fc;border-radius:14px;padding:16px 20px;}
.dash-field .label{font-size:10px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
.dash-field .value{font-size:15px;font-weight:700;color:var(--text);}
.stButton>button{
  background:linear-gradient(90deg,var(--primary),var(--secondary))!important;
  color:white!important;border:none!important;border-radius:14px!important;
  padding:11px 28px!important;font-family:'Plus Jakarta Sans',sans-serif!important;
  font-weight:700!important;font-size:.92rem!important;transition:all .25s!important;
}
.stButton>button:hover{transform:translateY(-3px)!important;box-shadow:0 10px 24px rgba(61,82,160,.3)!important;}
.stTextInput label,.stTextArea label,.stSelectbox label,.stMultiSelect label{
  color:var(--primary)!important;font-size:13px!important;font-weight:700!important;
}
.stTextInput input,.stTextArea textarea{
  background:#f7f8fc!important;border:1.5px solid var(--border)!important;
  border-radius:12px!important;color:var(--text)!important;
}
.stTextInput input:focus,.stTextArea textarea:focus{
  border-color:var(--secondary)!important;
  box-shadow:0 0 0 3px rgba(112,145,230,.18)!important;
}
.stSelectbox>div>div,.stMultiSelect>div>div{
  background:#f7f8fc!important;border:1.5px solid var(--border)!important;
  border-radius:12px!important;color:var(--text)!important;
}
.stTabs [data-baseweb="tab"]{
  color:var(--muted)!important;background:transparent!important;
  border-bottom:2px solid transparent!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;
}
.stTabs [aria-selected="true"]{color:var(--primary)!important;border-bottom:2px solid var(--primary)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
[data-testid="stMetricValue"]{color:var(--primary)!important;font-family:'Syne',sans-serif!important;font-weight:800!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-weight:600!important;}
.chat-user{
  background:linear-gradient(90deg,rgba(61,82,160,.12),rgba(112,145,230,.1));
  border:1px solid rgba(61,82,160,.2);
  border-radius:18px 18px 2px 18px;padding:12px 18px;margin:8px 0;
  max-width:78%;margin-left:auto;color:var(--text);
}
.chat-ai{
  background:white;border:1px solid var(--border);
  border-radius:18px 18px 18px 2px;padding:12px 18px;margin:8px 0;
  max-width:85%;color:var(--text);
  box-shadow:0 2px 12px rgba(61,82,160,.06);
}
.pf-footer{
  background:#0d1117;color:#ADBBDA;
  padding:70px 8% 40px;margin-top:80px;
  border-top:4px solid var(--primary);
}
.footer-grid{display:grid;grid-template-columns:2fr 1fr;gap:60px;}
.footer-bottom{margin-top:50px;padding-top:22px;border-top:1px solid rgba(255,255,255,.07);text-align:center;color:#555;font-size:13px;}
.team-name{color:white;font-weight:700;font-size:15px;margin-bottom:4px;}
.team-link a{color:#7091E6;text-decoration:none;font-size:13px;}
.team-link a:hover{color:white;text-decoration:underline;}
.doodle{position:fixed;opacity:.11;z-index:0;width:78px;pointer-events:none;}
.section-title{font-family:'Syne',sans-serif;font-size:40px;font-weight:900;color:var(--primary);margin-bottom:50px;text-align:center;}
hr{border-color:var(--border)!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:var(--primary)!important;}
.inst-card{border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;
  background:white;box-shadow:0 4px 16px rgba(61,82,160,.07);
  transition:all .3s;margin-bottom:20px;}
.inst-card:hover{transform:translateY(-4px);box-shadow:0 12px 32px rgba(61,82,160,.15);}
.journey-container{display:flex;justify-content:space-between;align-items:center;margin:30px 0 40px 0;position:relative;}
.journey-line{position:absolute;top:25px;left:0;width:100%;height:4px;background:#e2e8f0;z-index:0;}
.journey-step{position:relative;z-index:1;text-align:center;width:30%;}
.step-circle{width:50px;height:50px;border-radius:50%;background:white;border:3px solid #cbd5e1;
  display:flex;align-items:center;justify-content:center;font-size:1.5rem;
  margin:0 auto 10px auto;transition:all 0.3s;}
.step-active .step-circle{border-color:var(--primary);background:var(--primary);color:white;
  box-shadow:0 0 0 4px rgba(61,82,160,0.2);transform:scale(1.1);}
.step-done .step-circle{background:#22c55e;border-color:#22c55e;color:white;}
.step-title{font-weight:bold;color:#1e293b;font-size:0.95rem;}
.step-sub{color:#64748b;font-size:0.8rem;}
</style>
<img src="https://img.icons8.com/ios-filled/100/3D52A0/laptop-coding.png"  class="doodle" style="top:22%;left:2%;">
<img src="https://img.icons8.com/ios-filled/100/3D52A0/brainstorming.png" class="doodle" style="top:58%;right:2%;">
<img src="https://img.icons8.com/ios-filled/100/3D52A0/books.png"          class="doodle" style="bottom:18%;left:3%;">
<img src="https://img.icons8.com/ios-filled/100/3D52A0/goal.png"           class="doodle" style="top:14%;right:4%;">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# CSS — PathFinder Theme (UPDATED WITH HOVER EFFECTS)
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Syne:wght@700;800&display=swap');
:root{
  --primary:#3D52A0;--secondary:#7091E6;--accent:#ADBBDA;
  --bg:#EDE8F5;--card:#ffffff;
  --text:#1a1a2e;--muted:#8697C4;--border:#d0d7f0;
  --success:#22c55e;--warning:#f59e0b;--danger:#ef4444;
}
*{box-sizing:border-box;}
html,body,[class*="css"]{
  font-family:'Plus Jakarta Sans',sans-serif!important;
  background-color:var(--bg)!important;color:var(--text)!important;
}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#1e2d6b 0%,#3D52A0 55%,#2a3c8a 100%)!important;
  border-right:3px solid var(--secondary)!important;
}
[data-testid="stSidebar"] *{color:#EDE8F5!important;}
[data-testid="stSidebar"] .stButton>button{
  background:rgba(255,255,255,0.10)!important;
  border:1px solid rgba(255,255,255,0.22)!important;
  color:#EDE8F5!important;border-radius:14px!important;
  font-weight:700!important;padding:10px 16px!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  transition:all 0.2s!important;width:100%!important;text-align:left!important;
}
[data-testid="stSidebar"] .stButton>button:hover{
  background:rgba(255,255,255,0.22)!important;transform:translateX(5px)!important;
}
.pf-nav{
  position:fixed;top:0;width:100%;height:70px;
  background:rgba(237,232,245,0.97);backdrop-filter:blur(20px);
  z-index:1000;display:flex;align-items:center;
  padding:0 5%;border-bottom:2px solid var(--primary);
  box-shadow:0 2px 24px rgba(61,82,160,0.13);
}
.pf-brand{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--primary);letter-spacing:-1px;}
.pf-brand span{color:var(--secondary);}
.hero-wrap{width:100%;height:80vh;overflow:hidden;position:relative;}
.hero-img{width:100%;height:100%;
  background:url('https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?q=80&w=2000') no-repeat center;
  background-size:cover;transition:transform 3s ease;}
.hero-wrap:hover .hero-img{transform:scale(1.07);}
.hero-overlay{
  position:absolute;top:0;left:0;width:100%;height:100%;
  background:linear-gradient(135deg,rgba(13,17,23,.72),rgba(61,82,160,.55));
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  color:white;text-align:center;padding:0 20px;
}
.hero-title{
  font-family:'Syne',sans-serif;font-size:82px;font-weight:900;
  letter-spacing:-4px;margin:0;color:white;
  text-shadow:0 4px 30px rgba(0,0,0,.4);
  animation:heroIn .8s ease forwards;
  cursor:default;
  transition: all 0.3s ease;
}
/* --- HERO TITLE HOVER (COLOR & NEON) --- */
.hero-title:hover {
  color: var(--accent);
  text-shadow: 0 0 10px #fff, 0 0 20px var(--secondary), 0 0 40px var(--primary);
  transform: scale(1.02);
}
@keyframes heroIn{from{opacity:0;transform:translateY(40px);}to{opacity:1;transform:translateY(0);}}
.glass-card{
  background:rgba(255,255,255,.8);backdrop-filter:blur(12px);
  padding:36px;border-radius:28px;
  border:2px solid rgba(61,82,160,.1);
  transition:all .4s ease;text-align:left;margin-bottom:24px;
  min-height:260px;display:flex;flex-direction:column;justify-content:center;
}
.glass-card:hover{
  background:var(--primary);transform:translateY(-12px) scale(1.02);
  box-shadow:0 32px 64px rgba(61,82,160,.35);border-color:var(--secondary);
}
.glass-card h3{color:var(--primary);font-weight:800;font-size:22px;margin-bottom:14px;transition:.3s;}
.glass-card p{color:#2D3748;font-size:14px;line-height:1.75;transition:.3s;}
.glass-card:hover h3,.glass-card:hover p{color:white!important;}
.pf-card{
  background:white;border-radius:20px;padding:28px;
  border:2px solid rgba(61,82,160,.08);
  box-shadow:0 4px 24px rgba(61,82,160,.08);
  margin-bottom:20px;transition:border-color .2s;
}
.pf-card:hover{border-color:var(--secondary);}
.pf-card-title{
  font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
  color:var(--primary);margin-bottom:16px;display:flex;align-items:center;gap:8px;
}
.match-card{
  background:white;border:2px solid rgba(61,82,160,.1);
  border-radius:18px;padding:22px;margin-bottom:14px;
  position:relative;overflow:hidden;
  transition:all .3s;box-shadow:0 2px 16px rgba(61,82,160,.07);
}
.match-card:hover{box-shadow:0 8px 32px rgba(61,82,160,.18);transform:translateY(-3px);}
.match-card.top{border-color:var(--primary);}
.match-rank{
  font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:900;
  color:var(--primary);opacity:.18;position:absolute;top:10px;right:18px;
}
.match-title{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;margin:0 0 6px;color:var(--text);}
.score-bar-bg{background:#e8eaf6;border-radius:999px;height:6px;margin:6px 0;}
.score-bar-fill{height:6px;border-radius:999px;background:linear-gradient(90deg,var(--primary),var(--secondary));transition:width .8s;}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:700;margin:2px;}
.badge-blue{background:rgba(61,82,160,.12);color:var(--primary);border:1px solid rgba(61,82,160,.2);}
.badge-purple{background:rgba(112,145,230,.15);color:#5865a0;border:1px solid rgba(112,145,230,.3);}
.badge-green{background:rgba(34,197,94,.12);color:#16a34a;border:1px solid rgba(34,197,94,.25);}
.badge-yellow{background:rgba(245,158,11,.12);color:#b45309;border:1px solid rgba(245,158,11,.25);}
.badge-red{background:rgba(239,68,68,.12);color:#dc2626;border:1px solid rgba(239,68,68,.25);}
.badge-teal{background:rgba(20,184,166,.12);color:#0d9488;border:1px solid rgba(20,184,166,.25);}
.auth-wrap{
  background:white;border-radius:28px;padding:48px 52px;
  margin:30px auto;max-width:480px;
  box-shadow:0 8px 48px rgba(61,82,160,.13);
  border:2px solid rgba(61,82,160,.08);
  animation:fadeUp .5s ease;
}
.dash-wrap{
  background:white;border-radius:28px;padding:48px 52px;
  margin:30px auto;max-width:780px;
  box-shadow:0 8px 48px rgba(61,82,160,.13);
  border:2px solid rgba(61,82,160,.08);
  animation:fadeUp .5s ease;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(28px);}to{opacity:1;transform:translateY(0);}}
.dash-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
.dash-field{background:#f3f5fc;border-radius:14px;padding:16px 20px;}
.dash-field .label{font-size:10px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;}
.dash-field .value{font-size:15px;font-weight:700;color:var(--text);}
.stButton>button{
  background:linear-gradient(90deg,var(--primary),var(--secondary))!important;
  color:white!important;border:none!important;border-radius:14px!important;
  padding:11px 28px!important;font-family:'Plus Jakarta Sans',sans-serif!important;
  font-weight:700!important;font-size:.92rem!important;transition:all .25s!important;
}
.stButton>button:hover{transform:translateY(-3px)!important;box-shadow:0 10px 24px rgba(61,82,160,.3)!important;}
.stTextInput label,.stTextArea label,.stSelectbox label,.stMultiSelect label{
  color:var(--primary)!important;font-size:13px!important;font-weight:700!important;
}
.stTextInput input,.stTextArea textarea{
  background:#f7f8fc!important;border:1.5px solid var(--border)!important;
  border-radius:12px!important;color:var(--text)!important;
}
.stTextInput input:focus,.stTextArea textarea:focus{
  border-color:var(--secondary)!important;
  box-shadow:0 0 0 3px rgba(112,145,230,.18)!important;
}
.stSelectbox>div>div,.stMultiSelect>div>div{
  background:#f7f8fc!important;border:1.5px solid var(--border)!important;
  border-radius:12px!important;color:var(--text)!important;
}
.stTabs [data-baseweb="tab"]{
  color:var(--muted)!important;background:transparent!important;
  border-bottom:2px solid transparent!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;
}
.stTabs [aria-selected="true"]{color:var(--primary)!important;border-bottom:2px solid var(--primary)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--border)!important;}
[data-testid="stMetricValue"]{color:var(--primary)!important;font-family:'Syne',sans-serif!important;font-weight:800!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-weight:600!important;}
.chat-user{
  background:linear-gradient(90deg,rgba(61,82,160,.12),rgba(112,145,230,.1));
  border:1px solid rgba(61,82,160,.2);
  border-radius:18px 18px 2px 18px;padding:12px 18px;margin:8px 0;
  max-width:78%;margin-left:auto;color:var(--text);
}
.chat-ai{
  background:white;border:1px solid var(--border);
  border-radius:18px 18px 18px 2px;padding:12px 18px;margin:8px 0;
  max-width:85%;color:var(--text);
  box-shadow:0 2px 12px rgba(61,82,160,.06);
}
.pf-footer{
  background:#0d1117;color:#ADBBDA;
  padding:70px 8% 40px;margin-top:80px;
  border-top:4px solid var(--primary);
}
.footer-grid{display:grid;grid-template-columns:2fr 1fr;gap:60px;}
.footer-bottom{margin-top:50px;padding-top:22px;border-top:1px solid rgba(255,255,255,.07);text-align:center;color:#555;font-size:13px;}
.team-name{color:white;font-weight:700;font-size:15px;margin-bottom:4px;transition: all 0.3s ease;}
.team-link a{color:#7091E6;text-decoration:none;font-size:13px;transition: all 0.3s ease;}
/* --- FOOTER HOVER (NEON SHADOW) --- */
.team-name:hover,
.team-link a:hover{
  color: white;
  text-shadow: 0 0 5px #fff, 0 0 10px var(--primary), 0 0 20px var(--secondary);
  transform: translateX(5px);
}
.doodle{position:fixed;opacity:.11;z-index:0;width:78px;pointer-events:auto;transition: all 0.4s ease;}
/* --- DOODLE HOVER (SCALE & GLOW) --- */
.doodle:hover{
  opacity: 1 !important;
  transform: scale(1.2) rotate(10deg);
  filter: drop-shadow(0 0 15px var(--primary));
  z-index: 10;
}
.section-title{font-family:'Syne',sans-serif;font-size:40px;font-weight:900;color:var(--primary);margin-bottom:50px;text-align:center;}
hr{border-color:var(--border)!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:var(--primary)!important;}
.inst-card{border:1px solid #e2e8f0;border-radius:16px;overflow:hidden;
  background:white;box-shadow:0 4px 16px rgba(61,82,160,.07);
  transition:all .3s;margin-bottom:20px;}
.inst-card:hover{transform:translateY(-4px);box-shadow:0 12px 32px rgba(61,82,160,.15);}
.journey-container{display:flex;justify-content:space-between;align-items:center;margin:30px 0 40px 0;position:relative;}
.journey-line{position:absolute;top:25px;left:0;width:100%;height:4px;background:#e2e8f0;z-index:0;}
.journey-step{position:relative;z-index:1;text-align:center;width:30%;}
.step-circle{width:50px;height:50px;border-radius:50%;background:white;border:3px solid #cbd5e1;
  display:flex;align-items:center;justify-content:center;font-size:1.5rem;
  margin:0 auto 10px auto;transition:all 0.3s;}
.step-active .step-circle{border-color:var(--primary);background:var(--primary);color:white;
  box-shadow:0 0 0 4px rgba(61,82,160,0.2);transform:scale(1.1);}
.step-done .step-circle{background:#22c55e;border-color:#22c55e;color:white;}
.step-title{font-weight:bold;color:#1e293b;font-size:0.95rem;}
.step-sub{color:#64748b;font-size:0.8rem;}
</style>
<img src="https://img.icons8.com/ios-filled/100/3D52A0/laptop-coding.png"  class="doodle" style="top:22%;left:2%;">
<img src="https://img.icons8.com/ios-filled/100/3D52A0/brainstorming.png" class="doodle" style="top:58%;right:2%;">
<img src="https://img.icons8.com/ios-filled/100/3D52A0/books.png"          class="doodle" style="bottom:18%;left:3%;">
<img src="https://img.icons8.com/ios-filled/100/3D52A0/goal.png"           class="doodle" style="top:14%;right:4%;">
""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════
# CONSTANTS & PATHS
# ═══════════════════════════════════════════════════════════════════
DATASET_PATH   = "career_guidance_dataset.csv"
INST_PATH      = "institutes_dataset.csv"
FEATURE_COLS   = ["work_life_balance","creativity_level","social_interaction",
                  "remote_possibility","burnout_risk","automation_risk","growth_rate"]
MODEL_PATH     = "pf_rf_model.pkl"
SCALER_PATH    = "pf_scaler.pkl"
PF_COLORS      = ["#3D52A0","#7091E6","#ADBBDA","#22c55e","#f59e0b","#ef4444","#8b5cf6","#06b6d4"]

# ═══════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# GROQ CLIENT (Updated for Secrets & Deployment)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def get_groq():
    # Pehle Streamlit Secrets check karega (Deployment ke liye)
    # Agar wahan nahi mili toh environment variables check karega (Local testing ke liye)
    try:
        k = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
    except:
        k = os.getenv("GROQ_API_KEY", "")
        
    if not k: 
        return None
    return Groq(api_key=k)

def groq_complete(messages, system="", max_tokens=2048):
    client = get_groq()
    if not client: 
        return "⚠️ GROQ_API_KEY missing! Please add it to Streamlit Secrets or .env file."
    
    msgs = ([{"role":"system","content":system}] if system else []) + messages
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs, 
            max_tokens=max_tokens
        )
        return r.choices[0].message.content
    except Exception as e: 
        return f"⚠️ API Error: {str(e)}"

# ═══════════════════════════════════════════════════════════════════
# CAREER KNOWLEDGE BASE (built-in fallback — 55 careers)
# ═══════════════════════════════════════════════════════════════════
def _builtin_careers():
    rows = [
        ["Software Engineer",      "Python;JavaScript;Algorithms;Git;System Design",      "BS Computer Science",       115000,28,28,7,7,5,9,5,"analytical","Technology"],
        ["Data Scientist",         "Python;Statistics;ML;SQL;Pandas;Tableau",              "MS Data Science",           122000,36,22,6,8,4,8,6,"analytical","Technology"],
        ["AI/ML Engineer",         "Python;TensorFlow;PyTorch;Math;Deep Learning",         "MS CS/AI",                  138000,42,15,6,9,4,8,7,"analytical","Technology"],
        ["Cybersecurity Analyst",  "Networking;Ethical Hacking;Python;SIEM;Firewalls",     "BS Cybersecurity",          108000,32,18,7,6,5,7,6,"analytical","Technology"],
        ["Cloud Engineer",         "AWS;Azure;DevOps;Linux;Terraform",                     "BS CS / Cloud Certs",       125000,35,20,7,6,5,9,5,"analytical","Technology"],
        ["Mobile Developer",       "Flutter;Swift;Kotlin;React Native;Firebase",           "BS Computer Science",       110000,26,25,7,8,4,8,5,"analytical","Technology"],
        ["DevOps Engineer",        "Docker;Kubernetes;CI/CD;Linux;Scripting",              "BS CS / DevOps Certs",      118000,30,22,6,6,4,8,6,"analytical","Technology"],
        ["Blockchain Developer",   "Solidity;Web3;Cryptography;Smart Contracts",           "BS CS / Blockchain Certs",  130000,38,10,7,9,4,8,6,"analytical","Technology"],
        ["UX Designer",            "Figma;User Research;CSS;Prototyping;Usability",        "BS Design / HCI",            92000,22,20,8,10,7,8,5,"creative","Technology"],
        ["Product Manager",        "Communication;Strategy;Agile;Analytics",               "BS Business / MBA",         132000,24,12,5,7,9,6,7,"social","Technology"],
        ["Game Developer",         "Unity;C++;3D Modeling;Physics;Shaders",                "BS Computer Science",        95000,20,18,6,10,5,7,7,"creative","Technology"],
        ["Data Engineer",          "Spark;Kafka;SQL;Python;ETL Pipelines",                 "BS CS / Data Eng",          118000,33,25,6,6,4,9,5,"analytical","Technology"],
        ["QA Engineer",            "Testing;Selenium;JIRA;Automation;Debugging",           "BS CS / ISTQB Cert",         82000,18,30,7,5,5,8,4,"analytical","Technology"],
        ["IT Project Manager",     "Leadership;Agile;Budgeting;Risk Mgmt;PMP",             "BS IT / PMP Cert",           98000,12,15,6,5,9,7,6,"social","Technology"],
        ["Robotics Engineer",      "ROS;Python;Mechanical Design;Sensors;Control",         "BS Robotics/Mechatronics",  115000,35,10,7,9,5,7,6,"analytical","Technology"],
        ["General Physician",      "Medicine;Diagnosis;Patient Care;Pharmacology",         "MBBS (5 years)",            185000, 8, 5,3,6,10,2,9,"analytical","Healthcare"],
        ["Surgeon",                "Surgery;Anatomy;Precision;Stamina",                    "MBBS + 5yr Residency",      360000, 4, 4,2,7, 8,1,10,"analytical","Healthcare"],
        ["Dentist",                "Oral Surgery;Patient Care;Radiology;Anatomy",          "BDS (5 years)",             155000, 6, 8,5,6, 8,2, 8,"analytical","Healthcare"],
        ["Nurse",                  "Patient Care;Empathy;IVs;Monitoring;First Aid",        "BSN (4 years)",              78000,14,10,4,6,10,3, 8,"social","Healthcare"],
        ["Pharmacist",             "Pharmacology;Chemistry;Patient Counseling",            "Pharm-D (5 years)",         120000, 6,25,6,5, 7,5, 7,"analytical","Healthcare"],
        ["Psychologist",           "Empathy;Counseling;CBT;Research;Assessment",           "MS Psychology",              82000,22, 5,7,6,10,6, 7,"social","Healthcare"],
        ["Physiotherapist",        "Anatomy;Rehab Exercises;Patient Care;Sports Science",  "BS Physiotherapy",           68000,18, 8,6,7, 9,5, 6,"social","Healthcare"],
        ["Nutritionist",           "Dietetics;Biology;Counseling;Meal Planning",           "BS Nutrition",               58000,20,12,7,7, 8,6, 5,"analytical","Healthcare"],
        ["Radiologist",            "Medical Imaging;Anatomy;Diagnosis;Radiology",          "MBBS + Residency",          350000, 5, 8,5,5, 6,3, 8,"analytical","Healthcare"],
        ["Civil Engineer",         "AutoCAD;Structural Analysis;Project Mgmt;Math",        "BS Civil Engineering",       86000,10,14,6,5,6,5,6,"analytical","Engineering"],
        ["Mechanical Engineer",    "CAD;Thermodynamics;Manufacturing;Physics",             "BS Mechanical Engineering",  88000,12,20,6,6,5,4,6,"analytical","Engineering"],
        ["Electrical Engineer",    "Circuit Design;PLC;AutoCAD;Power Systems",             "BS Electrical Engineering",  90000,12,18,6,6,5,4,6,"analytical","Engineering"],
        ["Chemical Engineer",      "Chemistry;Process Design;Thermodynamics;Safety",       "BS Chemical Engineering",    95000,10,18,5,7,5,4,7,"analytical","Engineering"],
        ["Architect",              "AutoCAD;3D Modeling;Design;Urban Planning",            "B.Arch (5 years)",           82000, 9,14,7,10,7,6,6,"creative","Engineering"],
        ["Environmental Engineer", "GIS;Environmental Law;Chemistry;Data Analysis",        "BS Environmental Eng",       78000,16,12,7,7,6,6,6,"analytical","Engineering"],
        ["Financial Analyst",      "Excel;Financial Modeling;Bloomberg;Accounting;CFA",    "BS Finance",                 87000,12,28,5,6,6,5,8,"analytical","Finance"],
        ["Investment Banker",      "Financial Modeling;Valuation;Excel;Networking;CFA",    "BS Finance + MBA",          190000, 6,18,2,6,8,2,10,"analytical","Finance"],
        ["Actuary",                "Statistics;Risk Modeling;Excel;Probability;Coding",    "BS Mathematics",            115000,22,18,7,5,4,7, 5,"analytical","Finance"],
        ["Accountant",             "Excel;Tax Law;Tally;Bookkeeping;Audit",                "BS Accounting / CA",         72000, 6,40,7,3,5,7, 6,"analytical","Finance"],
        ["Marketing Manager",      "SEO;Analytics;Creativity;Branding;CRM",               "BS Marketing",               97000,14,14,6,8,8,6, 7,"creative","Business"],
        ["Entrepreneur",           "Leadership;Risk Tolerance;Finance;Networking;Vision",  "Variable (Any)",            104000,32, 4,5,10,9,7,10,"social","Business"],
        ["HR Manager",             "Communication;Labor Law;Empathy;Recruitment;HRMS",     "BS HRM",                     82000, 9,18,7, 5,10,6, 6,"social","Business"],
        ["Supply Chain Manager",   "Logistics;SAP;Inventory;Negotiation;ERP",              "BS Supply Chain / MBA",      92000,14,22,6, 5, 7,5, 7,"analytical","Business"],
        ["Business Analyst",       "SQL;Excel;Process Mapping;Communication;JIRA",         "BS Business / CS",           96000,18,20,6, 6, 7,6, 6,"analytical","Business"],
        ["School Teacher",         "Communication;Subject Expertise;Patience;Ed Tech",     "B.Ed (4 years)",             47000, 6, 4,7,8,10,6,5,"social","Education"],
        ["University Professor",   "Research;Writing;Teaching;Publishing;Grant Writing",   "PhD (4-6 years)",            92000,12, 4,8,9, 8,7,6,"analytical","Education"],
        ["Educational Consultant", "Curriculum Design;Training;Communication;Ed Policy",   "MS Education",               68000,16, 8,7,7, 7,7,5,"social","Education"],
        ["Graphic Designer",       "Adobe Suite;Typography;Branding;Illustration",         "BS Graphic Design",          57000,12,28,7,10,6,7,4,"creative","Creative Arts"],
        ["Content Creator",        "Video Editing;SEO;Social Media;Storytelling",          "BS Media / Self-taught",     55000,24,15,7,10,7,8,5,"creative","Creative Arts"],
        ["Journalist",             "Writing;Research;Ethics;Interviewing;Multimedia",      "BS Journalism",              58000, 6, 8,6, 8,9,6,7,"creative","Media"],
        ["Musician",               "Music Theory;Instrument;Composition;Performance",      "BM Music / Self-trained",    52000, 6, 4,7,10,8,7,7,"creative","Creative Arts"],
        ["Film Director",          "Storytelling;Cinematography;Leadership;Editing",       "BFA Film",                   78000,10, 8,6,10,8,5,8,"creative","Creative Arts"],
        ["Interior Designer",      "AutoCAD;3D Rendering;Color Theory;Client Mgmt",       "BS Interior Design",          62000,12,18,7,10,7,6,5,"creative","Creative Arts"],
        ["Marine Biologist",       "Biology;Field Research;Statistics;GIS;Diving",         "MS Marine Biology",           62000,10, 8,8, 8,5,5,6,"analytical","Science"],
        ["Environmental Scientist","Chemistry;GIS;Environmental Law;Data Analysis",        "BS Environmental Science",    72000,14,10,7, 7,6,6,6,"analytical","Science"],
        ["Research Scientist",     "Lab Skills;Statistics;Publishing;Critical Thinking",   "PhD in any Science",          90000,18, 8,8, 9,5,7,7,"analytical","Science"],
        ["Lawyer",                 "Constitutional Law;Research;Argumentation;Writing",    "LLB + Bar Exam",             125000, 8, 8,4, 7,8,3,9,"analytical","Law"],
        ["Social Worker",          "Empathy;Case Mgmt;Social Policy;Counseling",           "BS Social Work",              52000,16, 4,6, 6,10,5,8,"social","Social Services"],
        ["Pilot",                  "Aviation;Physics;Navigation;Quick Decisions;Stamina",  "BS Aviation + License",      132000, 6, 8,5, 5,6,3,7,"analytical","Transportation"],
        ["Chef",                   "Culinary Arts;Creativity;Kitchen Mgmt;Food Safety",    "Culinary Degree / Diploma",   67000, 9, 8,4,10,8,3,8,"creative","Hospitality"],
        ["Hotel Manager",          "Hospitality;Leadership;Finance;Customer Service",      "BS Hospitality",              82000,10,12,5, 6,9,4,7,"social","Hospitality"],
    ]
    cols = ["career","required_skills","education_path","avg_salary_usd","growth_rate",
            "automation_risk","work_life_balance","creativity_level","social_interaction",
            "remote_possibility","burnout_risk","cognitive_preference","industry"]
    return pd.DataFrame(rows, columns=cols)

# ═══════════════════════════════════════════════════════════════════
# LOAD CAREER DATA  ← reads from career_guidance_dataset.csv
# The student CSV has a different schema; we use the built-in career
# knowledge base for matching, and the student CSV for ML training.
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_career_data():
    return _builtin_careers()


# ═══════════════════════════════════════════════════════════════════
# LOAD INSTITUTE DATA  (WITH WEB SCRAPING)
# ═══════════════════════════════════════════════════════════════════
import requests
from bs4 import BeautifulSoup

@st.cache_data
def load_institute_data():
    # ── ATTEMPT 1: WEB SCRAPING (Dynamic Data) ─────────────────────
    try:
        # We are scraping Wikipedia's list of universities as a demo
        url = "https://en.wikipedia.org/wiki/List_of_universities_in_Pakistan"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main table (usually class 'wikitable')
            table = soup.find('table', {'class': 'wikitable'})
            
            rows = []
            if table:
                for tr in table.find_all('tr')[1:]:  # Skip header row
                    cols = tr.find_all('td')
                    if len(cols) >= 3:
                        name = cols[0].get_text(strip=True)
                        location = cols[1].get_text(strip=True)
                        # Split location into City/Country (Simplified logic)
                        city = location.split(',')[0] if ',' in location else location
                        country = "Pakistan"
                        
                        # Map generic types based on name (Simple NLP logic)
                        inst_type = "University"
                        if "Institute" in name: inst_type = "Institute"
                        if "College" in name: inst_type = "College"
                        
                        rows.append({
                            "name": name,
                            "city": city,
                            "country": country,
                            "type": inst_type,
                            "career_field": "General Education", # Wiki doesn't always have this specific field
                            "academic_level": "Undergraduate",
                            "fee_min": 2000, "fee_max": 8000, # Generic fees since Wiki doesn't list them
                            "ranking": 7,
                            "scholarship": True,
                            "website": "https://en.wikipedia.org" + (cols[0].find('a')['href'] if cols[0].find('a') else "")
                        })
            
            if rows:
                st.toast("✅ Fetched live data via Web Scraping!", icon="🕷️")
                return pd.DataFrame(rows)

    except Exception as e:
        st.warning(f"⚠️ Scraping failed ({e}). Loading fallback data...")

    # ── ATTEMPT 2: CSV FILE (Fallback / Static) ──────────────────────
    if os.path.exists(INST_PATH):
        try:
            df = pd.read_csv(INST_PATH)
            df.columns = [c.strip() for c in df.columns]
            # Ensure required cols exist
            for col in ["name","city","country","type","career_field",
                        "academic_level","fee_min","fee_max","ranking","scholarship","website"]:
                if col not in df.columns:
                    df[col] = "Unknown" if col in ["name","city","type","career_field","academic_level","website"] else 0
            df["fee_max"] = pd.to_numeric(df["fee_max"], errors="coerce").fillna(0).astype(int)
            df["fee_min"] = pd.to_numeric(df["fee_min"], errors="coerce").fillna(0).astype(int)
            df["ranking"] = pd.to_numeric(df["ranking"], errors="coerce").fillna(5).astype(int)
            return df
        except Exception as e:
            st.warning(f"⚠️ institutes_dataset.csv error: {e}")

    # ── ATTEMPT 3: BUILT-IN FALLBACK (If CSV is missing too) ─────────
    return _builtin_institutes()

# ═══════════════════════════════════════════════════════════════════
# ML MODEL  (uses career knowledge base)
# ═══════════════════════════════════════════════════════════════════
def augment(df, n=600):
    rows = []
    per = max(1, n // len(df))
    for _, row in df.iterrows():
        for _ in range(per):
            nr = row.copy()
            for c in FEATURE_COLS:
                nr[c] = float(np.clip(row[c] + np.random.normal(0, .45), 1, 10))
            rows.append(nr)
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

def do_train(df, n_aug=600):
    aug = augment(df, n_aug)
    
    # FIX: Explicitly convert to numpy arrays to avoid PyArrow errors on Cloud
    # Ensure features are float numpy arrays
    X = aug[FEATURE_COLS].astype(float).to_numpy()
    # Ensure target (career names) is a standard object numpy array (not PyArrow string)
    y = aug["career"].to_numpy(dtype=object)
    
    sc = MinMaxScaler()
    Xs = sc.fit_transform(X)
    
    # train_test_split now receives standard numpy arrays
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    cv = cross_val_score(rf, Xs, y, cv=5)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(sc, SCALER_PATH)
    return {"model":rf,"scaler":sc,"accuracy":acc,"cv_mean":cv.mean(),"cv_std":cv.std(),
            "fi":dict(zip(FEATURE_COLS, rf.feature_importances_)),"n_samples":len(aug)}
    rf.fit(Xtr, ytr)
    acc = accuracy_score(yte, rf.predict(Xte))
    cv = cross_val_score(rf, Xs, y, cv=5)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(sc, SCALER_PATH)
    return {"model":rf,"scaler":sc,"accuracy":acc,"cv_mean":cv.mean(),"cv_std":cv.std(),
            "fi":dict(zip(FEATURE_COLS, rf.feature_importances_)),"n_samples":len(aug)}

@st.cache_resource
def get_model(_hash):
    df = load_career_data()
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return {"model":joblib.load(MODEL_PATH),"scaler":joblib.load(SCALER_PATH),"cached":True,"accuracy":None}
    return do_train(df)

def ml_predict(profile, df, top_n=5):
    r = get_model(str(len(df)))
    m, sc = r["model"], r["scaler"]
    v = np.array([[profile.get("work_life_balance",7), profile.get("creativity",5),
                   profile.get("social",5), profile.get("remote",7), 5,
                   5-profile.get("risk_tolerance",5)/2, 5]])
    probs = m.predict_proba(sc.transform(v))[0]
    idx = np.argsort(probs)[::-1][:top_n]
    return [(m.classes_[i], round(probs[i]*100, 1)) for i in idx]

# ═══════════════════════════════════════════════════════════════════
# MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════════
def compute_matches(profile, df):
    uwlb = profile.get("work_life_balance", 7)
    scores = []
    for _, r in df.iterrows():
        s = 0.0
        s += max(0, 10 - abs(r["work_life_balance"] - uwlb)*2) * 2.0
        s += max(0, 10 - abs(r["creativity_level"] - profile.get("creativity",5))*2) * 1.5
        s += max(0, 10 - abs(r["social_interaction"] - profile.get("social",5))*2) * 1.2
        s += max(0, 10 - abs(r["remote_possibility"] - profile.get("remote",7))*2) * 1.0
        sal_s = min(10, r["avg_salary_usd"]/35000)
        s += max(0, 10 - abs(sal_s - profile.get("income_priority",7))*1.5) * 1.5
        s -= (r["automation_risk"]/10) * (10 - profile.get("risk_tolerance",5)) * 0.5
        s += (r["growth_rate"]/100) * 20
        if uwlb >= 7 and r["burnout_risk"] >= 8: s -= 15
        elif uwlb >= 5 and r["burnout_risk"] >= 9: s -= 8
        if profile.get("cognitive_pref") == r.get("cognitive_preference"): s += 10
        scores.append(round(s, 2))
    df2 = df.copy()
    df2["raw_score"] = scores
    mn, mx = df2["raw_score"].min(), df2["raw_score"].max()
    df2["match_score"] = ((df2["raw_score"] - mn) / max(mx-mn, 1)) * 100
    def bw(row):
        if uwlb >= 7 and row["burnout_risk"] >= 8: return "HIGH"
        if uwlb >= 5 and (row["burnout_risk"] >= 7 or row["work_life_balance"] <= 4): return "MEDIUM"
        return "LOW"
    df2["burnout_warning"] = df2.apply(bw, axis=1)
    return df2.sort_values("match_score", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════
# RESUME HELPERS
# ═══════════════════════════════════════════════════════════════════
def read_resume(f):
    name = f.name.lower()
    if name.endswith(".pdf") and HAS_PDF:
        r = PyPDF2.PdfReader(io.BytesIO(f.read()))
        return "\n".join(p.extract_text() or "" for p in r.pages)
    elif name.endswith(".docx") and HAS_DOCX:
        return docx2txt.process(io.BytesIO(f.read()))
    return f.read().decode("utf-8", errors="ignore")

def analyze_resume(text, age, profile, career=""):
    sys_p = ("You are an expert career counselor. Analyze and provide:\n"
             "1. Key Strengths (3-5)\n2. Skill Gaps for target career\n"
             "3. Age-appropriate feedback\n4. 3 Immediate action items\n"
             "5. Resume Score /100\nBe encouraging but honest.")
    prompt = f"Age:{age}\nTarget Career:{career or 'Not specified'}\nProfile:{json.dumps(profile)}\n\nRESUME:\n{text[:4000]}"
    return groq_complete([{"role":"user","content":prompt}], system=sys_p, max_tokens=1500)

def analyze_persona(profile):
    sys_p = "You are a psychometric career analyst. Create a concise but insightful career persona profile (150 words max)."
    return groq_complete([{"role":"user","content":f"Create career persona:\n{json.dumps(profile,indent=2)}"}],
                         system=sys_p, max_tokens=300)

# ═══════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════
def pf_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)",
        font=dict(color="#1a1a2e", family="Plus Jakarta Sans"),
        xaxis=dict(gridcolor="#e8eaf6", linecolor="#d0d7f0"),
        yaxis=dict(gridcolor="#e8eaf6", linecolor="#d0d7f0"),
        title_font=dict(family="Syne", size=15, color="#3D52A0"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#d0d7f0", borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    base.update(kwargs)
    return base

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def get_career_avatar(career_name, style="notionists"):
    safe_name = career_name.replace(" ", "%20")
    return f"https://api.dicebear.com/7.x/{style}/svg?seed={safe_name}&backgroundColor=transparent"

def highlight_keywords(text):
    keywords = ["creative","analytical","social","leader","strategic","risk","growth",
                "team","logic","detail-oriented","innovative","independent"]
    for k in keywords:
        pattern = re.compile(r'\b(' + re.escape(k) + r')\b', re.IGNORECASE)
        text = pattern.sub(r'<b>\1</b>', text)
    return text

def apply_clean_format(text):
    h1c, h2c, h3c, tc = "#3D52A0","#7091E6","#8697C4","#334155"
    lines = text.split('\n')
    html_out = ""
    in_list = False
    for line in lines:
        s = line.strip()
        if s.startswith("- "):
            if not in_list: html_out += "<ul style='list-style:none;padding-left:0;'>"
            in_list = True
            html_out += (f"<li style='margin-bottom:8px;padding-left:20px;position:relative;"
                         f"color:{tc};line-height:1.6;'>"
                         f"<span style='position:absolute;left:0;top:4px;color:{h2c};'>●</span> {s[2:]}</li>")
        else:
            if in_list: html_out += "</ul>"; in_list = False
            if s.startswith("### "):
                html_out += f"<h3 style='color:{h3c};border-bottom:2px solid {h3c};padding-bottom:5px;margin-top:25px;font-family:Syne,sans-serif;'>{s[4:]}</h3>"
            elif s.startswith("## "):
                html_out += f"<h2 style='color:{h2c};border-bottom:2px solid {h2c};padding-bottom:5px;margin-top:30px;font-family:Syne,sans-serif;'>{s[3:]}</h2>"
            elif s.startswith("# "):
                html_out += f"<h1 style='color:{h1c};border-bottom:3px solid {h1c};padding-bottom:8px;margin-top:35px;font-family:Syne,sans-serif;'>{s[2:]}</h1>"
            elif s:
                html_out += f"<p style='color:{tc};line-height:1.7;margin-bottom:15px;'>{s}</p>"
    if in_list: html_out += "</ul>"
    return html_out

# ═══════════════════════════════════════════════════════════════════
# NAV BAR
# ═══════════════════════════════════════════════════════════════════
def render_nav():
    st.markdown('<div class="pf-nav"><div class="pf-brand">PathFinder<span>.AI</span></div></div>',
                unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:24px 0 16px;'>
          <div style='font-family:Syne,sans-serif;font-size:1.7rem;font-weight:900;color:#EDE8F5;'>🧭</div>
          <div style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;color:#EDE8F5;'>
            PathFinder<span style="color:#ADBBDA;">.AI</span></div>
          <div style='color:#ADBBDA;font-size:.75rem;margin-top:3px;'>Career Intelligence Platform</div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        pages = [
            ("🏠","Dashboard","home"), ("📋","My Profile","profile"),
            ("🎯","Career Matches","matches"), ("🗺️","Skill Roadmap","roadmap"),
            ("🏫","Institute Finder","institute"), ("📄","Resume Analyzer","resume"),
            ("💬","AI Advisor","chat"), ("📊","Market Analysis","insights"),
            ("🤖","Model Training","training"),
        ]
        for icon, label, key in pages:
            c1, c2 = st.columns([1,4])
            with c1: st.write(icon)
            with c2:
                if st.button(label, key=f"sb_{key}", use_container_width=True):
                    st.session_state.app_page = key; st.rerun()
        st.divider()
        p = st.session_state.user_profile
        user = st.session_state.current_user or ""
        info = st.session_state.accounts.get(user, {})
        name = info.get("name","User")
        country = info.get("country","—")
        st.markdown(f"""
        <div style='background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
          border-radius:16px;padding:16px;margin-top:4px;'>
          <div style='color:#ADBBDA;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;'>Logged in as</div>
          <div style='font-weight:800;font-size:1rem;color:#EDE8F5;'>{name}</div>
          <div style='color:#ADBBDA;font-size:.8rem;margin-top:2px;'>🌍 {country}</div>
          {f"<div style='color:#ADBBDA;font-size:.78rem;margin-top:4px;'>Age {p.get('age','?')} · {p.get('academic_level','')}</div>" if p else ""}
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", key="sb_logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.page = "landing"
            st.session_state.app_page = "home"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# PUBLIC NAV BUTTONS
# ═══════════════════════════════════════════════════════════════════
def public_nav_buttons():
    _, c1, c2, c3, c4 = st.columns([5,1,1,1,1])
    with c1:
        if st.button("HOME"):
            st.session_state.page="landing"; st.session_state.modal=None; st.rerun()
    with c2:
        if st.button("ABOUT"):
            st.session_state.page="about"; st.session_state.modal=None; st.rerun()
    with c3:
        if st.button("LOGIN"):
            st.session_state.modal="login"; st.session_state.page="auth"; st.rerun()
    with c4:
        if st.button("JOIN"):
            st.session_state.modal="signup"; st.session_state.page="auth"; st.rerun()

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════
def render_footer():
    st.markdown("""
    <div class="pf-footer">
      <div class="footer-grid">
        <div>
          <h2 style="color:white;font-family:Syne,sans-serif;font-size:26px;font-weight:900;margin-bottom:14px;">PathFinder AI</h2>
          <p style="color:#8697C4;font-size:15px;line-height:1.8;">
            Redefining career guidance through intelligence.<br>
            A dedicated ecosystem for students aimed at global excellence.<br>
            Built with ❤️ by students, for students.
          </p>
        </div>
        <div>
          <h4 style="color:white;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;">Our Team</h4>
          <div style="margin-bottom:18px;">
            <div class="team-name">Tahira Muhammad Javed</div>
            <div class="team-link"><a href="https://www.linkedin.com/in/tahira-muhammad-javed-908494392/" target="_blank">🔗 LinkedIn Profile</a></div>
          </div>
          <div>
            <div class="team-name">Maheen Raza</div>
            <div class="team-link"><a href="https://www.linkedin.com/in/maheen-raza-001b842b9/" target="_blank">🔗 LinkedIn Profile</a></div>
          </div>
        </div>
      </div>
      <div class="footer-bottom">© 2026 PathFinder AI &nbsp;|&nbsp; Career Guidance Platform &nbsp;|&nbsp; All rights reserved.</div>
    </div>""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════
# PAGE: LANDING
# ═══════════════════════════════════════════════════════════════════
def page_landing():
    public_nav_buttons()
    st.markdown("""
    <div class="hero-wrap">
      <div class="hero-img"></div>
      <div class="hero-overlay">
        <h1 class="hero-title">PATHFINDER AI</h1>
        <p style="font-size:20px;max-width:780px;font-weight:300;opacity:.92;margin-top:14px;color:white;">
          A centralized AI ecosystem for high-performance students to discover, plan, and dominate their career paths.
        </p>
        <div style="margin-top:30px;display:flex;gap:14px;justify-content:center;flex-wrap:wrap;">
          <div style="background:rgba(255,255,255,.15);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:8px 22px;color:white;font-size:.88rem;font-weight:600;">🤖 AI-Powered Matching</div>
          <div style="background:rgba(255,255,255,.15);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:8px 22px;color:white;font-size:.88rem;font-weight:600;">🔥 Burnout Prevention</div>
          <div style="background:rgba(255,255,255,.15);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:8px 22px;color:white;font-size:.88rem;font-weight:600;">🗺️ Personalized Roadmaps</div>
          <div style="background:rgba(255,255,255,.15);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:8px 22px;color:white;font-size:.88rem;font-weight:600;">📄 Resume AI Review</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div style='padding:80px 8%;text-align:center;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Neural Ecosystem Systems</div>", unsafe_allow_html=True)
    systems = [
        (" Pathway Mapping","Our AI creates a personalized academic roadmap identifying strengths and weaknesses to guide you toward high-income skills in real-time."),
        (" Project Showcase","A dedicated space to upload and visualize your final-year projects. Get AI feedback on your code quality, documentation, and presentation skills before you graduate."),
        (" Dynamic Mentorship","Gain instant access to a network of global industry leaders — a direct bridge to veterans currently shaping the tech world."),
        (" Skill Benchmarking","Compare your progress with the top 1% of students globally. Our system shows you exactly where you stand."),
        (" Industry Interlink","Automatic profile syncing with global recruitment portals. As you complete projects, your portfolio is showcased to partners worldwide."),
        (" Velocity Loops","A continuous feedback cycle. Every project you finish updates your trajectory, suggesting the next high-impact certification."),
    ]
    c1, c2, c3 = st.columns(3)
    for i,(title,desc) in enumerate(systems):
        with [c1,c2,c3][i%3]:
            st.markdown(f'<div class="glass-card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    render_footer()

# ═══════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# PAGE: ABOUT (MEGA VISUAL UPGRADE)
# ═══════════════════════════════════════════════════════════════════
def page_about():
    public_nav_buttons()
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ALL CSS FOR ABOUT PAGE ────────────────────────────────────
    st.markdown("""
    <style>
    /* ── HERO GRADIENT SECTION ── */
    .about-hero {
      background: linear-gradient(135deg, #0f0c29 0%, #302b63 40%, #24243e 100%);
      border-radius: 32px;
      padding: 70px 50px;
      text-align: center;
      position: relative;
      overflow: hidden;
      margin-bottom: 50px;
    }
    .about-hero::before {
      content: '';
      position: absolute;
      top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: radial-gradient(circle, rgba(112,145,230,0.15) 0%, transparent 60%);
      animation: heroGlow 6s ease-in-out infinite;
    }
    @keyframes heroGlow {
      0%, 100% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(30px, -20px) scale(1.1); }
    }

    /* ── MAIN HEADING — NEON HOVER ── */
    .about-heading {
      font-family: 'Syne', sans-serif;
      font-size: 52px;
      font-weight: 900;
      background: linear-gradient(90deg, #EDE8F5, #ADBBDA, #7091E6, #ADBBDA, #EDE8F5);
      background-size: 300% 100%;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: shimmer 4s linear infinite;
      cursor: default;
      transition: all 0.4s ease;
      position: relative;
      z-index: 2;
    }
    .about-heading:hover {
      filter: brightness(1.3);
      text-shadow: 0 0 20px rgba(112,145,230,0.6), 0 0 40px rgba(61,82,160,0.4);
      transform: scale(1.04);
      letter-spacing: 2px;
    }
    @keyframes shimmer {
      0% { background-position: 0% 50%; }
      100% { background-position: 300% 50%; }
    }

    /* ── SUB HEADING — COLOR WAVE HOVER ── */
    .about-subheading {
      font-family: 'Syne', sans-serif;
      font-size: 22px;
      font-weight: 700;
      color: #ADBBDA;
      margin-top: 16px;
      transition: all 0.4s ease;
      position: relative;
      z-index: 2;
    }
    .about-subheading:hover {
      color: #fff;
      text-shadow: 0 0 12px #7091E6, 0 0 30px #3D52A0;
      transform: translateY(-3px);
    }

    /* ── DESCRIPTION TEXT ── */
    .about-desc {
      font-size: 17px;
      line-height: 2;
      color: #c4cae8;
      max-width: 820px;
      margin: 28px auto 0;
      position: relative;
      z-index: 2;
    }
    .about-desc b {
      color: #7091E6;
      text-shadow: 0 0 8px rgba(112,145,230,0.3);
    }

    /* ── FLOATING PARTICLES ── */
    .particle {
      position: absolute;
      border-radius: 50%;
      pointer-events: none;
      opacity: 0.25;
      animation: float 8s ease-in-out infinite;
    }
    .particle:nth-child(1) { width:80px;height:80px;background:#3D52A0;top:10%;left:8%;animation-delay:0s; }
    .particle:nth-child(2) { width:50px;height:50px;background:#7091E6;top:60%;left:15%;animation-delay:2s; }
    .particle:nth-child(3) { width:120px;height:120px;background:#ADBBDA;top:20%;right:10%;animation-delay:1s; }
    .particle:nth-child(4) { width:40px;height:40px;background:#22c55e;bottom:15%;right:20%;animation-delay:3s; }
    .particle:nth-child(5) { width:60px;height:60px;background:#f59e0b;top:50%;right:5%;animation-delay:4s; }
    @keyframes float {
      0%, 100% { transform: translateY(0) rotate(0deg); }
      33% { transform: translateY(-25px) rotate(120deg); }
      66% { transform: translateY(15px) rotate(240deg); }
    }

    /* ── SECTION TITLES — GLOW UNDERLINE HOVER ── */
    .section-glow {
      font-family: 'Syne', sans-serif;
      font-size: 38px;
      font-weight: 900;
      color: #3D52A0;
      text-align: center;
      margin-bottom: 40px;
      position: relative;
      display: inline-block;
      cursor: default;
      transition: all 0.4s ease;
    }
    .section-glow::after {
      content: '';
      position: absolute;
      bottom: -8px; left: 0;
      width: 0; height: 4px;
      background: linear-gradient(90deg, #3D52A0, #7091E6, #22c55e);
      border-radius: 4px;
      transition: width 0.5s ease;
    }
    .section-glow:hover::after { width: 100%; }
    .section-glow:hover {
      color: #7091E6;
      text-shadow: 0 0 15px rgba(112,145,230,0.4);
      transform: scale(1.05);
    }

    /* ── FLIP CARD SYSTEM ── */
    .flip-container {
      perspective: 1200px;
      width: 100%;
      height: 380px;
      margin-bottom: 24px;
      cursor: pointer;
    }
    .flip-inner {
      position: relative;
      width: 100%;
      height: 100%;
      transition: transform 0.8s cubic-bezier(0.4, 0, 0.2, 1);
      transform-style: preserve-3d;
    }
    .flip-container:hover .flip-inner { transform: rotateY(180deg); }
    .flip-front, .flip-back {
      position: absolute;
      width: 100%;
      height: 100%;
      backface-visibility: hidden;
      border-radius: 24px;
      overflow: hidden;
    }
    .flip-front {
      background: white;
      border: 2px solid rgba(61,82,160,0.1);
      box-shadow: 0 8px 32px rgba(61,82,160,0.1);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 30px;
    }
    .flip-back {
      background: linear-gradient(135deg, #1e2d6b 0%, #3D52A0 50%, #7091E6 100%);
      transform: rotateY(180deg);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 30px;
      color: white;
    }
    .flip-front img {
      width: 120px;
      height: 120px;
      border-radius: 50%;
      object-fit: cover;
      border: 4px solid #3D52A0;
      box-shadow: 0 6px 20px rgba(61,82,160,0.3);
      transition: transform 0.4s;
    }
    .flip-container:hover .flip-front img {
      transform: scale(1.1);
      box-shadow: 0 8px 30px rgba(61,82,160,0.5);
    }
    .flip-name {
      font-family: 'Syne', sans-serif;
      font-size: 1.3rem;
      font-weight: 900;
      color: #3D52A0;
      margin-top: 18px;
      transition: all 0.3s;
    }
    .flip-container:hover .flip-name {
      color: #7091E6;
    }
    .flip-role {
      font-size: 0.85rem;
      color: #8697C4;
      font-weight: 600;
      margin-top: 6px;
    }
    .flip-hint {
      font-size: 0.75rem;
      color: #ADBBDA;
      margin-top: 14px;
      opacity: 0.7;
    }
    .flip-back-title {
      font-family: 'Syne', sans-serif;
      font-size: 1.5rem;
      font-weight: 900;
      margin-bottom: 14px;
    }
    .flip-back-text {
      font-size: 0.88rem;
      line-height: 1.7;
      text-align: center;
      max-width: 280px;
      opacity: 0.92;
    }
    .flip-back-link {
      margin-top: 18px;
      display: inline-block;
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
      padding: 8px 20px;
      border-radius: 999px;
      color: white;
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 700;
      transition: all 0.3s;
    }
    .flip-back-link:hover {
      background: rgba(255,255,255,0.3);
      transform: translateY(-2px);
      box-shadow: 0 4px 15px rgba(255,255,255,0.2);
    }

    /* ── FEATURE CARDS — 3D TILT HOVER ── */
    .feature-card {
      background: white;
      border-radius: 24px;
      padding: 36px 28px;
      border: 2px solid rgba(61,82,160,0.08);
      box-shadow: 0 8px 32px rgba(61,82,160,0.08);
      text-align: center;
      transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      overflow: hidden;
      margin-bottom: 24px;
    }
    .feature-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 4px;
      background: linear-gradient(90deg, #3D52A0, #7091E6, #22c55e, #f59e0b);
      transform: scaleX(0);
      transform-origin: left;
      transition: transform 0.5s ease;
    }
    .feature-card:hover::before { transform: scaleX(1); }
    .feature-card:hover {
      transform: translateY(-14px) rotateX(2deg);
      box-shadow: 0 24px 60px rgba(61,82,160,0.2);
      border-color: #7091E6;
    }
    .feature-icon {
      font-size: 3rem;
      margin-bottom: 16px;
      display: inline-block;
      transition: transform 0.5s;
    }
    .feature-card:hover .feature-icon {
      transform: scale(1.2) rotate(10deg);
    }
    .feature-title {
      font-family: 'Syne', sans-serif;
      font-size: 1.15rem;
      font-weight: 800;
      color: #3D52A0;
      margin-bottom: 10px;
      transition: color 0.3s;
    }
    .feature-card:hover .feature-title { color: #7091E6; }
    .feature-desc {
      font-size: 0.88rem;
      color: #64748b;
      line-height: 1.7;
    }

    /* ── TECH STACK BADGES ── */
    .tech-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: white;
      border: 2px solid rgba(61,82,160,0.1);
      border-radius: 16px;
      padding: 12px 22px;
      margin: 6px;
      font-weight: 700;
      font-size: 0.88rem;
      color: #1a1a2e;
      transition: all 0.4s ease;
      box-shadow: 0 2px 10px rgba(61,82,160,0.06);
    }
    .tech-badge:hover {
      transform: translateY(-6px) scale(1.05);
      box-shadow: 0 12px 30px rgba(61,82,160,0.2);
      border-color: #7091E6;
      color: #3D52A0;
    }
    .tech-badge img {
      width: 28px; height: 28px;
      border-radius: 6px;
    }

    /* ── STATS ROW ── */
    .stat-box {
      background: white;
      border-radius: 20px;
      padding: 28px 20px;
      text-align: center;
      border: 2px solid rgba(61,82,160,0.08);
      box-shadow: 0 4px 20px rgba(61,82,160,0.08);
      transition: all 0.4s ease;
    }
    .stat-box:hover {
      transform: translateY(-8px);
      box-shadow: 0 16px 40px rgba(61,82,160,0.18);
      border-color: #7091E6;
    }
    .stat-number {
      font-family: 'Syne', sans-serif;
      font-size: 2.8rem;
      font-weight: 900;
      background: linear-gradient(135deg, #3D52A0, #7091E6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      transition: all 0.3s;
    }
    .stat-box:hover .stat-number {
      transform: scale(1.1);
    }
    .stat-label {
      font-size: 0.85rem;
      color: #8697C4;
      font-weight: 700;
      margin-top: 4px;
    }

    /* ── TIMELINE ── */
    .timeline-item {
      display: flex;
      align-items: flex-start;
      gap: 20px;
      margin-bottom: 30px;
      position: relative;
    }
    .timeline-dot {
      width: 18px; height: 18px;
      border-radius: 50%;
      background: linear-gradient(135deg, #3D52A0, #7091E6);
      flex-shrink: 0;
      margin-top: 6px;
      box-shadow: 0 0 0 4px rgba(61,82,160,0.15);
      transition: all 0.3s;
    }
    .timeline-item:hover .timeline-dot {
      transform: scale(1.4);
      box-shadow: 0 0 0 6px rgba(112,145,230,0.3), 0 0 20px rgba(61,82,160,0.3);
    }
    .timeline-content {
      background: white;
      border-radius: 16px;
      padding: 20px 24px;
      border: 2px solid rgba(61,82,160,0.08);
      box-shadow: 0 4px 16px rgba(61,82,160,0.06);
      flex: 1;
      transition: all 0.4s;
    }
    .timeline-item:hover .timeline-content {
      transform: translateX(8px);
      box-shadow: 0 8px 30px rgba(61,82,160,0.15);
      border-color: #7091E6;
    }
    .timeline-title {
      font-family: 'Syne', sans-serif;
      font-weight: 800;
      font-size: 1rem;
      color: #3D52A0;
      margin-bottom: 4px;
    }
    .timeline-desc {
      font-size: 0.85rem;
      color: #64748b;
      line-height: 1.6;
    }

    /* ── MISSION QUOTE BOX ── */
    .mission-box {
      background: linear-gradient(135deg, #3D52A0 0%, #7091E6 100%);
      border-radius: 28px;
      padding: 50px;
      text-align: center;
      position: relative;
      overflow: hidden;
      margin: 40px 0;
    }
    .mission-box::before {
      content: '"';
      position: absolute;
      top: -20px; left: 30px;
      font-size: 180px;
      font-family: Georgia, serif;
      color: rgba(255,255,255,0.08);
      line-height: 1;
    }
    .mission-text {
      font-size: 22px;
      line-height: 1.8;
      color: white;
      font-weight: 600;
      max-width: 750px;
      margin: 0 auto;
      position: relative;
      z-index: 2;
    }
    .mission-text b {
      color: #EDE8F5;
      text-shadow: 0 0 10px rgba(237,232,245,0.3);
    }

    /* ── IMAGE HOVER ZOOM + OVERLAY ── */
    .about-img-card {
      border-radius: 24px;
      overflow: hidden;
      position: relative;
      height: 220px;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(61,82,160,0.12);
      cursor: pointer;
    }
    .about-img-card img {
      width: 100%; height: 100%;
      object-fit: cover;
      transition: transform 0.6s ease;
    }
    .about-img-card:hover img { transform: scale(1.15); }
    .about-img-overlay {
      position: absolute;
      inset: 0;
      background: linear-gradient(0deg, rgba(61,82,160,0.85) 0%, transparent 60%);
      display: flex;
      align-items: flex-end;
      padding: 24px;
      opacity: 0;
      transition: opacity 0.4s;
    }
    .about-img-card:hover .about-img-overlay { opacity: 1; }
    .about-img-label {
      color: white;
      font-family: 'Syne', sans-serif;
      font-weight: 800;
      font-size: 1.1rem;
      transform: translateY(10px);
      transition: transform 0.4s;
    }
    .about-img-card:hover .about-img-label { transform: translateY(0); }

    /* ── COLORFUL DIVIDER ── */
    .rainbow-divider {
      height: 4px;
      background: linear-gradient(90deg, #3D52A0, #7091E6, #22c55e, #f59e0b, #ef4444, #8b5cf6, #3D52A0);
      background-size: 300% 100%;
      border-radius: 4px;
      animation: rainbow 4s linear infinite;
      margin: 40px 0;
    }
    @keyframes rainbow {
      0% { background-position: 0% 50%; }
      100% { background-position: 300% 50%; }
    }

    /* ── RESPONSIVE ── */
    @media (max-width: 768px) {
      .about-heading { font-size: 32px; }
      .section-glow { font-size: 28px; }
      .flip-container { height: 340px; }
      .mission-text { font-size: 17px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # HERO SECTION
    # ══════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="about-hero">
      <div class="particle"></div>
      <div class="particle"></div>
      <div class="particle"></div>
      <div class="particle"></div>
      <div class="particle"></div>
      <h1 class="about-heading">The Architecture of PathFinder</h1>
      <p class="about-subheading">✨ Where AI Meets Ambition — A Second Brain for Students ✨</p>
      <p class="about-desc">
        PathFinder AI was born out of a simple necessity: <b>Education is outdated, but your potential is not.</b>
        We built this portal to act as a second brain for students. By combining AI analytics with a
        community-first approach, we ensure that no student is left behind in the era of rapid automation.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # STATS ROW
    # ══════════════════════════════════════════════════════════════
    s1, s2, s3, s4, s5 = st.columns(5)
    stats = [
        ("55+", "Career Paths"), ("9,500+", "Data Points"), ("98%", "Match Accuracy"),
        ("6", "AI Models"), ("24/7", "AI Advisor"),
    ]
    for col, (num, label) in zip([s1,s2,s3,s4,s5], stats):
        with col:
            st.markdown(f"""
            <div class="stat-box">
              <div class="stat-number">{num}</div>
              <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='rainbow-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MISSION QUOTE
    # ══════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="mission-box">
      <p class="mission-text">
        We don't just predict careers — we <b>decode human potential</b>.
        Every student deserves a map, not just a compass.
        <b>PathFinder AI is that map.</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # OUR JOURNEY TIMELINE
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="text-align:center;"><span class="section-glow">🛤️ Our Journey</span></div>',
                unsafe_allow_html=True)

    timeline = [
        ("💡 Ideation Phase", "Identified the gap between outdated career counseling and modern AI capabilities. Researched 9,500+ student profiles to understand real patterns."),
        ("🔧 Development Sprint", "Built the ML pipeline — Random Forest, XGBoost, Logistic Regression — trained on real student data with 98% cross-validated accuracy."),
        ("🤖 AI Integration", "Connected Llama 3.3 70B via Groq API for personalized roadmaps, resume analysis, and real-time career chat with persona-aware responses."),
        ("🚀 Launch & Scale", "Deployed a full-featured Streamlit application with 9 integrated modules: Profile, Matching, Roadmap, Institutes, Resume, Chat, Insights, and Training."),
        ("🔮 What's Next", "Expanding to 200+ careers, adding real-time job market APIs, mentor matching, and collaborative project showcases for students worldwide."),
    ]
    for title, desc in timeline:
        st.markdown(f"""
        <div class="timeline-item">
          <div class="timeline-dot"></div>
          <div class="timeline-content">
            <div class="timeline-title">{title}</div>
            <div class="timeline-desc">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='rainbow-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # FEATURE CARDS (3D tilt hover)
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="text-align:center;"><span class="section-glow">⚡ Core Systems</span></div>',
                unsafe_allow_html=True)

    features = [
        ("🧠", "AI Career Matching", "Multi-algorithm ML engine analyzes 7 lifestyle dimensions to find your perfect career fit with burnout prevention."),
        ("🗺️", "Dynamic Roadmaps", "Age-aware AI-generated learning paths with specific courses, projects, and certifications tailored to your stage."),
        ("📄", "Resume Architect", "ATS-optimized resume analysis with skill gap detection, age-appropriate feedback, and actionable improvement steps."),
        ("💬", "AI Career Advisor", "Persona-aware chatbot powered by Llama 3.3 70B that remembers your profile and gives contextual, data-driven advice."),
        ("🏫", "Institute Finder", "Smart institute discovery with web-scraped live data, scholarship detection, and career-field filtering."),
        ("📊", "Market Intelligence", "Interactive Plotly dashboards showing salary benchmarks, automation risks, growth trends, and industry correlations."),
    ]
    fc1, fc2, fc3 = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with [fc1, fc2, fc3][i % 3]:
            st.markdown(f"""
            <div class="feature-card">
              <div class="feature-icon">{icon}</div>
              <div class="feature-title">{title}</div>
              <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='rainbow-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # IMAGE GALLERY (hover zoom + overlay)
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="text-align:center;"><span class="section-glow">📸 Inside PathFinder</span></div>',
                unsafe_allow_html=True)

    gallery = [
        ("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&h=300&fit=crop", "AI-Powered Analytics Dashboard"),
        ("https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=600&h=300&fit=crop", "Collaborative Learning Environment"),
        ("https://images.unsplash.com/photo-1531403009284-440f080d1e12?w=600&h=300&fit=crop", "Career Path Visualization"),
    ]
    gc1, gc2, gc3 = st.columns(3)
    for col, (url, label) in zip([gc1, gc2, gc3], gallery):
        with col:
            st.markdown(f"""
            <div class="about-img-card">
              <img src="{url}" alt="{label}">
              <div class="about-img-overlay">
                <div class="about-img-label">{label}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='rainbow-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TEAM — FLIP CARDS
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="text-align:center;"><span class="section-glow">👥 Meet the Builders</span></div>',
                unsafe_allow_html=True)

    tc1, tc2 = st.columns(2)

    with tc1:
        st.markdown("""
        <div class="flip-container">
          <div class="flip-inner">
            <div class="flip-front">
              <div class="flip-name">Tahira Muhammad Javed</div>
              <div class="flip-role">🚀 Co-Founder & Lead Developer</div>
              <div class="flip-hint">↻ Hover to flip</div>
            </div>
            <div class="flip-back">
              <div class="flip-back-title">Tahira Muhammad Javed</div>
              <div class="flip-back-text">
                Architect of PathFinder's ML pipeline and full-stack implementation.
                Specializes in AI/ML integration, career analytics, and building intelligent systems
                that actually help students. Passionate about making education accessible.
              </div>
              <a href="https://www.linkedin.com/in/tahira-muhammad-javed-908494392/" target="_blank"
                 class="flip-back-link">🔗 LinkedIn Profile</a>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with tc2:
        st.markdown("""
        <div class="flip-container">
          <div class="flip-inner">
            <div class="flip-front">
              <div class="flip-name">Maheen Raza</div>
              <div class="flip-role">🎨 Co-Founder & UX Architect</div>
              <div class="flip-hint">↻ Hover to flip</div>
            </div>
            <div class="flip-back">
              <div class="flip-back-title">Maheen Raza</div>
              <div class="flip-back-text">
                Designer of PathFinder's user experience and visual identity.
                Expert in user research, interface design, and creating intuitive journeys
                that make complex AI tools feel simple. Believes great design is invisible.
              </div>
              <a href="https://www.linkedin.com/in/maheen-raza-001b842b9/" target="_blank"
                 class="flip-back-link">🔗 LinkedIn Profile</a>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='rainbow-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TECH STACK BADGES
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div style="text-align:center;"><span class="section-glow">🛠️ Tech Stack</span></div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 10px 0 30px;">
      <span class="tech-badge">🐍 Python</span>
      <span class="tech-badge">🌊 Streamlit</span>
      <span class="tech-badge">🤖 Groq (Llama 3.3)</span>
      <span class="tech-badge">📊 Scikit-Learn</span>
      <span class="tech-badge">📈 Plotly</span>
      <span class="tech-badge">🌲 Random Forest</span>
      <span class="tech-badge">⚡ XGBoost</span>
      <span class="tech-badge">🐼 Pandas</span>
      <span class="tech-badge">🔮 NumPy</span>
      <span class="tech-badge">📄 PyPDF2</span>
      <span class="tech-badge">🕷️ BeautifulSoup</span>
      <span class="tech-badge">💾 Joblib</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='rainbow-divider'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # CLOSING CTA
    # ══════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      border-radius: 28px;
      padding: 60px 40px;
      text-align: center;
      position: relative;
      overflow: hidden;
    ">
      <div style="
        position: absolute;
        top: -80px; right: -80px;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(112,145,230,0.2), transparent 70%);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
      "></div>
      <h2 style="
        font-family: 'Syne', sans-serif;
        font-size: 36px;
        font-weight: 900;
        color: white;
        margin-bottom: 16px;
        position: relative;
        z-index: 2;
      ">Ready to Find Your Path?</h2>
      <p style="
        color: #ADBBDA;
        font-size: 17px;
        line-height: 1.8;
        max-width: 600px;
        margin: 0 auto;
        position: relative;
        z-index: 2;
      ">Join thousands of students who are using AI to take control of their future.
      Your career journey starts with a single click.</p>
    </div>
    """, unsafe_allow_html=True)

    render_footer()
# ═══════════════════════════════════════════════════════════════════
# PAGE: AUTH
# ═══════════════════════════════════════════════════════════════════
def page_auth():
    public_nav_buttons()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.logged_in:
        user = st.session_state.current_user or ""
        info = st.session_state.accounts.get(user, {})
        name = info.get("name","Student"); country = info.get("country","—")
        st.markdown(f"""
        <div class="dash-wrap">
          <div style="font-family:Syne,sans-serif;font-size:28px;font-weight:900;color:var(--primary);margin-bottom:4px;">👋 Welcome back, {name}!</div>
          <div style="color:var(--muted);font-size:14px;margin-bottom:30px;">Your PathFinder AI profile is active and ready.</div>
          <div class="dash-grid">
            <div class="dash-field"><div class="label">Full Name</div><div class="value">{name}</div></div>
            <div class="dash-field"><div class="label">Email Address</div><div class="value">{user}</div></div>
            <div class="dash-field"><div class="label">Country</div><div class="value">{country}</div></div>
            <div class="dash-field"><div class="label">Account Status</div><div class="value" style="color:var(--success);">✅ Active</div></div>
          </div>
          <p style="color:var(--muted);font-size:13px;margin:0;">Use the <b>sidebar</b> to access Career Matching, Roadmaps, Resume Analysis, and more.</p>
        </div>""", unsafe_allow_html=True)
    elif st.session_state.modal == "login":
        _, col_form, __ = st.columns([1,2,1])
        with col_form:
            st.markdown("""
            <div class="auth-wrap">
              <div style="font-family:Syne,sans-serif;font-size:26px;font-weight:900;color:var(--primary);text-align:center;margin-bottom:4px;">🔐 Welcome Back</div>
              <div style="color:var(--muted);text-align:center;font-size:13px;margin-bottom:24px;">Login to your PathFinder account</div>
            </div>""", unsafe_allow_html=True)
            email = st.text_input("Email Address", placeholder="you@example.com", key="li_email")
            pwd = st.text_input("Password", type="password", placeholder="••••••••", key="li_pwd")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Login", key="do_login"):
                    accs = st.session_state.accounts
                    if email in accs and accs[email]["password"] == pwd:
                        st.session_state.logged_in = True; st.session_state.current_user = email
                        st.session_state.page = "auth"; st.success("Logged in! 🎉"); st.rerun()
                    elif email not in accs: st.error("❌ No account found. Please JOIN first!")
                    else: st.error("❌ Incorrect password.")
            with c2:
                if st.button("✖ Close", key="cl_login"):
                    st.session_state.modal = None; st.session_state.page = "landing"; st.rerun()
            st.markdown("<div style='text-align:center;color:#888;font-size:13px;margin-top:14px;'>Don't have an account? <b>Click JOIN above</b></div>", unsafe_allow_html=True)
    elif st.session_state.modal == "signup":
        _, col_form, __ = st.columns([1,2,1])
        with col_form:
            st.markdown("""
            <div class="auth-wrap">
              <div style="font-family:Syne,sans-serif;font-size:26px;font-weight:900;color:var(--primary);text-align:center;margin-bottom:4px;">🚀 Join PathFinder</div>
              <div style="color:var(--muted);text-align:center;font-size:13px;margin-bottom:24px;">Create your free account today</div>
            </div>""", unsafe_allow_html=True)
            rname = st.text_input("Full Name", placeholder="Your full name", key="su_name")
            remail = st.text_input("Email Address", placeholder="you@example.com", key="su_email")
            rpwd = st.text_input("Password", type="password", placeholder="••••••••", key="su_pwd")
            countries = ["Select your country","Pakistan","India","United States","United Kingdom",
                         "Canada","Australia","UAE","Saudi Arabia","Germany","Other"]
            rcountry = st.selectbox("Country", countries, key="su_country")
            terms = st.checkbox("I agree to **Terms & Conditions** and **Privacy Policy**", key="su_terms")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🚀 Create Account", key="do_signup"):
                    if not terms: st.error("Accept Terms & Conditions!")
                    elif rcountry == "Select your country": st.error("Select your country!")
                    elif not rname or not remail or not rpwd: st.error("Fill in all fields!")
                    elif remail in st.session_state.accounts: st.error("Email already registered. Please LOGIN!")
                    else:
                        st.session_state.accounts[remail] = {"name":rname,"password":rpwd,"country":rcountry}
                        st.session_state.logged_in = True; st.session_state.current_user = remail
                        st.success("Account created! Welcome 🎉"); st.rerun()
            with c2:
                if st.button("✖ Close", key="cl_signup"):
                    st.session_state.modal = None; st.session_state.page = "landing"; st.rerun()
            st.markdown("<div style='text-align:center;color:#888;font-size:13px;margin-top:14px;'>Already have an account? <b>Click LOGIN above</b></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# APP: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════════
def app_home():
    user = st.session_state.current_user or ""
    info = st.session_state.accounts.get(user, {})
    name = info.get("name", st.session_state.user_profile.get("name","Student"))
    st.markdown(f"<h1>👋 Welcome back, {name}!</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);margin-top:-8px;margin-bottom:24px;'>Your Personalized Career Intelligence Dashboard.</p>", unsafe_allow_html=True)
 
    has_profile = bool(st.session_state.user_profile)
    has_matches = st.session_state.career_matches is not None
 
    if not has_profile:
        st.warning("🚀 Please complete your profile first.")
        if st.button("📋 Go to Profile", use_container_width=True):
            st.session_state.app_page = "profile"; st.rerun()
        return
 
    profile = st.session_state.user_profile
    age = profile.get("age", 18)
    top_3_careers = []
    if has_matches:
        matches = st.session_state.career_matches
        if isinstance(matches, pd.DataFrame) and not matches.empty:
            top_3_careers = matches.head(3).to_dict('records')
    summary_text = st.session_state.get("persona_summary") or "Complete your profile to see your career persona."
 
    # ── METRICS ROW (age-aware) ────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    if age <= 13:
        top_match = top_3_careers[0].get('career','Explorer') if top_3_careers else "Explorer"
        metrics = [
            ("🌟","Top Talent",top_match,"var(--primary)","1.1rem"),
            ("🎨","Creativity",f"{profile.get('creativity',5)}/10","#f59e0b","1.5rem"),
            ("🚀","Future Tech","High" if any(x in top_match.lower() for x in ['software','engineer','data','ai','robot']) else "Moderate","#3D52A0","1.5rem"),
            ("⚡","Superpower",profile.get("energy","Ambivert").split()[0],"var(--secondary)","1.2rem"),
        ]
    elif age <= 17:
        top_cn = top_3_careers[0].get('career','Dream Job') if top_3_careers else "Dream Job"
        sc = int(top_3_careers[0].get('match_score',0)) if top_3_careers else 0
        gr = int(top_3_careers[0].get('growth_rate',0)) if top_3_careers else 0
        metrics = [
            ("🎯","Dream Job",top_cn,"var(--primary)","1.1rem"),
            ("🎓","College Fit",f"{sc}%","#22c55e" if sc>80 else "#f59e0b","1.5rem"),
            ("📈","Future Growth",f"+{gr}%","#3D52A0","1.5rem"),
            ("🏫","Current Stage",profile.get("academic_level","High School"),"var(--secondary)","1.2rem"),
        ]
    else:
        top_cn = top_3_careers[0].get('career','Analyze Profile') if top_3_careers else "Analyze Profile"
        sc = int(top_3_careers[0].get('match_score',0)) if top_3_careers else 0
        gr = int(top_3_careers[0].get('growth_rate',0)) if top_3_careers else 0
        sc_col = "#22c55e" if sc>80 else "#f59e0b" if sc>50 else "#ef4444"
        metrics = [
            ("🏆","Top Match",top_cn,"var(--primary)","1.1rem"),
            ("💯","Compatibility",f"{sc}%",sc_col,"1.5rem"),
            ("📈","Demand",f"+{gr}%","#3D52A0","1.5rem"),
            ("🧘","Life Fit","High" if profile.get("work_life_balance",5)>7 else "Moderate","#8b5cf6","1.2rem"),
        ]
    for col,(icon,label,val,color,fsize) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""
            <div class="pf-card" style="text-align:center;">
              <div style="font-size:2rem;margin-bottom:6px;">{icon}</div>
              <div style="color:var(--muted);font-size:.8rem;font-weight:600;">{label}</div>
              <div style="font-family:Syne,sans-serif;font-size:{fsize};font-weight:900;color:{color};line-height:1.2;">{val}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2,1])
 
    with col_left:
        header_map = {True: "🌟 Talents to Explore" if age<=13 else "🎯 Future Career Paths" if age<=17 else "🏆 Top Career Matches"}
        st.markdown(f"<h3>{'🌟 Talents to Explore' if age<=13 else '🎯 Future Career Paths' if age<=17 else '🏆 Top Career Matches'}</h3>", unsafe_allow_html=True)
        if top_3_careers:
            for idx, career in enumerate(top_3_careers):
                bc = "#fbbf24" if idx==0 else "#9ca3af" if idx==1 else "#b45309"
                rank_badge = "🥇" if idx==0 else "🥈" if idx==1 else "🥉"
                if age<=17:
                    salary_text = career.get('industry','General')
                else:
                    salary_text = f"{career.get('industry','General')} | 💰 ${career.get('avg_salary_usd',0):,}/yr"
                if age<=13:
                    b1,b1c = f"🎨 Fun: {profile.get('creativity',5)}/10","#f59e0b"
                    b2,b2c = f"🧠 Social: {profile.get('social',5)}/10","#3D52A0"
                else:
                    risk = career.get('burnout_risk',5)
                    rc = "#22c55e" if risk<6 else "#f59e0b" if risk<8 else "#ef4444"
                    b1,b1c = f"🔥 {'Low' if risk<6 else 'Med' if risk<8 else 'High'} Burnout",rc
                    b2,b2c = f"📈 Growth: +{int(career.get('growth_rate',0))}%","#059669"
                st.markdown(f"""
                <div class="pf-card" style="border-left:5px solid {bc};margin-bottom:15px;padding:15px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                      <div style="font-size:1.1rem;font-weight:700;color:var(--text);">{rank_badge} {career['career']}</div>
                      <div style="font-size:0.8rem;color:var(--muted);margin-top:4px;">{salary_text}</div>
                    </div>
                    <div style="text-align:right;">
                      <div style="font-size:1.2rem;font-weight:800;color:var(--primary);">{int(career.get('match_score',0))}%</div>
                      <div style="font-size:0.7rem;color:var(--muted);">Match</div>
                    </div>
                  </div>
                  <div style="margin-top:10px;display:flex;gap:10px;font-size:0.8rem;">
                    <span style="background:#f1f5f9;padding:2px 8px;border-radius:4px;color:{b1c};font-weight:600;">{b1}</span>
                    <span style="background:#f1f5f9;padding:2px 8px;border-radius:4px;color:{b2c};font-weight:600;">{b2}</span>
                  </div>
                </div>""", unsafe_allow_html=True)
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button(f"🗺️ Roadmap", key=f"home_rm_{idx}", use_container_width=True):
                        st.session_state.selected_career = career['career']
                        st.session_state.app_page = "roadmap"; st.rerun()
                with btn2:
                    if st.button(f"🏫 Institutes", key=f"home_inst_{idx}", use_container_width=True):
                        st.session_state.selected_career = career['career']
                        st.session_state.app_page = "institute"; st.rerun()
        else:
            st.info("Complete your profile to see career matches.")
 
    with col_right:
        header = "🧒 Kid's Personality" if age<=13 else "🧑‍🎓 Student Persona" if age<=17 else "🧠 Lifestyle Alignment"
        text = "You are full of energy! Focus on having fun while learning new skills." if age<=13 else summary_text
        st.markdown(f"""
        <div class="pf-card" style="background:linear-gradient(135deg,#f8fafc 0%,#e2e8f0 100%);">
          <div class="pf-card-title" style="font-size:1.2rem;margin-bottom:12px;">{header}</div>
          <p style="font-size:0.95rem;color:#334155;line-height:1.7;margin:0;">{highlight_keywords(text)}</p>
        </div>""", unsafe_allow_html=True)
 
# ═══════════════════════════════════════════════════════════════════
# APP: PROFILE
# ═══════════════════════════════════════════════════════════════════
def app_profile():
    st.markdown("<h2>📋 Build Your Profile</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);'>The more detail you provide, the better your career matches will be.</p>", unsafe_allow_html=True)
    if "user_profile" not in st.session_state: st.session_state.user_profile = {}
    p = st.session_state.user_profile
    acad_opts = ["Primary School","Middle School","High School","Undergraduate","Graduate","PhD","Working Professional"]
    cog_opts  = ["analytical","creative","social","mechanical","mixed"]
    defaults  = {
        "p_name": p.get("name",""), "p_age": int(p.get("age",18)),
        "p_location": p.get("location",""),
        "p_academic": p.get("academic_level","Undergraduate") if p.get("academic_level") in acad_opts else "Undergraduate",
        "p_financial": p.get("financial_range","$30K–$60K"),
        "p_hobbies": p.get("hobbies",""),
        "p_energy": p.get("energy","Ambivert"),
        "p_fav_sub": p.get("fav_subjects",[]),
        "p_work_pref": p.get("work_pref",[]),
        "p_risk": int(p.get("risk_tolerance",5)),
        "p_creativity": int(p.get("creativity",5)),
        "p_social": int(p.get("social",5)),
        "p_cog": p.get("cognitive_pref","analytical") if p.get("cognitive_pref") in cog_opts else "analytical",
        "p_wlb": int(p.get("work_life_balance",7)),
        "p_income": int(p.get("income_priority",7)),
        "p_travel": int(p.get("travel",5)),
        "p_family": int(p.get("family_time",7)),
        "p_impact": int(p.get("social_impact",5)),
        "p_remote": int(p.get("remote",7)),
        "p_vision": p.get("vision_25",""),
        "p_dream":  p.get("dream_life",""),
        "p_skills": p.get("current_skills",""),
    }
    for k,v in defaults.items():
        st.session_state.setdefault(k,v)
 
    t1,t2,t3,t4 = st.tabs(["👤 Basic Info","🧠 Personality","🌟 Lifestyle Goals","🔭 Long-Term Vision"])
    with t1:
        c1,c2 = st.columns(2)
        with c1:
            st.text_input("Full Name *", key="p_name")
            st.number_input("Age", 10, 65, key="p_age")
            st.text_input("City, Country *", key="p_location")
        with c2:
            st.selectbox("Academic Level", acad_opts, key="p_academic")
            st.selectbox("Financial Range", ["Below $10K","$10K–$30K","$30K–$60K","$60K–$100K","$100K+"], index=2, key="p_financial")
            st.text_input("Hobbies (comma separated)", key="p_hobbies")
    with t2:
        c1,c2 = st.columns(2)
        with c1:
            st.select_slider("Energy Style", ["Strong Introvert","Introvert","Ambivert","Extrovert","Strong Extrovert"], key="p_energy")
            st.multiselect("Favourite Subjects",
                ["Mathematics","Physics","Chemistry","Biology","Computer Science","History",
                 "Literature","Art","Music","Economics","Psychology","Law","Business","Languages"],
                key="p_fav_sub")
            st.multiselect("Work Preference",
                ["Remote","Office","Fieldwork","Creative Studio","Technical Lab","Outdoors","Hospital","Classroom"],
                key="p_work_pref")
        with c2:
            st.slider("Risk Tolerance",1,10,key="p_risk",help="1=very safe, 10=love risk")
            st.slider("Creativity Drive",1,10,key="p_creativity")
            st.slider("Social Interaction Preference",1,10,key="p_social")
            st.selectbox("Thinking Style",cog_opts,key="p_cog")
    with t3:
        c1,c2 = st.columns(2)
        with c1:
            st.slider("Work-Life Balance Importance",1,10,key="p_wlb")
            st.slider("Income Priority",1,10,key="p_income")
            st.slider("Travel Desire",1,10,key="p_travel")
        with c2:
            st.slider("Family Time Priority",1,10,key="p_family")
            st.slider("Social Impact Desire",1,10,key="p_impact")
            st.slider("Remote Work Preference",1,10,key="p_remote")
    with t4:
        st.text_area("Where do you see yourself in 5 years? *", height=100, key="p_vision")
        st.text_area("What kind of life do you want? *", height=100, key="p_dream")
        st.text_area("Current Skills / Experience *", height=80, key="p_skills")
 
    st.markdown("---")
    if st.button("💾 Save Profile & Find Matches →", use_container_width=True):
        missing = []
        if not st.session_state.p_name.strip():    missing.append("👤 **Full Name**")
        if not st.session_state.p_location.strip():missing.append("👤 **Location**")
        if not st.session_state.p_vision.strip():  missing.append("🔭 **5-Year Vision**")
        if not st.session_state.p_dream.strip():   missing.append("🔭 **Dream Life**")
        if not st.session_state.p_skills.strip():  missing.append("🔭 **Current Skills**")
        if missing:
            st.error("❌ Please complete all required fields:")
            for f in missing: st.markdown(f"- {f}")
            st.stop()
        profile = dict(
            name=st.session_state.p_name, age=st.session_state.p_age,
            location=st.session_state.p_location, academic_level=st.session_state.p_academic,
            financial_range=st.session_state.p_financial, hobbies=st.session_state.p_hobbies,
            energy=st.session_state.p_energy, fav_subjects=st.session_state.p_fav_sub,
            work_pref=st.session_state.p_work_pref, risk_tolerance=st.session_state.p_risk,
            creativity=st.session_state.p_creativity, social=st.session_state.p_social,
            cognitive_pref=st.session_state.p_cog, work_life_balance=st.session_state.p_wlb,
            income_priority=st.session_state.p_income, travel=st.session_state.p_travel,
            family_time=st.session_state.p_family, social_impact=st.session_state.p_impact,
            remote=st.session_state.p_remote, vision_25=st.session_state.p_vision,
            dream_life=st.session_state.p_dream, current_skills=st.session_state.p_skills,
        )
        st.session_state.user_profile = profile
        df = load_career_data()
        with st.spinner("🤖 Analyzing your profile..."):
            st.session_state.career_matches = compute_matches(profile, df)
            st.session_state.persona_summary = analyze_persona(profile)
        st.success("✅ Profile saved! Redirecting to your matches...")
        time.sleep(1)
        st.session_state.app_page = "matches"; st.rerun()
 
# ═══════════════════════════════════════════════════════════════════
# APP: PROFILE
# ═══════════════════════════════════════════════════════════════════
def app_profile():
    st.markdown("<h2>📋 Build Your Profile</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);'>The more detail you provide, the better your career matches will be.</p>", unsafe_allow_html=True)
    if "user_profile" not in st.session_state: st.session_state.user_profile = {}
    p = st.session_state.user_profile
    acad_opts = ["Primary School","Middle School","High School","Undergraduate","Graduate","PhD","Working Professional"]
    cog_opts  = ["analytical","creative","social","mechanical","mixed"]
    defaults  = {
        "p_name": p.get("name",""), "p_age": int(p.get("age",18)),
        "p_location": p.get("location",""),
        "p_academic": p.get("academic_level","Undergraduate") if p.get("academic_level") in acad_opts else "Undergraduate",
        "p_financial": p.get("financial_range","$30K–$60K"),
        "p_hobbies": p.get("hobbies",""),
        "p_energy": p.get("energy","Ambivert"),
        "p_fav_sub": p.get("fav_subjects",[]),
        "p_work_pref": p.get("work_pref",[]),
        "p_risk": int(p.get("risk_tolerance",5)),
        "p_creativity": int(p.get("creativity",5)),
        "p_social": int(p.get("social",5)),
        "p_cog": p.get("cognitive_pref","analytical") if p.get("cognitive_pref") in cog_opts else "analytical",
        "p_wlb": int(p.get("work_life_balance",7)),
        "p_income": int(p.get("income_priority",7)),
        "p_travel": int(p.get("travel",5)),
        "p_family": int(p.get("family_time",7)),
        "p_impact": int(p.get("social_impact",5)),
        "p_remote": int(p.get("remote",7)),
        "p_vision": p.get("vision_25",""),
        "p_dream":  p.get("dream_life",""),
        "p_skills": p.get("current_skills",""),
    }
    for k,v in defaults.items():
        st.session_state.setdefault(k,v)

    t1,t2,t3,t4 = st.tabs(["👤 Basic Info","🧠 Personality","🌟 Lifestyle Goals","🔭 Long-Term Vision"])
    with t1:
        c1,c2 = st.columns(2)
        with c1:
            st.text_input("Full Name *", key="p_name")
            st.number_input("Age", 10, 65, key="p_age")
            st.text_input("City, Country *", key="p_location")
        with c2:
            st.selectbox("Academic Level", acad_opts, key="p_academic")
            st.selectbox("Financial Range", ["Below $10K","$10K–$30K","$30K–$60K","$60K–$100K","$100K+"], index=2, key="p_financial")
            st.text_input("Hobbies (comma separated)", key="p_hobbies")
    with t2:
        c1,c2 = st.columns(2)
        with c1:
            st.select_slider("Energy Style", ["Strong Introvert","Introvert","Ambivert","Extrovert","Strong Extrovert"], key="p_energy")
            st.multiselect("Favourite Subjects",
                ["Mathematics","Physics","Chemistry","Biology","Computer Science","History",
                 "Literature","Art","Music","Economics","Psychology","Law","Business","Languages"],
                key="p_fav_sub")
            st.multiselect("Work Preference",
                ["Remote","Office","Fieldwork","Creative Studio","Technical Lab","Outdoors","Hospital","Classroom"],
                key="p_work_pref")
        with c2:
            st.slider("Risk Tolerance",1,10,key="p_risk",help="1=very safe, 10=love risk")
            st.slider("Creativity Drive",1,10,key="p_creativity")
            st.slider("Social Interaction Preference",1,10,key="p_social")
            st.selectbox("Thinking Style",cog_opts,key="p_cog")
    with t3:
        c1,c2 = st.columns(2)
        with c1:
            st.slider("Work-Life Balance Importance",1,10,key="p_wlb")
            st.slider("Income Priority",1,10,key="p_income")
            st.slider("Travel Desire",1,10,key="p_travel")
        with c2:
            st.slider("Family Time Priority",1,10,key="p_family")
            st.slider("Social Impact Desire",1,10,key="p_impact")
            st.slider("Remote Work Preference",1,10,key="p_remote")
    with t4:
        st.text_area("Where do you see yourself in 5 years? *", height=100, key="p_vision")
        st.text_area("What kind of life do you want? *", height=100, key="p_dream")
        st.text_area("Current Skills / Experience *", height=80, key="p_skills")

    st.markdown("---")
    if st.button("💾 Save Profile & Find Matches →", use_container_width=True):
        missing = []
        if not st.session_state.p_name.strip():    missing.append("👤 **Full Name**")
        if not st.session_state.p_location.strip():missing.append("👤 **Location**")
        if not st.session_state.p_vision.strip():  missing.append("🔭 **5-Year Vision**")
        if not st.session_state.p_dream.strip():   missing.append("🔭 **Dream Life**")
        if not st.session_state.p_skills.strip():  missing.append("🔭 **Current Skills**")
        if missing:
            st.error("❌ Please complete all required fields:")
            for f in missing: st.markdown(f"- {f}")
            st.stop()
        profile = dict(
            name=st.session_state.p_name, age=st.session_state.p_age,
            location=st.session_state.p_location, academic_level=st.session_state.p_academic,
            financial_range=st.session_state.p_financial, hobbies=st.session_state.p_hobbies,
            energy=st.session_state.p_energy, fav_subjects=st.session_state.p_fav_sub,
            work_pref=st.session_state.p_work_pref, risk_tolerance=st.session_state.p_risk,
            creativity=st.session_state.p_creativity, social=st.session_state.p_social,
            cognitive_pref=st.session_state.p_cog, work_life_balance=st.session_state.p_wlb,
            income_priority=st.session_state.p_income, travel=st.session_state.p_travel,
            family_time=st.session_state.p_family, social_impact=st.session_state.p_impact,
            remote=st.session_state.p_remote, vision_25=st.session_state.p_vision,
            dream_life=st.session_state.p_dream, current_skills=st.session_state.p_skills,
        )
        st.session_state.user_profile = profile
        df = load_career_data()
        with st.spinner("🤖 Analyzing your profile..."):
            st.session_state.career_matches = compute_matches(profile, df)
            st.session_state.persona_summary = analyze_persona(profile)
        st.success("✅ Profile saved! Redirecting to your matches...")
        time.sleep(1)
        st.session_state.app_page = "matches"; st.rerun()

# ═══════════════════════════════════════════════════════════════════
# APP: MATCHES
# ═══════════════════════════════════════════════════════════════════
def app_matches():
    st.markdown("<h2> Your Career Matches</h2>", unsafe_allow_html=True)
    if st.session_state.career_matches is None:
        st.info("📋 Complete your profile first.")
        if st.button("Go to Profile →"): st.session_state.app_page="profile"; st.rerun()
        return

    df = st.session_state.career_matches
    profile = st.session_state.user_profile
    if st.session_state.persona_summary:
        st.markdown(f"""<div class="pf-card">
          <div class="pf-card-title">Your Persona Analysis</div>
          <p style="color:var(--muted);line-height:1.75;margin:0;">{st.session_state.persona_summary}</p>
        </div>""", unsafe_allow_html=True)

    df_raw = load_career_data()
    with st.spinner("Running ML predictions..."):
        ml = ml_predict(profile, df_raw)

    col_l, col_r = st.columns([3,2])
    colors_ = ["#3D52A0","#7091E6","#ADBBDA","#22c55e","#f59e0b"]

    with col_l:
        st.markdown("<h3>Top Career Matches</h3>", unsafe_allow_html=True)
        for i,(_, row) in enumerate(df.head(5).iterrows()):
            color = colors_[i]
            bc = {"LOW":"badge-green","MEDIUM":"badge-yellow","HIGH":"badge-red"}[row["burnout_warning"]]
            bl = {"LOW":"✅ Low Burnout","MEDIUM":"⚠️ Med Burnout","HIGH":"🔴 High Burnout"}[row["burnout_warning"]]
            skills = [s.strip() for s in str(row["required_skills"]).split(";")[:4]]
            sbadges = " ".join(f"<span class='badge badge-blue'>{s}</span>" for s in skills)
            st.markdown(f"""
            <div class="match-card {'top' if i==0 else ''}" style="border-left:4px solid {color};">
              <div class="match-rank">#{i+1}</div>
              <div class="match-title">{row['career']}</div>
              <div style="margin-bottom:10px;">
                <span class='badge badge-teal'>{row.get('industry','')}</span>
                <span class='badge {bc}'>{bl}</span>
                <span class='badge badge-yellow'>💰 ${row['avg_salary_usd']:,}/yr</span>
                <span class='badge badge-purple'>📈 {row['growth_rate']}% growth</span>
              </div>
              <div class="score-bar-bg"><div class="score-bar-fill" style="width:{row['match_score']:.0f}%;background:linear-gradient(90deg,{color},{color}88);"></div></div>
              <div style="color:{color};font-family:Syne,sans-serif;font-weight:800;font-size:1.15rem;margin-bottom:10px;">{row['match_score']:.1f}% Match</div>
              <div style="margin-bottom:8px;">{sbadges}</div>
              <div style="color:var(--muted);font-size:.8rem;">🎓 {row['education_path']}</div>
            </div>""", unsafe_allow_html=True)
            ca, cb, cc = st.columns(3)
            with ca:
                if st.button("🗺️ Roadmap", key=f"rm_{i}", use_container_width=True):
                    st.session_state.selected_career=row["career"]; st.session_state.app_page="roadmap"; st.rerun()
            with cb:
                if st.button("💬 Ask AI", key=f"ai_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":f"Tell me about a career as {row['career']}. Based on my profile, is it a good fit?"})
                    st.session_state.app_page="chat"; st.rerun()
            with cc:
                if st.button("🏫 Institutes", key=f"inst_{i}", use_container_width=True):
                    st.session_state.selected_career=row["career"]; st.session_state.app_page="institute"; st.rerun()

    with col_r:
        if len(df) >= 3:
            cats = ["Work-Life","Creativity","Social","Remote","Growth","Salary"]
            fig = go.Figure()
            rc = ["rgba(61,82,160,.7)","rgba(112,145,230,.7)","rgba(173,187,218,.7)"]
            for i,(_, row) in enumerate(df.head(3).iterrows()):
                v = [row["work_life_balance"],row["creativity_level"],row["social_interaction"],
                     row["remote_possibility"],min(10,row["growth_rate"]/5),min(10,row["avg_salary_usd"]/35000)]
                fig.add_trace(go.Scatterpolar(r=v+[v[0]],theta=cats+[cats[0]],fill="toself",
                    name=row["career"][:20],fillcolor=rc[i],
                    line=dict(color=rc[i].replace(".7","1"))))
            fig.update_layout(polar=dict(bgcolor="rgba(255,255,255,.7)",
                radialaxis=dict(visible=True,range=[0,10],tickfont=dict(color="#8697C4"),gridcolor="#e8eaf6"),
                angularaxis=dict(tickfont=dict(color="#1a1a2e"),gridcolor="#e8eaf6")),
                paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#1a1a2e",family="Plus Jakarta Sans"),
                legend=dict(bgcolor="rgba(255,255,255,.8)"),margin=dict(l=20,r=20,t=40,b=20),
                title=dict(text="Top 3 — Attribute Radar",font=dict(family="Syne",size=13,color="#3D52A0")))
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='pf-card-title'>🤖 ML Model Picks</div>", unsafe_allow_html=True)
        for career, prob in ml:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border);">
              <span style="font-size:.85rem;">{career}</span>
              <span style="color:var(--primary);font-family:Syne,sans-serif;font-weight:800;">{prob}%</span>
            </div>""", unsafe_allow_html=True)
        fig2 = px.bar(df.head(10),x="match_score",y="career",orientation="h",
            color="match_score",color_continuous_scale=[[0,"#EDE8F5"],[0.5,"#7091E6"],[1,"#3D52A0"]],
            title="Top 10 Match Scores")
        fig2.update_layout(**pf_layout(),coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# APP: ROADMAP  (age-aware selectbox + custom text + career image)
# ═══════════════════════════════════════════════════════════════════
def get_career_image(career_name: str, w: int = 400, h: int = 220) -> str:
    CAREER_PHOTO_IDS = {
        "Software Engineer":       "1461749280684-dccba630e2f6",
        "Data Scientist":          "1551288049-bebda4e38f71",
        "AI/ML Engineer":          "1677442135703-1787eea5ce01",
        "Cybersecurity Analyst":   "1550751827-4bd374c3f58b",
        "Cloud Engineer":          "1544197150-b99a580bb7a8",
        "Mobile Developer":        "1512941937669-90a1b58e7e9c",
        "DevOps Engineer":         "1518432031352-d6fc5c10da5a",
        "Blockchain Developer":    "1639762681485-074b7f938ba0",
        "UX Designer":             "1561070791-2526d30994b5",
        "Product Manager":         "1552664730-d307ca884978",
        "Game Developer":          "1493711662062-fa541adb3fc8",
        "Data Engineer":           "1558494949-ef010cbdcc31",
        "QA Engineer":             "1587620962725-abab7fe55159",
        "IT Project Manager":      "1531403009284-440f080d1e12",
        "Robotics Engineer":       "1485827404703-89b55fcc595e",
        "General Physician":       "1579684385127-1ef15d508118",
        "Surgeon":                 "1559757175-5700dde675bc",
        "Dentist":                 "1606811841689-23dfddce3e95",
        "Nurse":                   "1584515933487-779824d29309",
        "Pharmacist":              "1587854692152-cbe660dbde88",
        "Psychologist":            "1573497019940-1c28c88b4f3e",
        "Physiotherapist":         "1571019613454-1cb2f99b2d8b",
        "Nutritionist":            "1490645935967-10de6ba17061",
        "Radiologist":             "1516549655169-df83a0774514",
        "Civil Engineer":          "1503387762-592deb58ef4e",
        "Mechanical Engineer":     "1537462715879-360eeb61a0ad",
        "Electrical Engineer":     "1621905251189-08b45d6a269e",
        "Chemical Engineer":       "1532187863486-abf9dbad1b69",
        "Architect":               "1486325212027-8081e485255e",
        "Environmental Engineer":  "1497435334941-8c899ee9e8e9",
        "Financial Analyst":       "1611974789855-9c2a0a7236a3",
        "Investment Banker":       "1560472354-b33ff0c44a43",
        "Actuary":                 "1554224155-6726b3ff858f",
        "Accountant":              "1450101499163-c8848c66ca85",
        "Marketing Manager":       "1533750349088-cd871a92f312",
        "Entrepreneur":            "1507003211169-0a1dd7228f2d",
        "HR Manager":              "1521737711867-e3b97375f902",
        "Supply Chain Manager":    "1586528116311-ad8dd3c8310d",
        "Business Analyst":        "1460925895917-afdab827c52f",
        "School Teacher":          "1509062522246-3755977927d7",
        "University Professor":    "1524178232363-1fb2b075b655",
        "Educational Consultant":  "1434030216411-0b793f4b4173",
        "Graphic Designer":        "1626785774573-4b799315345d",
        "Content Creator":         "1598387993441-a364f854cfbd",
        "Journalist":              "1504711434969-e33886168f5c",
        "Musician":                "1511379938547-c1f69419868d",
        "Film Director":           "1485846234645-a62644f84728",
        "Interior Designer":       "1586023492125-27b2c045efd7",
        "Marine Biologist":        "1518020382113-a7e8fc38eac9",
        "Environmental Scientist": "1542601906897-c6e0dce2e2e6",
        "Research Scientist":      "1507413245164-6160d8298b31",
        "Lawyer":                  "1589391886645-d51941baf7fb",
        "Social Worker":           "1559027615-cd4628902d4a",
        "Pilot":                   "1436491865332-7a61a109cc05",
        "Chef":                    "1556909114-f6e7ad7d3136",
        "Hotel Manager":           "1566073771259-6a8506099945",
    }
    pid = CAREER_PHOTO_IDS.get(career_name)
    if pid:
        return f"https://images.unsplash.com/photo-{pid}?w={w}&h={h}&fit=crop&auto=format"
    return f"https://images.unsplash.com/photo-1521737604893-d14cc237f11d?w={w}&h={h}&fit=crop&auto=format"


def get_level_image(career_name: str, level_idx: int) -> str:
    LEVEL_PHOTO_IDS = [
        ["1434030216411-0b793f4b4173", "1513258496099-48168024aec0", "1523240795612-9a054b0db644"],
        ["1552664730-d307ca884978",    "1521737711867-e3b97375f902", "1460925895917-afdab827c52f"],
        ["1507003211169-0a1dd7228f2d", "1531403009284-440f080d1e12", "1556761175-4b46a572b786"],
    ]
    ids  = LEVEL_PHOTO_IDS[level_idx % 3]
    pick = abs(hash(career_name)) % 3
    return f"https://images.unsplash.com/photo-{ids[pick]}?w=400&h=200&fit=crop&auto=format"


# ═══════════════════════════════════════════════════════════════════
# LLM ROADMAP GENERATOR
# ═══════════════════════════════════════════════════════════════════
def get_ai_roadmap(career, age, profile):
    if age < 14:
        persona = "Fun and engaging Career Guide for Kids (Age 11-13)."
        tone    = "Exciting, simple, encouraging. Use emojis. Keep language very simple."
        stage_logic = f"""
## Stage 1: Explorer (Age {age}-14)
Fun tools: Scratch, Minecraft Education, Khan Academy Kids, Tynker.
Focus on curiosity and play.

## Stage 2: Skill Builder (Age 14-17)
Join school clubs, take relevant subjects, start small projects.
Try competitions like Science Olympiad or coding clubs.

## Stage 3: Future Star (Age 18+)
University or vocational training. Specialise in {career}."""
        portfolio_req = "Suggest a Digital Treasure Chest — Google Drive folder with drawings, code projects, or school awards."

    elif age < 18:
        persona = "Strategic High School Career Counselor."
        tone    = "Motivational, structured, ambitious."
        stage_logic = f"""
## Stage 1: Strong Foundation (Age {age}-16)
Focus on high academic scores in relevant subjects. Build base skills.

## Stage 2: Exposure and Competitions (Age 16-18)
Join Olympiads, hackathons, internships. Build portfolio for college.

## Stage 3: University Entry (Age 18+)
Choose the right major for {career}. Prepare strong applications."""
        portfolio_req = "Build a GitHub repo or personal website to showcase projects for college admissions."

    else:
        persona = "Senior Career Coach and Mentor."
        tone    = "Professional, direct, practical. Use real course names and salaries."
        stage_logic = f"""
## Stage 1: Upskilling (Now - 1 Year)
Close skill gaps with certifications (Coursera, Udemy, edX). Build portfolio.

## Stage 2: Entry Level (1-3 Years)
Land first {career} job. Network on LinkedIn. Learn on the job fast.

## Stage 3: Growth (3+ Years)
Specialise, lead teams, negotiate salary. Target senior {career} roles."""
        portfolio_req = "Optimise LinkedIn profile and create a Case Study Portfolio showing real impact."

    user_message = f"""
Target Career: {career}
User Age: {age}
Profile: {json.dumps({k:v for k,v in profile.items() if k in ['creativity','social','work_life_balance','income_priority','cognitive_pref','academic_level','location']})}

{stage_logic}

### Skills to Build Now
List 5-6 key skills age-appropriate for {career}.

### Recommended Courses and Resources
Specific platforms with course names. Age-appropriate.

### Hands-on Projects (3 projects)
Concrete, doable project ideas for {career}.

### Certifications and Achievements
Age-appropriate: school awards for kids, professional certs for adults.

### Portfolio Strategy
{portfolio_req}
"""
    system_instruction = f"You are {persona}. {tone} Return in clean Markdown format."
    try:
        return groq_complete([{"role": "user", "content": user_message}],
                             system=system_instruction, max_tokens=2000)
    except Exception as e:
        return f"### Error\n\n{e}\n\nPlease try again."


# ═══════════════════════════════════════════════════════════════════
# APP: ROADMAP PAGE
# ═══════════════════════════════════════════════════════════════════
def app_roadmap():
    st.markdown("<h2>Your Career Journey</h2>", unsafe_allow_html=True)
    profile = st.session_state.user_profile
    if not profile:
        st.info("Complete your profile first.")
        return

    df           = load_career_data()
    careers_list = sorted(df["career"].tolist())
    age          = int(profile.get("age", 18))

    if age <= 13:
        kid_careers = ["Game Developer","Graphic Designer","Content Creator","Musician",
                       "Architect","Environmental Scientist","Marine Biologist","Chef",
                       "Film Director","School Teacher","Robotics Engineer"]
        options = sorted(set(kid_careers) & set(careers_list)) + ["Other (Type your own)"]
    elif age <= 17:
        teen_careers = ["Software Engineer","Data Scientist","AI/ML Engineer","UX Designer",
                        "Game Developer","Cybersecurity Analyst","Mobile Developer",
                        "Content Creator","Graphic Designer","Architect","Journalist",
                        "Musician","Film Director","Entrepreneur","General Physician",
                        "Lawyer","Environmental Scientist","Marine Biologist",
                        "Psychologist","School Teacher","Robotics Engineer","Cloud Engineer"]
        options = sorted(set(teen_careers) & set(careers_list)) + ["Other (Type your own)"]
    else:
        options = careers_list + ["Other (Type your own)"]

    current_career = st.session_state.get("selected_career", options[0] if options else "")
    default_idx    = options.index(current_career) if current_career in options else len(options) - 1

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_opt = st.selectbox("Select Career", options, index=default_idx, key="rm_select")
        if selected_opt == "Other (Type your own)":
            final_career = st.text_input(
                "Enter Career Name",
                value=st.session_state.get("custom_career_input", ""),
                placeholder="e.g. Footballer, Quantum Physicist...",
                key="rm_text",
            )
            st.session_state.custom_career_input = final_career
        else:
            final_career = selected_opt
    with c2:
        age = st.number_input("Your Age", 10, 65, age, key="rm_age")

    if st.button("Generate Full Journey Plan", use_container_width=True, type="primary"):
        if not final_career or not final_career.strip():
            st.warning("Please enter or select a career name.")
        else:
            with st.spinner("AI is designing your path..."):
                st.session_state.roadmap         = get_ai_roadmap(final_career, age, profile)
                st.session_state.selected_career = final_career
            st.rerun()

    if not st.session_state.roadmap:
        return

    sel  = st.session_state.selected_career
    img  = get_career_image(sel)
    r_df = df[df["career"] == sel]

    # build badge strings separately to avoid f-string nesting
    if not r_df.empty:
        r      = r_df.iloc[0]
        bc     = "#ef4444" if r["burnout_risk"] >= 8 else "#f59e0b" if r["burnout_risk"] >= 6 else "#22c55e"
        ind    = "<span class='badge badge-teal'>" + str(r['industry']) + "</span>"
        sal    = "<span class='badge badge-yellow'>&#128176; $" + f"{int(r['avg_salary_usd']):,}" + "/yr</span>"
        gro    = "<span class='badge badge-purple'>&#128200; " + str(r['growth_rate']) + "% growth</span>"
        br_val = str(r['burnout_risk'])
        br_html = (
            "<div style='text-align:right;'>"
            "<div style='color:var(--muted);font-size:.78rem;'>Burnout Risk</div>"
            "<div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:900;color:" + bc + ";'>" + br_val + "/10</div>"
            "</div>"
        )
        avg_sal = int(r['avg_salary_usd'])
    else:
        r       = {}
        ind     = "<span class='badge badge-blue'>Custom Path</span>"
        sal     = "<span class='badge badge-yellow'>&#128176; Variable</span>"
        gro     = "<span class='badge badge-purple'>&#128200; High Potential</span>"
        br_html = ""
        avg_sal = 100000

    # CSS
    st.markdown("""
    <style>
    .rm-hero-wrap { overflow:hidden; }
    .rm-hero-img  { transition:transform .4s ease; width:100%; height:100%; object-fit:cover; display:block; }
    .rm-hero-wrap:hover .rm-hero-img { transform:scale(1.05); }
    .prog-card {
        border:2px solid #e2e8f0; border-radius:18px; overflow:hidden;
        background:white; box-shadow:0 4px 16px rgba(61,82,160,.07);
        transition:transform .3s, box-shadow .3s, border-color .3s;
        margin-bottom: 0;
    }
    .prog-card:hover {
        transform:translateY(-8px);
        box-shadow:0 18px 40px rgba(61,82,160,.2);
        border-color:var(--secondary);
    }
    .prog-img { transition:transform .4s ease; width:100%; height:130px; object-fit:cover; display:block; }
    .prog-card:hover .prog-img { transform:scale(1.06); }
    </style>""", unsafe_allow_html=True)

    # TOP CARD
    top_html = (
        "<div class='pf-card' style='padding:0;overflow:hidden;border-radius:20px;'>"
        "<div style='display:flex;align-items:stretch;min-height:170px;'>"
        "<div class='rm-hero-wrap' style='width:260px;min-width:260px;position:relative;'>"
        "<img class='rm-hero-img' src='" + img + "' alt='" + sel + "'>"
        "<div style='position:absolute;inset:0;background:linear-gradient(90deg,transparent 55%,white 100%);'></div>"
        "</div>"
        "<div style='padding:24px 28px;flex:1;display:flex;flex-direction:column;justify-content:center;'>"
        "<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
        "<div>"
        "<div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:900;color:var(--primary);margin-bottom:10px;'>" + sel + "</div>"
        "<div style='display:flex;gap:6px;flex-wrap:wrap;'>" + ind + sal + gro + "</div>"
        "</div>"
        + br_html +
        "</div>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(top_html, unsafe_allow_html=True)

    # TIMELINE
    current_age = int(profile.get("age", 18))
    if current_age < 14:
        stages = [
            {"title":"Explorer",     "sub":"Age " + str(current_age), "status":"active", "icon":"🧸"},
            {"title":"Skill Builder","sub":"Age 14-16",                "status":"future", "icon":"🧱"},
            {"title":"Future Star",  "sub":"Age 17+",                  "status":"future", "icon":"🚀"},
        ]
    elif current_age < 18:
        stages = [
            {"title":"School Prep",  "sub":"Completed",               "status":"done",   "icon":"🏫"},
            {"title":"Competitions", "sub":"Age " + str(current_age), "status":"active", "icon":"🏆"},
            {"title":"University",   "sub":"Age 18+",                 "status":"future", "icon":"🎓"},
        ]
    else:
        stages = [
            {"title":"Education",   "sub":"Completed",                "status":"done",   "icon":"🎓"},
            {"title":"Entry Level", "sub":"Current Focus",            "status":"active", "icon":"🚀"},
            {"title":"Leadership",  "sub":"Future Goal",              "status":"future", "icon":"👑"},
        ]

    tl = "<div class='journey-container'><div class='journey-line'></div>"
    for s in stages:
        tl += (
            "<div class='journey-step step-" + s['status'] + "'>"
            "<div class='step-circle'>" + s['icon'] + "</div>"
            "<div class='step-title'>" + s['title'] + "</div>"
            "<div class='step-sub'>" + s['sub'] + "</div>"
            "</div>"
        )
    tl += "</div>"
    st.markdown(tl, unsafe_allow_html=True)

    # ACTION PLAN
    st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pf-card-title'>Detailed Action Plan</div>", unsafe_allow_html=True)

    col_text, col_side = st.columns([4, 1])
    with col_side:
        side_html = (
            "<div style='text-align:center;padding:12px 0;'>"
            "<div style='border-radius:14px;overflow:hidden;"
            "box-shadow:0 6px 20px rgba(61,82,160,.18);width:110px;margin:0 auto;'>"
            "<img src='" + img + "' width='110' height='110' "
            "style='object-fit:cover;display:block;transition:transform .35s;' "
            "onmouseover=\"this.style.transform='scale(1.09)'\" "
            "onmouseout=\"this.style.transform='scale(1)'\">"
            "</div>"
            "<div style='margin-top:9px;font-size:.75rem;font-weight:700;color:var(--primary);'>" + sel + "</div>"
            "</div>"
        )
        st.markdown(side_html, unsafe_allow_html=True)

    with col_text:
        h1c, h2c, h3c, tc = "#3D52A0", "#7091E6", "#8697C4", "#334155"
        lines    = st.session_state.roadmap.split('\n')
        html_out = ""
        in_list  = False
        for line in lines:
            s = line.strip()
            if s.startswith("- "):
                if not in_list:
                    html_out += "<ul style='list-style:none;padding-left:0;'>"
                in_list   = True
                html_out += (
                    "<li style='margin-bottom:8px;padding-left:20px;position:relative;"
                    "color:" + tc + ";line-height:1.6;'>"
                    "<span style='position:absolute;left:0;top:4px;color:" + h2c + ";'>&#9679;</span> "
                    + s[2:] + "</li>"
                )
            else:
                if in_list:
                    html_out += "</ul>"
                    in_list   = False
                if s.startswith("### "):
                    html_out += "<h3 style='color:" + h3c + ";border-bottom:2px solid " + h3c + ";padding-bottom:5px;margin-top:25px;font-family:Syne,sans-serif;'>" + s[4:] + "</h3>"
                elif s.startswith("## "):
                    html_out += "<h2 style='color:" + h2c + ";border-bottom:2px solid " + h2c + ";padding-bottom:5px;margin-top:30px;font-family:Syne,sans-serif;'>" + s[3:] + "</h2>"
                elif s.startswith("# "):
                    html_out += "<h1 style='color:" + h1c + ";border-bottom:3px solid " + h1c + ";padding-bottom:8px;margin-top:35px;font-family:Syne,sans-serif;'>" + s[2:] + "</h1>"
                elif s:
                    html_out += "<p style='color:" + tc + ";line-height:1.7;margin-bottom:15px;'>" + s + "</p>"
        if in_list:
            html_out += "</ul>"
        st.markdown(html_out, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # PROGRESSION CARDS
    st.markdown("<h3 style='margin-top:30px;'>Career Progression</h3>", unsafe_allow_html=True)

    if current_age < 14:
        levels = [
            {"label":"Beginner",     "desc":"Learning the basics — fun first!", "salary":None},
            {"label":"Intermediate", "desc":"Building cool projects.",           "salary":None},
            {"label":"Expert",       "desc":"Teaching and inspiring others.",    "salary":None},
        ]
    else:
        levels = [
            {"label":"Junior",    "desc":"Learning the ropes.",       "salary":int(avg_sal * 0.6)},
            {"label":"Mid-Level", "desc":"Independent work.",         "salary":int(avg_sal)},
            {"label":"Senior",    "desc":"Leadership. High impact.",  "salary":int(avg_sal * 1.6)},
        ]

    prog_cols = st.columns(3)
    for i, job in enumerate(levels):
        with prog_cols[i]:
            card_img = get_level_image(sel, i)
            if job["salary"]:
                sal_html = "<div style='font-weight:800;color:#059669;font-size:1.05rem;'>$" + f"{job['salary']:,}" + "</div>"
                focus_label = "Est. Annual Salary"
            else:
                sal_html    = "<div style='font-weight:700;color:var(--primary);'>Learning Phase</div>"
                focus_label = "Current Focus"

            card_html = (
                "<div class='prog-card'>"
                "<div style='overflow:hidden;'>"
                "<img class='prog-img' src='" + card_img + "' alt='" + job['label'] + "'>"
                "</div>"
                "<div style='position:relative;'>"
                "<div style='position:absolute;top:-14px;left:16px;"
                "background:var(--primary);color:white;"
                "padding:3px 14px;border-radius:999px;"
                "font-size:.72rem;font-weight:700;"
                "box-shadow:0 2px 8px rgba(61,82,160,.3);'>"
                + job['label'] +
                "</div>"
                "</div>"
                "<div style='padding:20px 16px 16px;'>"
                "<h4 style='margin:8px 0 6px;color:var(--primary);"
                "font-family:Syne,sans-serif;font-size:.95rem;'>"
                + job['label'] + " " + sel +
                "</h4>"
                "<div style='font-size:.8rem;color:#64748b;margin-bottom:14px;'>" + job['desc'] + "</div>"
                "<div style='background:#f8fafc;padding:10px;border-radius:10px;'>"
                "<div style='font-size:.7rem;color:#94a3b8;margin-bottom:3px;'>" + focus_label + "</div>"
                + sal_html +
                "</div>"
                "</div>"
                "</div>"
            )
            st.markdown(card_html, unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════
# APP: INSTITUTE FINDER  (FIXED — uses load_institute_data)
# ═══════════════════════════════════════════════════════════════════
def app_institutes():
    st.markdown("<h2>🏫 Institute Finder</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);'>Find the best universities, schools, bootcamps & online platforms for your career path.</p>",
                unsafe_allow_html=True)

    profile = st.session_state.user_profile
    if not profile:
        st.warning("Please complete your profile first.")
        if st.button("Go to Profile"): st.session_state.app_page="profile"; st.rerun()
        return

    df_inst = load_institute_data()
    if df_inst.empty:
        st.error("Institute database is empty."); return

    city_col = 'city' if 'city' in df_inst.columns else 'location'
    all_cities = sorted(df_inst[city_col].dropna().unique().tolist())
    all_types  = sorted(df_inst['type'].dropna().unique().tolist())

    career_list = []
    if st.session_state.career_matches is not None:
        try:
            career_list = st.session_state.career_matches.head(5)["career"].tolist()
        except: pass

    selected_career_ctx = st.session_state.get("selected_career","")

    # ── FILTERS (main area, not sidebar to avoid conflict) ─────────
    st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    with f1:
        default_city_idx = 0
        if profile.get("location"):
            loc = profile["location"].split(",")[0].strip()
            if loc in all_cities: default_city_idx = all_cities.index(loc)
        city = st.selectbox("📍 City", all_cities, index=default_city_idx, key="inst_city")
    with f2:
        type_filter = st.multiselect("🏛️ Type", all_types, default=all_types, key="inst_type")
    with f3:
        degree_filter = st.selectbox("🎓 Level",
            ["Any Level","Primary School","High School","Undergraduate","Graduate","Online / Bootcamp"],
            key="inst_degree")
    with f4:
        career_opts = ["All Careers"] + career_list + (
            [selected_career_ctx] if selected_career_ctx and selected_career_ctx not in career_list else [])
        default_career_idx = 0
        if selected_career_ctx in career_opts: default_career_idx = career_opts.index(selected_career_ctx)
        career_focus = st.selectbox("Career Focus", career_opts, index=default_career_idx, key="inst_career")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── AI STRATEGY BUTTON ─────────────────────────────────────────
    if st.button("🤖 Get AI Strategy for My Path", use_container_width=True, type="secondary"):
        with st.spinner(" Generating personalised strategy..."):
            sys_p = "You are a career counselor helping a student choose the right institute. Be specific and practical."
            user_p = (f"Student in {city}, age {profile.get('age',18)}, "
                      f"looking for {degree_filter} education in {career_focus}. "
                      f"Financial range: {profile.get('financial_range','$30K-$60K')}. "
                      f"Give 3 specific actionable tips for finding the right institute.")
            response = groq_complete([{"role":"user","content":user_p}], system=sys_p, max_tokens=400)
        st.markdown(f"""<div class="pf-card" style="border-left:4px solid var(--primary);">
          <div class="pf-card-title"> AI Strategy</div>
          <div style="color:var(--text);line-height:1.7;">{response}</div>
        </div>""", unsafe_allow_html=True)

    # ── FILTERING ──────────────────────────────────────────────────
    filtered = df_inst[df_inst[city_col] == city].copy()
    if type_filter:
        filtered = filtered[filtered['type'].isin(type_filter)]
    if career_focus != "All Careers":
        kws = career_focus.lower().split()
        mask = filtered['career_field'].str.lower().apply(
            lambda x: any(kw in x for kw in kws) if isinstance(x,str) else False)
        filtered = filtered[mask]
    if degree_filter != "Any Level":
        level_map = {
            "Primary School":     ["Primary School","High","Middle"],
            "High School":        ["High School","High","Diploma","College"],
            "Undergraduate":      ["Undergraduate","High School"],
            "Graduate":           ["Graduate","Undergraduate"],
            "Online / Bootcamp":  ["Undergraduate","High School","Primary School"],
        }
        allowed = level_map.get(degree_filter, [])
        if allowed:
            filtered = filtered[filtered['academic_level'].apply(
                lambda x: any(a.lower() in str(x).lower() for a in allowed))]

    st.markdown(f"### 📍 Results in **{city}** — {len(filtered)} found", unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No institutes found. Try changing the city or removing some filters.")
        return

    # ── DISPLAY CARDS (2-col grid) ─────────────────────────────────
    cols = st.columns(2)
    for i,(_, inst) in enumerate(filtered.iterrows()):
        with cols[i % 2]:
            safe_seed = str(inst['name']).replace(" ","")
            img_url = f"https://api.dicebear.com/7.x/shapes/svg?seed={safe_seed}&backgroundColor=3D52A0,7091E6,ADBBDA"
            scholarship_badge = "<span class='badge badge-green'>🎓 Scholarship</span>" if str(inst.get('scholarship','')).lower() in ['true','yes','1'] else ""
            fee_text = "Free" if int(inst.get('fee_max',0)) == 0 else f"${int(inst.get('fee_max',0)):,}/yr"
            ranking = int(inst.get('ranking', 5))
            stars = "⭐" * min(ranking, 5)
            website = inst.get('website','#')

            st.markdown(f"""
            <div class="inst-card">
              <div style="background:linear-gradient(135deg,#3D52A0,#7091E6);height:120px;
                display:flex;align-items:center;justify-content:center;position:relative;">
                <img src="{img_url}" width="80" height="80"
                  style="border-radius:50%;border:3px solid white;background:white;">
                <div style="position:absolute;top:10px;right:12px;background:rgba(255,255,255,.2);
                  backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.4);
                  border-radius:20px;padding:3px 10px;color:white;font-size:.72rem;font-weight:700;">
                  {inst['type']}
                </div>
              </div>
              <div style="padding:20px;">
                <h3 style="margin:0 0 6px;color:var(--primary);font-family:'Syne',sans-serif;font-size:1.05rem;">
                  {inst['name']}
                </h3>
                <div style="margin-bottom:10px;">
                  <span class='badge badge-teal'>{inst.get('country','')}</span>
                  <span class='badge badge-purple'>{inst.get('academic_level','')}</span>
                  {scholarship_badge}
                </div>
                <p style="color:var(--muted);font-size:.82rem;line-height:1.5;margin-bottom:12px;">
                  <strong>Focus:</strong> {str(inst['career_field'])[:80]}{'...' if len(str(inst['career_field']))>80 else ''}
                </p>
                <div style="display:flex;justify-content:space-between;align-items:center;
                  background:#f8fafc;padding:10px 14px;border-radius:10px;">
                  <div>
                    <div style="font-size:.7rem;color:var(--muted);">Annual Fee</div>
                    <div style="font-weight:800;color:var(--primary);">{fee_text}</div>
                  </div>
                  <div style="text-align:right;">
                    <div style="font-size:.7rem;color:var(--muted);">Rating</div>
                    <div style="font-size:.85rem;">{stars} ({ranking}/10)</div>
                  </div>
                  <div>
                    <a href="https://{website}" target="_blank"
                      style="background:var(--primary);color:white;padding:6px 14px;
                      border-radius:8px;font-size:.75rem;font-weight:700;text-decoration:none;">
                      Visit →
                    </a>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# APP: RESUME ANALYZER
# ═══════════════════════════════════════════════════════════════════
def app_resume():
    st.markdown("<h2>📄 AI Resume Architect</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);'>Optimize your resume with AI-driven ATS scoring and persona alignment.</p>",
                unsafe_allow_html=True)
    profile = st.session_state.user_profile or {}
    df = load_career_data()
    default_career = st.session_state.get("selected_career","Auto-detect")
    career_options = ["Auto-detect"] + df["career"].tolist()
    if default_career not in career_options: default_career = "Auto-detect"

    st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([2,2,1])
    with col1: uploaded = st.file_uploader("Upload Resume", type=["pdf","docx","txt"], label_visibility="collapsed")
    with col2: tc = st.selectbox("Target Career", career_options, index=career_options.index(default_career), key="res_career")
    with col3: age = st.number_input("Age", 10, 65, int(profile.get("age",18)) if profile else 20, key="res_age")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("⚙️ System Status"):
        st.success("✅ PyPDF2 ready") if HAS_PDF else st.error("Missing: pip install PyPDF2")
        st.success("✅ docx2txt ready") if HAS_DOCX else st.error("Missing: pip install docx2txt")

    analyze_btn = st.button("🔍 Run Deep Analysis", use_container_width=True, disabled=uploaded is None)
    if analyze_btn and uploaded:
        with st.spinner("🤖 Analyzing structure, keywords, and ATS compatibility..."):
            text = read_resume(uploaded)
            career = "" if tc=="Auto-detect" else tc
            raw_analysis = analyze_resume(text, age, profile, career)
            st.session_state.resume_analysis = {"text":text,"raw_analysis":raw_analysis,"career":career,"age":age}
        st.rerun()

    if st.session_state.resume_analysis:
        res = st.session_state.resume_analysis
        word_count = len(res["text"].split())
        ats_score = min(95, 60 + (word_count//10)) if word_count>50 else 40
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("ATS Score", f"{ats_score}%", delta="Good Fit" if ats_score>70 else "Needs Work")
        with c2: st.metric("Word Count", word_count, delta="Optimal" if 200<word_count<800 else "Check Length")
        with c3: st.markdown(f"""<div style="text-align:center;"><div style="color:var(--muted);font-size:.8rem;">Target Role</div>
          <div style="font-weight:700;color:var(--primary);font-size:1.1rem;">{res['career'] or 'General'}</div></div>""",
          unsafe_allow_html=True)
        with c4: st.markdown(f"""<div style="text-align:center;"><div style="color:var(--muted);font-size:.8rem;">Age Group</div>
          <div style="font-weight:700;color:var(--primary);font-size:1.1rem;">{res['age']} yrs</div></div>""",
          unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        tab1,tab2,tab3 = st.tabs(["📊 AI Insights","Gap Analysis","Raw Text"])
        with tab1:
            cl,cr = st.columns([2,1])
            with cl:
                st.markdown(f"""<div class='pf-card'>
                  <div class='pf-card-title'>🤖 Detailed AI Feedback</div>
                  <div style='line-height:1.6;color:var(--text);'>{res['raw_analysis']}</div>
                </div>""", unsafe_allow_html=True)
            with cr:
                av = res["age"]
                tip = (" **Child Focus:** Highlight fun projects, school competitions, curiosity." if av<14 else
                       " **Teen Focus:** Emphasize academic rigor, extracurricular leadership, tech skills." if av<19 else
                       " **Adult Focus:** Lead with impact, quantifiable results, specific technical tools.")
                st.markdown(f"""<div class="pf-card" style="border-top:4px solid var(--primary);">
                  <div class="pf-card-title">💡 Age Strategy</div>
                  <p style="color:var(--muted);margin:0;font-size:.9rem;">{tip}</p>
                </div>""", unsafe_allow_html=True)
                st.download_button("📥 Download Report", data=res["raw_analysis"],
                                   file_name="resume_feedback.txt", mime="text/plain", use_container_width=True)
        with tab2:
            st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
            st.markdown("<div class='pf-card-title'>🧩 Profile vs Resume Gap</div>", unsafe_allow_html=True)
            profile_skills = profile.get("fav_subjects",[])
            resume_lower = res["text"].lower()
            if profile_skills:
                found = [s for s in profile_skills if s.lower() in resume_lower]
                miss  = [s for s in profile_skills if s.lower() not in resume_lower]
                if found:
                    st.markdown("**Found in Resume ✅**")
                    fc = st.columns(3)
                    for i,s in enumerate(found):
                        with fc[i%3]: st.markdown(f"<span style='background:#dcfce7;color:#166534;padding:5px 10px;border-radius:5px;font-size:.85rem;'>✅ {s}</span>", unsafe_allow_html=True)
                if miss:
                    st.markdown("**Missing from Resume ❌**")
                    mc = st.columns(3)
                    for i,s in enumerate(miss):
                        with mc[i%3]: st.markdown(f"<span style='background:#fee2e2;color:#991b1b;padding:5px 10px;border-radius:5px;font-size:.85rem;'>❌ {s}</span>", unsafe_allow_html=True)
                    st.warning(" Add these missing skills to boost your ATS score!")
                elif not miss:
                    st.success(" All profile skills are in your resume!")
            else:
                st.info("Add subjects to your profile to enable gap analysis.")
            st.markdown("</div>", unsafe_allow_html=True)
        with tab3:
            st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
            st.text_area("Extracted Text", res["text"], height=400, disabled=True, label_visibility="collapsed")
            st.caption(f"Extracted {len(res['text'])} characters.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👆 Upload a resume above to unlock the AI dashboard.")

# ═══════════════════════════════════════════════════════════════════
# APP: AI CHAT
# ═══════════════════════════════════════════════════════════════════
def app_chat():
    st.markdown("<h2>💬 AI Career Advisor</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--muted);'>Ask anything about careers, education, skills, or your future. "
        "Powered by Llama 3.3 70B.</p>",
        unsafe_allow_html=True,
    )

    profile = st.session_state.user_profile
    sys_p = (
        "You are PathFinder AI, an expert career counselor and life coach. "
        "Give specific, actionable, personalized advice. Be encouraging and data-driven. "
        f"User profile: {json.dumps(profile) if profile else 'Not provided.'}"
    )

    # ── EMPTY STATE ───────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;background:white;
          border-radius:20px;border:2px solid var(--border);margin-bottom:20px;">
          <div style="font-size:2.8rem;margin-bottom:10px;">🧭</div>
          <div style="font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;color:var(--primary);">
            Ask me anything about your career journey</div>
          <div style="color:var(--muted);font-size:.88rem;margin-top:6px;">
            Career choices · Skills · Education paths · Salary info · Work-life balance</div>
        </div>""", unsafe_allow_html=True)

    # ── CHAT HISTORY DISPLAY ──────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-user'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-ai'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # ── QUICK QUESTIONS ───────────────────────────────────────────
    quick = [
        "What career suits a creative introvert?",
        "How to avoid burnout?",
        "Best tech careers 2026?",
        "How to switch careers at 30?",
    ]
    qcols = st.columns(4)
    for col, q in zip(qcols, quick):
        with col:
            if st.button(q, key=f"q_{q[:15]}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    resp = groq_complete(st.session_state.chat_history, system=sys_p)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
                st.rerun()

    # ── INPUT BOX + SEND ──────────────────────────────────────────
    # KEY FIX: use a form so the input clears after submit,
    # preventing the repeated-answer loop on every rerun.
    with st.form(key="chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            inp = st.text_input(
                "",
                key="chat_inp",
                label_visibility="collapsed",
                placeholder="e.g. What skills should a 17-year-old learn to become a Data Scientist?",
            )
        with c2:
            send = st.form_submit_button("Send")

    # Only process when the form was actually submitted with non-empty text
    if send and inp.strip():
        user_msg = inp.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.spinner("Thinking..."):
            resp = groq_complete(st.session_state.chat_history, system=sys_p)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
        st.rerun()

    # ── CLEAR BUTTON ──────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# APP: MARKET INSIGHTS
# ═══════════════════════════════════════════════════════════════════
def app_insights():
    st.markdown("<h2>📊 Market Analysis & Insights</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);'>Explore career trends, salary benchmarks, automation risks, and demand indicators.</p>",
                unsafe_allow_html=True)
    df = load_career_data()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Avg Salary",f"${df['avg_salary_usd'].mean():,.0f}")
    with c2: st.metric("Avg Growth Rate",f"{df['growth_rate'].mean():.1f}%")
    with c3: st.metric("Avg Automation Risk",f"{df['automation_risk'].mean():.1f}%")
    with c4: st.metric("High-Demand Careers",str(len(df[df['growth_rate']>=20])))
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: inds = st.multiselect("Filter Industry",df["industry"].unique().tolist(),default=df["industry"].unique().tolist())
    with c2: sort_by = st.selectbox("Sort by",["avg_salary_usd","growth_rate","work_life_balance","remote_possibility","burnout_risk"])
    with c3: top_n = st.slider("Show top N",5,len(df),min(20,len(df)))
    fdf = df[df["industry"].isin(inds)].sort_values(sort_by,ascending=False).head(top_n)
    t1,t2,t3,t4 = st.tabs(["💰 Salary","📈 Growth & Automation","⚖️ Work-Life","🔬 Correlations"])
    with t1:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.bar(fdf,x="career",y="avg_salary_usd",color="industry",
                title=f"Salary — Top {top_n}",color_discrete_sequence=PF_COLORS)
            fig.update_layout(**pf_layout(xaxis=dict(gridcolor="#e8eaf6",tickangle=-40,linecolor="#d0d7f0")))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2 = px.box(df[df["industry"].isin(inds)],x="industry",y="avg_salary_usd",
                color="industry",title="Salary by Industry",color_discrete_sequence=PF_COLORS)
            fig2.update_layout(**pf_layout(showlegend=False,xaxis=dict(gridcolor="#e8eaf6",tickangle=-30,linecolor="#d0d7f0")))
            st.plotly_chart(fig2,use_container_width=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("<div class='pf-card'><div class='pf-card-title'>🏆 Top 5 Highest Paid</div>",unsafe_allow_html=True)
            for _,row in df.nlargest(5,"avg_salary_usd").iterrows():
                w=(row["avg_salary_usd"]/df["avg_salary_usd"].max())*100
                st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border);">
                  <div style="min-width:150px;font-size:.85rem;font-weight:600;">{row['career']}</div>
                  <div style="flex:1;background:#e8eaf6;border-radius:999px;height:7px;">
                    <div style="width:{w:.0f}%;height:7px;border-radius:999px;background:linear-gradient(90deg,#3D52A0,#7091E6);"></div></div>
                  <div style="color:var(--primary);font-family:Syne,sans-serif;font-weight:800;min-width:80px;text-align:right;">${row['avg_salary_usd']:,}</div>
                </div>""",unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='pf-card'><div class='pf-card-title'>📚 Entry-Level Friendly</div>",unsafe_allow_html=True)
            for _,row in df.nsmallest(5,"avg_salary_usd").iterrows():
                w=(row["avg_salary_usd"]/df["avg_salary_usd"].max())*100
                st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border);">
                  <div style="min-width:150px;font-size:.85rem;font-weight:600;">{row['career']}</div>
                  <div style="flex:1;background:#e8eaf6;border-radius:999px;height:7px;">
                    <div style="width:{w:.0f}%;height:7px;border-radius:999px;background:linear-gradient(90deg,#22c55e,#86efac);"></div></div>
                  <div style="color:#16a34a;font-family:Syne,sans-serif;font-weight:800;min-width:80px;text-align:right;">${row['avg_salary_usd']:,}</div>
                </div>""",unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)
    with t2:
        c1,c2 = st.columns(2)
        with c1:
            fig3=px.scatter(df[df["industry"].isin(inds)],x="growth_rate",y="automation_risk",
                size="avg_salary_usd",color="industry",hover_name="career",
                title="Growth vs Automation Risk",color_discrete_sequence=PF_COLORS)
            fig3.update_layout(**pf_layout(xaxis=dict(gridcolor="#e8eaf6",title="Growth Rate (%)",linecolor="#d0d7f0"),
                yaxis=dict(gridcolor="#e8eaf6",title="Automation Risk (%)",linecolor="#d0d7f0")))
            st.plotly_chart(fig3,use_container_width=True)
        with c2:
            fig4=px.bar(fdf.sort_values("growth_rate",ascending=False).head(15),
                x="growth_rate",y="career",orientation="h",color="growth_rate",
                color_continuous_scale=[[0,"#EDE8F5"],[1,"#3D52A0"]],title="Top Growing Careers")
            fig4.update_layout(**pf_layout(coloraxis_showscale=False))
            st.plotly_chart(fig4,use_container_width=True)
        for cat,color,fn in [
            ("🟢 High Demand","#22c55e",lambda r:r["growth_rate"]>=20 and r["automation_risk"]<30),
            ("🟡 Medium Demand","#f59e0b",lambda r:r["growth_rate"]>=10 and r["automation_risk"]<50),
            ("🔴 Automation Risk","#ef4444",lambda r:r["automation_risk"]>=40),
        ]:
            cs=[r["career"] for _,r in df[df["industry"].isin(inds)].iterrows() if fn(r)][:12]
            if cs:
                st.markdown(f"""<div style="padding:10px 16px;background:white;border-left:4px solid {color};
                  border-radius:8px;margin:6px 0;box-shadow:0 2px 8px rgba(0,0,0,.05);">
                  <strong style="color:{color};">{cat}</strong>
                  <span style="color:var(--muted);font-size:.83rem;margin-left:10px;">{" · ".join(cs)}</span>
                </div>""",unsafe_allow_html=True)
    with t3:
        c1,c2 = st.columns(2)
        with c1:
            fig5=px.scatter(df[df["industry"].isin(inds)],x="work_life_balance",y="burnout_risk",
                color="industry",hover_name="career",size="avg_salary_usd",
                title="Work-Life Balance vs Burnout Risk",color_discrete_sequence=PF_COLORS)
            fig5.update_layout(**pf_layout(xaxis=dict(gridcolor="#e8eaf6",title="Work-Life Balance",linecolor="#d0d7f0"),
                yaxis=dict(gridcolor="#e8eaf6",title="Burnout Risk",linecolor="#d0d7f0")))
            st.plotly_chart(fig5,use_container_width=True)
        with c2:
            st.markdown("<div class='pf-card'><div class='pf-card-title'>🏠 Remote Work Leaders</div>",unsafe_allow_html=True)
            for _,row in df.nlargest(10,"remote_possibility").iterrows():
                pct=row["remote_possibility"]*10
                st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border);">
                  <div style="min-width:160px;font-size:.82rem;font-weight:600;">{row['career']}</div>
                  <div style="flex:1;background:#e8eaf6;border-radius:999px;height:6px;">
                    <div style="width:{pct:.0f}%;height:6px;border-radius:999px;background:linear-gradient(90deg,#7091E6,#3D52A0);"></div></div>
                  <div style="color:var(--primary);font-weight:800;font-size:.82rem;min-width:30px;">{row['remote_possibility']}/10</div>
                </div>""",unsafe_allow_html=True)
            st.markdown("</div>",unsafe_allow_html=True)
    with t4:
        num_cols=["avg_salary_usd","growth_rate","automation_risk","work_life_balance",
                  "creativity_level","social_interaction","remote_possibility","burnout_risk"]
        corr=df[num_cols].corr()
        fig6=go.Figure(data=go.Heatmap(z=corr.values,x=corr.columns.tolist(),y=corr.columns.tolist(),
            colorscale=[[0,"#EDE8F5"],[0.5,"#7091E6"],[1,"#3D52A0"]],zmin=-1,zmax=1,
            text=corr.round(2).values,texttemplate="%{text}"))
        fig6.update_layout(**pf_layout(title="Feature Correlation Matrix",height=480))
        st.plotly_chart(fig6,use_container_width=True)
        c1,c2=st.columns(2)
        with c1:
            fig7=px.scatter(df,x="avg_salary_usd",y="burnout_risk",color="industry",
                hover_name="career",trendline="ols",title="Salary vs Burnout Risk",color_discrete_sequence=PF_COLORS)
            fig7.update_layout(**pf_layout()); st.plotly_chart(fig7,use_container_width=True)
        with c2:
            fig8=px.histogram(df,x="avg_salary_usd",nbins=15,color="industry",
                title="Salary Distribution",color_discrete_sequence=PF_COLORS,barmode="overlay",opacity=.75)
            fig8.update_layout(**pf_layout()); st.plotly_chart(fig8,use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# APP: MODEL TRAINING  (uses career_guidance_dataset.csv with
#                       Recommended_Career_Path target column)
# ═══════════════════════════════════════════════════════════════════
def app_training():
    warnings.filterwarnings("ignore")
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except: HAS_XGB = False
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    st.markdown("<h2>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--muted);'>Supervised ML on the 9,500-row student career dataset.</p>",
                unsafe_allow_html=True)

    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ Dataset not found: `{DATASET_PATH}`. Place `career_guidance_dataset.csv` in the same folder.")
        return

    df_raw = pd.read_csv(DATASET_PATH)
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    TARGET = "Recommended_Career_Path"
    if TARGET not in df.columns:
        st.error(f"❌ Column '{TARGET}' not found. Available: {list(df.columns)}")
        return

    if 'train_data' not in st.session_state:
        with st.spinner("🔧 Preprocessing data..."):
            df_clean = df.copy()
            drop_cols = [c for c in ["Student_ID","Name"] if c in df_clean.columns]
            df_clean.drop(columns=drop_cols, inplace=True)
            y_raw = df_clean.pop(TARGET)
            for col in df_clean.columns:
                if df_clean[col].dtype == "object":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            le_dict = {}
            for col in df_clean.select_dtypes(include="object").columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                le_dict[col] = le
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str))
            class_names = le_target.classes_
            X = df_clean.values.astype(float)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state['train_data'] = {
                'df_clean':df_clean,'scaler':scaler,'le_target':le_target,
                'class_names':class_names,'le_dict':le_dict,'drop_cols':drop_cols,
                'X_scaled':X_scaled,'y':y
            }

    ds = st.session_state['train_data']
    df_clean   = ds['df_clean'];   scaler   = ds['scaler']
    le_target  = ds['le_target'];  class_names = ds['class_names']
    le_dict    = ds['le_dict'];    drop_cols   = ds['drop_cols']
    X_scaled   = ds['X_scaled'];   y           = ds['y']

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Records",len(df))
    with c2: st.metric("Features",len(df.columns)-1)
    with c3: st.metric("Career Classes",df[TARGET].nunique())
    with c4: st.metric("Missing Values",df.isnull().sum().sum())

    with st.expander("📊 Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
    st.markdown("<div class='pf-card-title'>⚙️ Training Configuration</div>", unsafe_allow_html=True)
    co1,co2 = st.columns(2)
    with co1:
        test_size = st.slider("Test Split",0.10,0.40,0.20,0.05)
        cv_folds  = st.slider("CV Folds",3,10,5)
    model_choices = ["Logistic Regression","Naive Bayes","KNN","Random Forest"]
    if HAS_XGB: model_choices += ["XGBoost"]
    with co2:
        models_to_run = st.multiselect("Select Models",model_choices,
            default=["Logistic Regression","Random Forest"] + (["XGBoost"] if HAS_XGB else []))
    st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button("Train All Selected Models", use_container_width=True, type="primary")
    if run_btn and models_to_run:
        with st.spinner(" Training Models..."):
            X_tr,X_te,y_tr,y_te = train_test_split(X_scaled,y,test_size=test_size,random_state=42,stratify=y)
            MODEL_MAP = {
                "Logistic Regression": LogisticRegression(max_iter=500,random_state=42),
                "SVM":                 SVC(kernel="rbf",probability=True,random_state=42),
                "Naive Bayes":         GaussianNB(),
                "KNN":                 KNeighborsClassifier(n_neighbors=5),
                "Random Forest":       RandomForestClassifier(n_estimators=200,max_depth=12,random_state=42,n_jobs=-1),
            }
            if HAS_XGB:
                MODEL_MAP["XGBoost"] = XGBClassifier(n_estimators=200,max_depth=6,
                    use_label_encoder=False,eval_metric="mlogloss",random_state=42,verbosity=0)
            results = {}
            for name in models_to_run:
                clf = MODEL_MAP[name]; clf.fit(X_tr,y_tr); y_pred=clf.predict(X_te)
                acc = accuracy_score(y_te,y_pred)
                cv  = cross_val_score(clf,X_scaled,y,cv=cv_folds,scoring="accuracy")
                results[name] = {"model":clf,"accuracy":acc,"cv_mean":cv.mean(),"cv_std":cv.std(),
                                 "y_pred":y_pred}
            st.session_state['train_results'] = results
            st.success("✅ Training Complete!")

    results = st.session_state.get('train_results',{})
    if results:
        st.markdown("<br><h3>📈 Model Comparison</h3>", unsafe_allow_html=True)
        best_name = max(results,key=lambda n:results[n]["accuracy"])
        st.success(f"🏆 Best Model: **{best_name}** — **{results[best_name]['accuracy']*100:.1f}%** accuracy")
        acc_df = pd.DataFrame({"Model":list(results.keys()),
            "Accuracy":[r["accuracy"]*100 for r in results.values()]}).sort_values("Accuracy",ascending=False)
        fig_acc = px.bar(acc_df,x="Model",y="Accuracy",
            text=acc_df["Accuracy"].apply(lambda x:f"{x:.1f}%"),color="Accuracy",
            color_continuous_scale=[[0,"#EDE8F5"],[0.5,"#7091E6"],[1,"#3D52A0"]],title="Model Accuracy (%)")
        fig_acc.update_traces(textposition="outside")
        fig_acc.update_layout(**pf_layout(coloraxis_showscale=False,yaxis=dict(range=[0,115])))
        st.plotly_chart(fig_acc,use_container_width=True)

        st.markdown("---")
        st.markdown("<h3>⚡ Live Career Predictor</h3>", unsafe_allow_html=True)
        pred_model_name = st.selectbox("Model",list(results.keys()),key="pred_model")
        original_df_nodrop = df.drop(columns=[TARGET]+drop_cols,errors="ignore")
        feat_names = df_clean.columns.tolist()
        input_vals = []
        st.markdown("<div class='pf-card'>", unsafe_allow_html=True)
        cols_ui = st.columns(3)
        for idx_f,feat in enumerate(feat_names):
            orig_col = original_df_nodrop[feat] if feat in original_df_nodrop.columns else None
            with cols_ui[idx_f % 3]:
                if orig_col is not None and orig_col.dtype == "object":
                    unique_vals = sorted(orig_col.dropna().unique().tolist())
                    selected = st.selectbox(feat,unique_vals,key=f"live_{feat}")
                    encoded = le_dict[feat].transform([str(selected)])[0]
                    input_vals.append(float(encoded))
                else:
                    col_min=float(df_clean[feat].min()); col_max=float(df_clean[feat].max())
                    col_med=float(df_clean[feat].median())
                    val = st.slider(feat,col_min,col_max,col_med,key=f"live_{feat}")
                    input_vals.append(val)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button(" Predict Now",use_container_width=True,type="primary"):
            raw_input = np.array([input_vals])
            scaled_input = scaler.transform(raw_input)
            clf_live = results[pred_model_name]["model"]
            pred_label = le_target.inverse_transform(clf_live.predict(scaled_input))[0]
            if hasattr(clf_live,"predict_proba"):
                probs = clf_live.predict_proba(scaled_input)[0]
                top_idx = np.argsort(probs)[::-1][:5]
                st.markdown(f"""<div class='pf-card'>
                  <div class='pf-card-title'> Predicted: <span style='color:#22c55e;'>{pred_label}</span></div>""",
                  unsafe_allow_html=True)
                for rank,ix in enumerate(top_idx):
                    pct = probs[ix]*100; clr = PF_COLORS[rank%len(PF_COLORS)]
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:14px;margin:8px 0;">
                      <div style="color:{clr};font-family:Syne,sans-serif;font-weight:900;width:26px;">#{rank+1}</div>
                      <div style="flex:1;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                          <span style="font-weight:600;">{le_target.classes_[ix]}</span>
                          <span style="color:{clr};font-family:Syne,sans-serif;font-weight:800;">{pct:.1f}%</span>
                        </div>
                        <div class="score-bar-bg">
                          <div class="score-bar-fill" style="width:{pct:.1f}%;background:{clr};"></div>
                        </div>
                      </div>
                    </div>""",unsafe_allow_html=True)
                st.markdown("</div>",unsafe_allow_html=True)
            else:
                st.success(f" Predicted: **{pred_label}**")
    else:
        st.info("👆 Train a model above to unlock the Live Predictor.")

# ═══════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════
def main():
    render_nav()
    if st.session_state.logged_in:
        render_sidebar()
        pg = st.session_state.app_page
        {
            "home":     app_home,
            "profile":  app_profile,
            "matches":  app_matches,
            "roadmap":  app_roadmap,
            "institute":app_institutes,
            "resume":   app_resume,
            "chat":     app_chat,
            "insights": app_insights,
            "training": app_training,
        }.get(pg, app_home)()
    else:
        pg = st.session_state.page
        if   pg == "landing": page_landing()
        elif pg == "about":   page_about()
        elif pg == "auth":    page_auth()
        else:                 page_landing()

if __name__ == "__main__":
    main()
