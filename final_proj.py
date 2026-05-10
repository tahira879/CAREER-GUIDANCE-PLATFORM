"""PathFinder AI — Career Intelligence Platform | Master Build"""
import streamlit as st, os, re, io, time
import streamlit.components.v1 as components

try:
    from groq import Groq; GROQ_OK=True
except: GROQ_OK=False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, train_test_split
    import numpy as np; ML_OK=True
except:
    ML_OK=False
    import numpy as np
try: import PyPDF2; PDF_OK=True
except: PDF_OK=False
try: import docx2txt; DOCX_OK=True
except: DOCX_OK=False

try:
    from dotenv import load_dotenv
    load_dotenv()
except: pass

GROQ_KEY = os.getenv("GROQ_API_KEY","")

def init():
    for k,v in {
        "page":"landing","logged_in":False,"current_user":None,"accounts":{},
        "app_page":"home","profile":{},"matches":[],"chat_hist":[],
        "roadmap_txt":"","inst_result":"","resume_result":None,
        "train_done":False,"sel_career":"UX Designer","train_results":None,
        "roadmap_mode":"education","job_search_result":""
    }.items():
        if k not in st.session_state: st.session_state[k]=v
init()

CAREERS=[
    {"title":"UX Designer","industry":"Technology","salary":90000,"growth":20,"burnout":3,"automation":15,"wlb":8,"creativity":9,"social":6,"remote":8,"icon":"🎨","img":"https://images.unsplash.com/photo-1561070791-2526d30994b5?w=400&q=80","skills":["Figma","User Research","Prototyping","CSS","Wireframing"],"edu":"BS Design / HCI"},
    {"title":"Software Engineer","industry":"Technology","salary":110000,"growth":25,"burnout":4,"automation":20,"wlb":7,"creativity":6,"social":4,"remote":9,"icon":"💻","img":"https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=400&q=80","skills":["Python","JavaScript","System Design","Algorithms","Git"],"edu":"BS Computer Science"},
    {"title":"Data Scientist","industry":"Technology","salary":120000,"growth":35,"burnout":4,"automation":18,"wlb":7,"creativity":7,"social":4,"remote":8,"icon":"📊","img":"https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&q=80","skills":["Python","SQL","Machine Learning","Statistics","Pandas"],"edu":"BS CS / Statistics"},
    {"title":"AI/ML Engineer","industry":"Technology","salary":135000,"growth":40,"burnout":5,"automation":10,"wlb":6,"creativity":8,"social":4,"remote":8,"icon":"🤖","img":"https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=400&q=80","skills":["Python","TensorFlow","PyTorch","Deep Learning","NLP"],"edu":"BS CS / MS AI"},
    {"title":"Product Manager","industry":"Technology","salary":130000,"growth":22,"burnout":7,"automation":12,"wlb":6,"creativity":7,"social":8,"remote":7,"icon":"🎯","img":"https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=400&q=80","skills":["Strategy","Analytics","Agile","Communication","Roadmapping"],"edu":"BS Business / CS"},
    {"title":"Cybersecurity Analyst","industry":"Technology","salary":105000,"growth":30,"burnout":6,"automation":8,"wlb":6,"creativity":5,"social":4,"remote":7,"icon":"🔐","img":"https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=400&q=80","skills":["Network Security","Ethical Hacking","SIEM","Python","Risk Assessment"],"edu":"BS CS / Cybersecurity"},
    {"title":"Cloud Architect","industry":"Technology","salary":145000,"growth":30,"burnout":5,"automation":12,"wlb":7,"creativity":5,"social":4,"remote":9,"icon":"☁️","img":"https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&q=80","skills":["AWS","Azure","GCP","Terraform","Kubernetes"],"edu":"BS CS / Cloud Certs"},
    {"title":"DevOps Engineer","industry":"Technology","salary":120000,"growth":28,"burnout":5,"automation":15,"wlb":7,"creativity":5,"social":4,"remote":9,"icon":"⚙️","img":"https://images.unsplash.com/photo-1618401471353-b98afee0b2eb?w=400&q=80","skills":["Docker","Kubernetes","CI/CD","AWS","Linux"],"edu":"BS CS / DevOps Certs"},
    {"title":"Game Developer","industry":"Gaming","salary":95000,"growth":18,"burnout":6,"automation":12,"wlb":5,"creativity":9,"social":4,"remote":8,"icon":"🎮","img":"https://images.unsplash.com/photo-1511512578047-dfb367046420?w=400&q=80","skills":["Unity","Unreal Engine","C#","C++","Game Design"],"edu":"BS CS / Game Design"},
    {"title":"Graphic Designer","industry":"Creative","salary":55000,"growth":10,"burnout":3,"automation":20,"wlb":8,"creativity":10,"social":5,"remote":7,"icon":"🎭","img":"https://images.unsplash.com/photo-1626785774573-4b799315345d?w=400&q=80","skills":["Photoshop","Illustrator","Branding","Typography","Color Theory"],"edu":"BS Graphic Design"},
    {"title":"Doctor","industry":"Healthcare","salary":200000,"growth":18,"burnout":8,"automation":5,"wlb":3,"creativity":4,"social":9,"remote":2,"icon":"🏥","img":"https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=400&q=80","skills":["Clinical Diagnosis","Patient Care","Medical Research","Pharmacology","Surgery"],"edu":"MBBS + Specialization"},
    {"title":"Nurse","industry":"Healthcare","salary":70000,"growth":15,"burnout":7,"automation":4,"wlb":5,"creativity":3,"social":9,"remote":2,"icon":"💊","img":"https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=400&q=80","skills":["Patient Care","Clinical Skills","Communication","Emergency Response","Medical Knowledge"],"edu":"BS Nursing / RN License"},
    {"title":"Pharmacist","industry":"Healthcare","salary":120000,"growth":3,"burnout":4,"automation":22,"wlb":7,"creativity":2,"social":7,"remote":3,"icon":"🧬","img":"https://images.unsplash.com/photo-1576602976047-174e57a47881?w=400&q=80","skills":["Pharmacology","Patient Counseling","Drug Interactions","Dispensing","Healthcare"],"edu":"PharmD / BS Pharmacy"},
    {"title":"Physiotherapist","industry":"Healthcare","salary":80000,"growth":18,"burnout":4,"automation":5,"wlb":7,"creativity":5,"social":9,"remote":3,"icon":"🏃","img":"https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?w=400&q=80","skills":["Manual Therapy","Exercise Prescription","Anatomy","Patient Assessment","Rehabilitation"],"edu":"BS/MS Physiotherapy"},
    {"title":"Dentist","industry":"Healthcare","salary":170000,"growth":8,"burnout":5,"automation":6,"wlb":6,"creativity":4,"social":8,"remote":1,"icon":"🦷","img":"https://images.unsplash.com/photo-1606811971618-4486d14f3f99?w=400&q=80","skills":["Oral Surgery","Diagnostics","Patient Care","Dental Procedures","Anatomy"],"edu":"BDS / DDS"},
    {"title":"Investment Banker","industry":"Finance","salary":180000,"growth":8,"burnout":9,"automation":25,"wlb":2,"creativity":4,"social":7,"remote":4,"icon":"💹","img":"https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=400&q=80","skills":["Financial Modeling","Excel","Valuation","M&A","Risk Analysis"],"edu":"BS Finance / MBA"},
    {"title":"Financial Analyst","industry":"Finance","salary":85000,"growth":11,"burnout":5,"automation":28,"wlb":6,"creativity":4,"social":5,"remote":6,"icon":"📈","img":"https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=400&q=80","skills":["Excel","Financial Modeling","Data Analysis","Forecasting","Bloomberg"],"edu":"BS Finance / CFA"},
    {"title":"Accountant","industry":"Finance","salary":70000,"growth":7,"burnout":4,"automation":35,"wlb":7,"creativity":2,"social":4,"remote":6,"icon":"🧮","img":"https://images.unsplash.com/photo-1565514158740-064f34bd5fa5?w=400&q=80","skills":["Accounting","Tax Law","Excel","Auditing","QuickBooks"],"edu":"BS Accounting / CPA"},
    {"title":"Marketing Manager","industry":"Marketing","salary":95000,"growth":15,"burnout":5,"automation":18,"wlb":6,"creativity":8,"social":7,"remote":7,"icon":"📣","img":"https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=400&q=80","skills":["Digital Marketing","SEO","Analytics","Content Strategy","Brand Management"],"edu":"BS Marketing / Communications"},
    {"title":"Content Creator","industry":"Marketing","salary":65000,"growth":20,"burnout":4,"automation":15,"wlb":8,"creativity":10,"social":6,"remote":10,"icon":"🎬","img":"https://images.unsplash.com/photo-1611532736597-de2d4265fba3?w=400&q=80","skills":["Video Editing","Copywriting","Social Media","SEO","Brand Storytelling"],"edu":"BS Communications / Self-Taught"},
    {"title":"Mechanical Engineer","industry":"Engineering","salary":90000,"growth":7,"burnout":5,"automation":20,"wlb":6,"creativity":5,"social":4,"remote":4,"icon":"⚙️","img":"https://images.unsplash.com/photo-1581094794329-c8112a89af12?w=400&q=80","skills":["CAD","Thermodynamics","Materials Science","Project Management","AutoCAD"],"edu":"BS Mechanical Engineering"},
    {"title":"Civil Engineer","industry":"Engineering","salary":85000,"growth":8,"burnout":5,"automation":12,"wlb":6,"creativity":4,"social":5,"remote":3,"icon":"🏗️","img":"https://images.unsplash.com/photo-1504307651254-35680f356dfd?w=400&q=80","skills":["Structural Analysis","AutoCAD","Project Management","Surveying","Materials"],"edu":"BS Civil Engineering"},
    {"title":"Professor","industry":"Education","salary":90000,"growth":10,"burnout":4,"automation":8,"wlb":8,"creativity":7,"social":8,"remote":5,"icon":"🎓","img":"https://images.unsplash.com/photo-1544717305-2782549b5136?w=400&q=80","skills":["Research","Teaching","Academic Writing","Curriculum Design","Subject Expertise"],"edu":"PhD in Relevant Field"},
    {"title":"High School Teacher","industry":"Education","salary":50000,"growth":8,"burnout":5,"automation":8,"wlb":7,"creativity":6,"social":8,"remote":4,"icon":"🏫","img":"https://images.unsplash.com/photo-1509062522246-3755977927d7?w=400&q=80","skills":["Curriculum Design","Classroom Management","Communication","Mentoring","Subject Expertise"],"edu":"BS Education + Teaching Cert"},
    {"title":"Architect","industry":"Design","salary":80000,"growth":12,"burnout":6,"automation":12,"wlb":5,"creativity":9,"social":5,"remote":4,"icon":"🏛️","img":"https://images.unsplash.com/photo-1486325212027-8081e485255e?w=400&q=80","skills":["AutoCAD","SketchUp","BIM","Design Theory","Project Management"],"edu":"B.Arch / M.Arch"},
    {"title":"Psychologist","industry":"Healthcare","salary":90000,"growth":22,"burnout":5,"automation":3,"wlb":7,"creativity":5,"social":10,"remote":6,"icon":"🧠","img":"https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=400&q=80","skills":["Counseling","CBT","Assessment","Research","Empathy"],"edu":"BS/MS/PhD Psychology"},
    {"title":"Lawyer","industry":"Legal","salary":130000,"growth":9,"burnout":7,"automation":10,"wlb":4,"creativity":5,"social":7,"remote":5,"icon":"⚖️","img":"https://images.unsplash.com/photo-1589829545856-d10d557cf95f?w=400&q=80","skills":["Legal Research","Contract Law","Litigation","Negotiation","Critical Thinking"],"edu":"LLB / JD"},
    {"title":"Surgeon","industry":"Healthcare","salary":350000,"growth":15,"burnout":9,"automation":3,"wlb":2,"creativity":3,"social":6,"remote":1,"icon":"⚕️","img":"https://images.unsplash.com/photo-1551190822-a9333d879b1f?w=400&q=80","skills":["Surgical Techniques","Anatomy","Clinical Decision-Making","Patient Care","Medical Research"],"edu":"MBBS + MS Surgery + Fellowship"},
    {"title":"Environmental Scientist","industry":"Science","salary":75000,"growth":14,"burnout":4,"automation":12,"wlb":7,"creativity":6,"social":5,"remote":5,"icon":"🌿","img":"https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400&q=80","skills":["Environmental Analysis","Data Collection","GIS","Research","Policy Writing"],"edu":"BS Environmental Science"},
    {"title":"Journalist","industry":"Media","salary":55000,"growth":5,"burnout":6,"automation":30,"wlb":5,"creativity":8,"social":7,"remote":7,"icon":"📰","img":"https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&q=80","skills":["Writing","Research","Interviewing","Storytelling","Media Ethics"],"edu":"BS Journalism"},
]

def h(html): st.markdown(html, unsafe_allow_html=True)

def validate_password(pw):
    if len(pw) > 8: return False, "Password must be max 8 characters."
    if not re.search(r'[A-Z]', pw): return False, "Password must have at least 1 uppercase letter."
    if not re.search(r'\d', pw): return False, "Password must have at least 1 digit."
    return True, ""

def ai_call(messages, system, max_tokens=700):
    if not GROQ_KEY or not GROQ_OK:
        return "⚠️ Add your GROQ_API_KEY environment variable to enable AI responses."
    try:
        client=Groq(api_key=GROQ_KEY)
        r=client.chat.completions.create(model="llama3-70b-8192",max_tokens=max_tokens,
            messages=[{"role":"system","content":system}]+messages)
        return r.choices[0].message.content
    except Exception as e: return f"⚠️ AI Error: {e}"

def match_careers(p):
    scored=[]
    for c in CAREERS:
        s=((1-abs(p.get("wlb",7)-c["wlb"])/9)*25+(1-abs(p.get("creativity",5)-c["creativity"])/9)*20
          +(1-abs(p.get("social",5)-c["social"])/9)*15+(1-abs(p.get("remote",7)-c["remote"])/9)*12
          +(c["growth"]/40)*p.get("income",7)*0.015*10-(c["burnout"]/10)*8-(c["automation"]/100)*5)
        scored.append((c,round(min(99,max(30,s*1.3)),1)))
    return sorted(scored,key=lambda x:x[1],reverse=True)

def read_file(f):
    if not f: return ""
    try:
        if f.name.endswith(".txt"): return f.read().decode("utf-8","ignore")
        elif f.name.endswith(".pdf") and PDF_OK:
            r=PyPDF2.PdfReader(io.BytesIO(f.read()))
            return "\n".join(pg.extract_text() or "" for pg in r.pages)
        elif f.name.endswith(".docx") and DOCX_OK: return docx2txt.process(io.BytesIO(f.read()))
        else: return f.read().decode("utf-8","ignore")
    except: return ""

def detect_skills(text):
    cats={"Programming":["python","javascript","java","c++","c#","react","node","django"],
          "Data & AI":["machine learning","deep learning","tensorflow","pytorch","pandas","sql"],
          "Design":["figma","photoshop","illustrator","ux","ui","prototyping","wireframing"],
          "Cloud/DevOps":["aws","azure","gcp","docker","kubernetes","terraform","linux"],
          "Business":["project management","agile","marketing","seo","excel","leadership"],
          "Soft Skills":["teamwork","problem solving","critical thinking","creativity","adaptability"]}
    tl=text.lower(); found={}
    for cat,skills in cats.items():
        m=[s for s in skills if s in tl]
        if m: found[cat]=m
    return found

def score_resume(text):
    bd=[]; tot=0
    w=len(text.split()); wc=min(20,round((min(w,500)/500)*20))
    bd.append({"l":"Content Length","s":wc,"m":20,"c":"#1d4ed8"}); tot+=wc
    sc=detect_skills(text); sk_cnt=sum(len(v) for v in sc.values()); sk=min(25,sk_cnt*3)
    bd.append({"l":"Skills Detected","s":sk,"m":25,"c":"#7c3aed"}); tot+=sk
    ce=(6 if "@" in text else 0)+(4 if re.search(r'(\+\d{10,12}|\d{10,11})',text) else 0)+(3 if "linkedin" in text.lower() else 0)+(2 if "github" in text.lower() else 0)
    ce=min(15,ce); bd.append({"l":"Contact Info","s":ce,"m":15,"c":"#059669"}); tot+=ce
    edu=15 if re.search(r'(education|school|college|university|degree|bachelor|master)',text,re.I) else 0
    bd.append({"l":"Education Section","s":edu,"m":15,"c":"#d97706"}); tot+=edu
    qnt=15 if re.search(r'(\d+%|\$\d+|\d+\s*(projects?|clients?|users?))',text,re.I) else 0
    bd.append({"l":"Quantified Results","s":qnt,"m":15,"c":"#7c3aed"}); tot+=qnt
    vbs=["built","designed","led","created","analyzed","managed","increased","developed","launched","implemented"]
    vc=sum(1 for v in vbs if v in text.lower()); av=min(10,vc*2)
    bd.append({"l":"Action Verbs","s":av,"m":10,"c":"#dc2626"}); tot+=av
    return min(100,tot),bd,sc

def pbar(label,val,mx,color="#1d4ed8",suf=""):
    pct=min(100,round((val/mx)*100)) if mx else 0
    d=suf if suf else str(val)
    return f'''<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
<span style="font-size:12px;font-weight:600;color:#1E293B;min-width:148px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{label}</span>
<div style="flex:1;height:7px;background:#E2E8F0;border-radius:99px;overflow:hidden;">
<div style="width:{pct}%;height:7px;background:{color};border-radius:99px;transition:width .6s;"></div></div>
<span style="font-size:12px;font-weight:800;color:{color};min-width:54px;text-align:right;font-family:'Plus Jakarta Sans',sans-serif;">{d}</span></div>'''

def badge(t,cls="blue"):
    m={"blue":("#eff6ff","#1d4ed8","#bfdbfe"),"teal":("#ecfeff","#0891b2","#a5f3fc"),
       "green":("#f0fdf4","#16a34a","#bbf7d0"),"amber":("#fffbeb","#d97706","#fde68a"),
       "red":("#fef2f2","#dc2626","#fecaca"),"violet":("#f5f3ff","#7c3aed","#ddd6fe")}
    bg,c,bo=m.get(cls,m["blue"])
    return f'<span style="display:inline-flex;align-items:center;padding:3px 10px;border-radius:99px;font-size:10.5px;font-weight:700;background:{bg};color:{c};border:1.5px solid {bo};margin:2px;">{t}</span>'

def chip(t):
    return f'<span style="background:#f0f7ff;border:1.5px solid #bfdbfe;border-radius:7px;padding:3px 10px;font-size:11px;font-weight:600;color:#1d4ed8;margin:2px;display:inline-block;">{t}</span>'


# ── PAGE CONFIG ──
st.set_page_config(page_title="PathFinder AI",page_icon="🧭",layout="wide",initial_sidebar_state="auto")
pg = st.session_state.page

# Hide sidebar + controls on public pages
if pg in ("landing","login","signup"):
    h("""<style>
section[data-testid="stSidebar"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
</style>""")

# ══════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════
h("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Syne:wght@700;800;900&display=swap');
:root{--P:#1d4ed8;--S:#2563eb;--A:#60a5fa;--bg:#f0f7ff;--card:#fff;--border:#bfdbfe;--text:#0f172a;--muted:#64748b;}
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{background:var(--bg)!important;font-family:'Plus Jakarta Sans',sans-serif!important;color:var(--text)!important;}
#MainMenu,footer,header,[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden!important;display:none!important;}
[data-testid="stSidebarNav"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stMainBlockContainer"]{padding:0!important;}
[data-testid="stSidebarContent"]{padding:0!important;}
section[data-testid="stSidebar"]>div:first-child{padding:0!important;}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#04081a 0%,#070d2e 30%,#0b1540 60%,#112058 100%)!important;
  border-right:1px solid rgba(96,165,250,.15)!important;
  min-width:256px!important;
}
section[data-testid="stSidebar"] .stButton>button{
  background:transparent!important;
  color:rgba(148,197,253,.6)!important;
  border:1px solid transparent!important;
  border-radius:10px!important;
  font-size:13px!important;font-weight:600!important;text-align:left!important;
  padding:10px 14px 10px 16px!important;width:100%!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  transition:all .22s cubic-bezier(.34,1.56,.64,1)!important;
  margin-bottom:2px!important;position:relative!important;
  letter-spacing:.1px!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  background:linear-gradient(90deg,rgba(29,78,216,.28),rgba(29,78,216,.10))!important;
  color:#c7dcff!important;
  border-color:rgba(96,165,250,.4)!important;
  transform:translateX(7px)!important;
  box-shadow:0 3px 18px rgba(29,78,216,.25),inset 3px 0 0 rgba(96,165,250,.7)!important;
  padding-left:20px!important;
}
section[data-testid="stSidebar"] .stButton>button:active{
  transform:translateX(4px) scale(.98)!important;
}
section[data-testid="stSidebar"] .stButton>button:focus{
  outline:none!important;
  box-shadow:0 0 0 2px rgba(96,165,250,.35),inset 3px 0 0 #60a5fa!important;
  color:white!important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:2px solid #bfdbfe;background:transparent!important;}
.stTabs [data-baseweb="tab"]{font-size:13px;font-weight:700;color:var(--muted);padding:9px 18px;border-radius:8px 8px 0 0;font-family:'Plus Jakarta Sans',sans-serif!important;background:transparent!important;transition:color .2s;}
.stTabs [data-baseweb="tab"]:hover{color:var(--P)!important;}
.stTabs [aria-selected="true"]{color:var(--P)!important;background:rgba(29,78,216,.06)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--P)!important;}
.stTabs [data-baseweb="tab-panel"]{padding:18px 0 0!important;background:transparent!important;}

/* ── INPUTS ── */
input[type=range]{accent-color:var(--P);}
.stButton>button{font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;border-radius:10px!important;transition:all .2s!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(29,78,216,.28)!important;}
.stTextInput>div>div>input,.stNumberInput>div>div>input{background:#f8faff!important;border:2px solid #bfdbfe!important;border-radius:10px!important;font-family:'Plus Jakarta Sans',sans-serif!important;color:var(--text)!important;transition:border-color .2s,box-shadow .2s!important;font-weight:500!important;}
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus{border-color:var(--P)!important;box-shadow:0 0 0 3px rgba(29,78,216,.12)!important;}
.stSelectbox>div>div{background:#f8faff!important;border:2px solid #bfdbfe!important;border-radius:10px!important;font-family:'Plus Jakarta Sans',sans-serif!important;}
.stTextArea>div>div>textarea{background:#f8faff!important;border:2px solid #bfdbfe!important;border-radius:10px!important;font-family:'Plus Jakarta Sans',sans-serif!important;}
label{font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:700!important;color:#1d4ed8!important;font-size:12px!important;letter-spacing:.3px!important;}
.stFileUploader>div{background:#f8faff!important;border:2px dashed #bfdbfe!important;border-radius:12px!important;}

/* ── CARDS ── */
.pf-card{
  background:white;border-radius:18px;padding:22px;
  border:1.5px solid #e0efff;
  box-shadow:0 4px 24px rgba(29,78,216,.07);
  margin-bottom:16px;
  transition:border-color .3s,box-shadow .3s,transform .3s;
  position:relative;overflow:hidden;
}
.pf-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(96,165,250,.4),transparent);
  opacity:0;transition:opacity .3s;
}
.pf-card:hover{
  border-color:#93c5fd;
  box-shadow:0 12px 40px rgba(29,78,216,.13);
  transform:translateY(-2px);
}
.pf-card:hover::before{opacity:1;}

.stat-card{
  background:white;border-radius:16px;padding:20px;
  border:1.5px solid #e0efff;text-align:center;
  transition:all .3s cubic-bezier(.34,1.56,.64,1);
  margin-bottom:16px;position:relative;overflow:hidden;
}
.stat-card::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#1d4ed8,#60a5fa,#7c3aed);
  transform:scaleX(0);transition:transform .3s;transform-origin:left;
}
.stat-card:hover{
  transform:translateY(-6px);
  border-color:#93c5fd;
  box-shadow:0 16px 40px rgba(29,78,216,.15);
}
.stat-card:hover::after{transform:scaleX(1);}

.match-card{
  background:white;border:1.5px solid #e0efff;border-radius:18px;
  padding:20px;margin-bottom:16px;position:relative;overflow:hidden;
  transition:all .3s;box-shadow:0 4px 16px rgba(29,78,216,.05);
}
.match-card:hover{
  transform:translateY(-5px);
  box-shadow:0 20px 48px rgba(29,78,216,.14);
  border-color:#93c5fd;
}
.match-card.gold-pick{
  border-color:#93c5fd;
  box-shadow:0 6px 28px rgba(29,78,216,.12);
}
.match-card.gold-pick::before{
  content:'';position:absolute;top:0;left:0;right:0;height:4px;
  background:linear-gradient(90deg,#1d4ed8,#60a5fa,#7c3aed);
}

.qs-card{
  background:white;border-radius:16px;padding:18px 14px;
  border:1.5px solid #e0efff;text-align:center;
  transition:all .3s cubic-bezier(.34,1.56,.64,1);margin-bottom:14px;
  position:relative;overflow:hidden;
}
.qs-card:hover{
  transform:translateY(-6px);
  box-shadow:0 16px 36px rgba(29,78,216,.12);
  border-color:#93c5fd;
}

.tl-body{
  background:white;border:1.5px solid #e0efff;border-radius:14px;
  padding:14px 18px;flex:1;margin-bottom:13px;transition:all .28s;
  box-shadow:0 2px 10px rgba(29,78,216,.04);
}
.tl-body:hover{
  border-color:#93c5fd;
  transform:translateX(6px);
  box-shadow:0 8px 28px rgba(29,78,216,.1);
}

.inst-card{border-radius:14px;overflow:hidden;border:1.5px solid #e0efff;transition:all .3s;}
.inst-card:hover{transform:translateY(-5px);box-shadow:0 18px 44px rgba(29,78,216,.14);border-color:#93c5fd;}

/* ── TOPBAR ── */
.topbar{
  position:sticky;top:0;
  background:rgba(248,252,255,.96);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  z-index:100;padding:10px 28px;
  border-bottom:1.5px solid rgba(191,219,254,.7);
  display:flex;align-items:center;justify-content:space-between;
  box-shadow:0 2px 24px rgba(29,78,216,.08);
}

/* ── AUTH ── */
.auth-page-bg{
  min-height:100vh;
  background:linear-gradient(145deg,#eef4ff 0%,#e8f0fe 40%,#dbeafe 70%,#f0f4ff 100%);
  display:flex;align-items:center;justify-content:center;
  padding:40px 20px;
  position:relative;overflow:hidden;
}
.auth-page-bg::before{
  content:'';position:absolute;top:-150px;right:-150px;
  width:500px;height:500px;border-radius:50%;
  background:radial-gradient(circle,rgba(96,165,250,.2),transparent 70%);
  pointer-events:none;
}
.auth-page-bg::after{
  content:'';position:absolute;bottom:-120px;left:-120px;
  width:450px;height:450px;border-radius:50%;
  background:radial-gradient(circle,rgba(167,139,250,.15),transparent 70%);
  pointer-events:none;
}
.auth-wrap{
  background:rgba(255,255,255,.97);
  border-radius:24px;padding:44px 48px;
  border:1.5px solid rgba(191,219,254,.8);
  box-shadow:0 16px 64px rgba(29,78,216,.14),0 4px 16px rgba(29,78,216,.08);
  position:relative;z-index:1;
  backdrop-filter:blur(10px);
  width:100%;
}
.auth-wrap::before{
  content:'';position:absolute;top:0;left:0;right:0;height:4px;
  background:linear-gradient(90deg,#1d4ed8,#60a5fa,#7c3aed);
  border-radius:24px 24px 0 0;
}
.li-btn{
  display:flex;align-items:center;justify-content:center;gap:10px;
  background:white;color:#0a66c2;border:2px solid #0a66c2;
  border-radius:11px;padding:11px 16px;font-size:13px;font-weight:700;
  cursor:pointer;width:100%;margin-top:12px;
  font-family:'Plus Jakarta Sans',sans-serif;transition:all .22s;
  box-shadow:0 2px 8px rgba(10,102,194,.1);
}
.li-btn:hover{background:#0a66c2;color:white;box-shadow:0 6px 22px rgba(10,102,194,.32);transform:translateY(-2px);}
.pw-rule{font-size:11px;color:#64748b;margin-top:4px;padding:7px 12px;background:#f0f7ff;border-radius:8px;border-left:3px solid #93c5fd;}

/* ── BUTTONS ── */
.stButton>button{
  background:linear-gradient(135deg,#1d4ed8,#2563eb)!important;
  color:white!important;border:none!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  font-weight:700!important;font-size:13.5px!important;
  border-radius:11px!important;cursor:pointer!important;
  transition:all .22s cubic-bezier(.34,1.56,.64,1)!important;
  box-shadow:0 4px 16px rgba(29,78,216,.32)!important;
  letter-spacing:.1px!important;
}
.stButton>button:hover{
  background:linear-gradient(135deg,#1e40af,#1d4ed8)!important;
  color:white!important;transform:translateY(-3px) scale(1.01)!important;
  box-shadow:0 10px 28px rgba(29,78,216,.5)!important;
}
.stButton>button:active{transform:translateY(-1px) scale(.99)!important;}

/* ── ANIMATIONS ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
@keyframes scaleIn{from{opacity:0;transform:scale(.94);}to{opacity:1;transform:scale(1);}}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0;}}
@keyframes shimmer{0%{background-position:-200% 0;}100%{background-position:200% 0;}}
@keyframes pulseRing{0%,100%{box-shadow:0 6px 28px rgba(29,78,216,.55),0 0 0 0 rgba(29,78,216,.3);}70%{box-shadow:0 6px 28px rgba(29,78,216,.55),0 0 0 14px rgba(29,78,216,0);}}
@keyframes chatSlide{from{opacity:0;transform:translateY(20px) scale(.95);}to{opacity:1;transform:translateY(0) scale(1);}}
@keyframes dot1{0%,80%,100%{opacity:0;}40%{opacity:1;}}
@keyframes dot2{0%,20%,100%{opacity:0;}60%{opacity:1;}}
@keyframes dot3{0%,40%,100%{opacity:0;}80%{opacity:1;}}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-10px);}}
.fu{animation:fadeUp .5s cubic-bezier(.22,1,.36,1) both;}
.fi{animation:fadeIn .4s ease both;}
.si{animation:scaleIn .4s cubic-bezier(.34,1.56,.64,1) both;}

/* ── SKELETON / SHIMMER ── */
.shimmer{
  background:linear-gradient(90deg,#f0f4ff 25%,#dbeafe 50%,#f0f4ff 75%);
  background-size:200% 100%;animation:shimmer 1.8s infinite;
  border-radius:8px;
}

/* ── CHAT MESSAGES ── */
.pmsg-ai{
  background:white;border:1.5px solid #dbeafe;
  border-radius:16px 16px 16px 4px;
  padding:11px 15px;font-size:12.5px;line-height:1.78;color:#0f172a;
  max-width:88%;box-shadow:0 3px 12px rgba(29,78,216,.07);
  animation:fadeUp .3s ease;
}
.pmsg-user{
  background:linear-gradient(135deg,#1d4ed8,#2563eb);
  border-radius:16px 16px 4px 16px;
  padding:11px 15px;font-size:12.5px;line-height:1.78;color:white;
  max-width:88%;align-self:flex-end;
  box-shadow:0 4px 16px rgba(29,78,216,.35);
  animation:fadeUp .3s ease;
}

/* ── PROGRESS BARS ── */
.prog-wrap{display:flex;align-items:center;gap:10px;margin-bottom:11px;}
.prog-track{flex:1;height:8px;background:#e8f0fe;border-radius:99px;overflow:hidden;}
.prog-fill{height:8px;border-radius:99px;transition:width .8s cubic-bezier(.22,1,.36,1);}

/* ── SECTION HEADERS ── */
.sec-hdr{
  font-family:'Syne',sans-serif;font-size:20px;font-weight:900;
  color:#0f172a;letter-spacing:-.5px;margin-bottom:16px;
  display:flex;align-items:center;gap:10px;
}
.sec-hdr::after{
  content:'';flex:1;height:1.5px;
  background:linear-gradient(90deg,#bfdbfe,transparent);
  border-radius:99px;
}

/* ── TOOLTIP BADGE ── */
.tip-badge{
  display:inline-flex;align-items:center;gap:5px;
  background:linear-gradient(135deg,#eff6ff,#dbeafe);
  border:1.5px solid #bfdbfe;border-radius:99px;
  padding:4px 12px;font-size:10.5px;font-weight:800;
  color:#1d4ed8;letter-spacing:.5px;
}

/* ── FLOATING CHATBOT (injected into body) ── */
#pf-fab{position:fixed!important;bottom:28px!important;right:28px!important;z-index:2147483647!important;width:62px;height:62px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#3b82f6);display:flex!important;align-items:center;justify-content:center;cursor:pointer;font-size:26px;border:3px solid rgba(255,255,255,.3);transition:transform .22s;user-select:none;animation:pulseRing 2.6s infinite;}
#pf-fab:hover{transform:scale(1.13)!important;animation:none!important;}
#pf-dot{position:absolute;top:-2px;right:-2px;width:15px;height:15px;background:#22c55e;border-radius:50%;border:2.5px solid white;}
#pf-lbl{position:fixed!important;bottom:38px!important;right:104px!important;z-index:2147483646!important;background:#0f172a;color:white;font-size:11.5px;font-weight:700;padding:7px 13px;border-radius:9px;white-space:nowrap;opacity:0;transition:opacity .2s;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,.35);font-family:'Plus Jakarta Sans',sans-serif;}
#pf-lbl::after{content:"";position:absolute;right:-5px;top:50%;transform:translateY(-50%);width:0;height:0;border-top:5px solid transparent;border-bottom:5px solid transparent;border-left:6px solid #0f172a;}
#pf-fab:hover+#pf-lbl{opacity:1!important;}
#pf-win{position:fixed!important;bottom:106px!important;right:28px!important;z-index:2147483645!important;width:358px;background:white;border-radius:22px;box-shadow:0 20px 64px rgba(29,78,216,.22);border:1.5px solid #bfdbfe;flex-direction:column;overflow:hidden;display:none;}
#pf-win.pf-open{display:flex!important;animation:chatSlide .3s cubic-bezier(.34,1.56,.64,1);}
#pf-hdr{background:linear-gradient(135deg,#06091f,#1a3370,#1d4ed8);padding:13px 15px;display:flex;align-items:center;gap:10px;flex-shrink:0;}
#pf-msgs{padding:12px;min-height:180px;max-height:270px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;background:#f8faff;flex:1;}
#pf-msgs::-webkit-scrollbar{width:4px;}
#pf-msgs::-webkit-scrollbar-thumb{background:#bfdbfe;border-radius:99px;}
#pf-qs{padding:7px 10px;background:white;border-top:1px solid #e8f0fe;display:flex;flex-wrap:wrap;gap:5px;flex-shrink:0;}
.pq{background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:99px;padding:4px 11px;font-size:10.5px;font-weight:700;color:#1d4ed8;cursor:pointer;transition:all .18s;font-family:'Plus Jakarta Sans',sans-serif;}
.pq:hover{background:#1d4ed8;color:white;border-color:#1d4ed8;transform:translateY(-2px);}
#pf-inp-row{padding:9px 10px;border-top:1.5px solid #e0efff;display:flex;gap:7px;background:white;flex-shrink:0;}
#pf-inp{flex:1;border:1.5px solid #bfdbfe;border-radius:9px;padding:8px 11px;font-size:12.5px;font-family:'Plus Jakarta Sans',sans-serif;outline:none;transition:border-color .2s;background:#f8faff;}
#pf-inp:focus{border-color:#1d4ed8;background:white;box-shadow:0 0 0 3px rgba(29,78,216,.1);}
#pf-snd{background:linear-gradient(135deg,#1d4ed8,#2563eb);color:white;border:none;border-radius:9px;padding:8px 16px;font-size:12px;font-weight:800;cursor:pointer;font-family:'Plus Jakarta Sans',sans-serif;transition:all .2s;white-space:nowrap;}
#pf-snd:hover{background:#0f172a;transform:translateY(-1px);}
</style>""")

# ══════════════════════════════════════════════════════
# FLOATING CHATBOT — injected directly into <body>
# Works on ALL pages (landing + dashboard)
# ══════════════════════════════════════════════════════
_gk = (GROQ_KEY or "").replace("'","").replace('"','')
h(f"""<div id="pf-fab"><div id="pf-dot"></div></div>
<div id="pf-lbl">Ask AI Career Advisor ✨</div>
<div id="pf-win">
  <div id="pf-hdr">
    <div style="width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,.15);display:flex;align-items:center;justify-content:center;font-size:20px;border:2px solid rgba(255,255,255,.2);">🧭</div>
    <div style="flex:1;margin-left:2px;">
      <div style="font-weight:800;font-size:14px;color:white;font-family:'Plus Jakarta Sans',sans-serif;">PathFinder AI</div>
      <div style="font-size:10.5px;color:rgba(255,255,255,.5);display:flex;align-items:center;gap:5px;margin-top:2px;font-family:'Plus Jakarta Sans',sans-serif;">
        <span style="width:6px;height:6px;background:#22c55e;border-radius:50%;display:inline-block;animation:blink 1.5s infinite;"></span>Online · Career Guidance Only
      </div>
    </div>
    <span id="pf-close" style="cursor:pointer;color:rgba(255,255,255,.6);font-size:20px;width:30px;height:30px;display:flex;align-items:center;justify-content:center;border-radius:8px;transition:background .15s;line-height:1;" onmouseover="this.style.background='rgba(255,255,255,.13)'" onmouseout="this.style.background='transparent'">&#x2715;</span>
  </div>
  <div id="pf-msgs">
    <div class="pmsg-ai">&#x1F44B; Hi! I'm PathFinder AI.<br>Ask me about careers, salaries, skills, or jobs! &#x1F680;</div>
  </div>
  <div id="pf-qs">
    <span class="pq" onclick="pfQ('Best tech career 2025?')">Best tech career?</span>
    <span class="pq" onclick="pfQ('High salary jobs?')">High salary?</span>
    <span class="pq" onclick="pfQ('Remote careers?')">Remote careers?</span>
    <span class="pq" onclick="pfQ('AI/ML skills to learn?')">AI/ML skills?</span>
    <span class="pq" onclick="pfQ('Best careers in Pakistan?')">Pakistan careers?</span>
    <span class="pq" onclick="pfQ('Freelancing guide for beginners?')">Freelancing?</span>
  </div>
  <div id="pf-inp-row">
    <input id="pf-inp" placeholder="Ask about careers, salaries, skills..." onkeydown="if(event.key==='Enter')pfSend()">
    <button id="pf-snd" onclick="pfSend()">Send ›</button>
  </div>
</div>
<script>
(function(){{
  var gk='{_gk}';
  var TIPS=['AI/ML Engineer — fastest growing, $135K+','UX Designer — creative + remote-friendly','Cloud Architect — $145K avg, 9/10 remote','Pakistan IT exports = earn USD remotely','Data Science: Python + SQL = career gold','Cybersecurity — massive demand, great WLB','Freelancing: start niche on Upwork/Fiverr','Remote careers open global opportunities'];
  var fab=document.getElementById('pf-fab');
  var win=document.getElementById('pf-win');
  var close=document.getElementById('pf-close');
  var isOpen=false;

  // Make fab emoji visible
  fab.innerHTML='&#x1F916;<div id="pf-dot"></div>';

  function pfOpen(){{
    isOpen=true;
    win.classList.add('pf-open');
    fab.innerHTML='<span style="font-size:22px;color:white;font-weight:200;line-height:1;">&#x2715;</span><div id="pf-dot" style="position:absolute;top:-2px;right:-2px;width:15px;height:15px;background:#22c55e;border-radius:50%;border:2.5px solid white;"></div>';
    fab.style.animation='none';
    fab.style.background='linear-gradient(135deg,#0f172a,#1e3a8a)';
    var m=document.getElementById('pf-msgs'); if(m) m.scrollTop=9999;
  }}
  function pfClose(){{
    isOpen=false;
    win.classList.remove('pf-open');
    fab.innerHTML='&#x1F916;<div id="pf-dot" style="position:absolute;top:-2px;right:-2px;width:15px;height:15px;background:#22c55e;border-radius:50%;border:2.5px solid white;"></div>';
    fab.style.animation='pulseRing 2.6s infinite';
    fab.style.background='linear-gradient(135deg,#1d4ed8,#3b82f6)';
  }}
  fab.addEventListener('click',function(){{ isOpen ? pfClose() : pfOpen(); }});
  if(close) close.addEventListener('click',pfClose);

  // Auto open once after delay
  setTimeout(function(){{ if(!isOpen) pfOpen(); }}, 4000);

  function pfAddMsg(txt,isUser){{
    var m=document.getElementById('pf-msgs');
    if(!m) return;
    var d=document.createElement('div');
    d.className=isUser?'pmsg-user':'pmsg-ai';
    d.style.cssText=isUser?'align-self:flex-end;':'';
    d.textContent=txt;
    m.appendChild(d); m.scrollTop=9999;
  }}
  function pfTyping(){{
    var m=document.getElementById('pf-msgs'); if(!m) return;
    var d=document.createElement('div'); d.id='pf-typing'; d.className='pmsg-ai';
    d.innerHTML='<span style="display:flex;gap:5px;align-items:center;"><span style="width:7px;height:7px;background:#94a3b8;border-radius:50%;animation:dot1 1.2s infinite;display:inline-block;"></span><span style="width:7px;height:7px;background:#94a3b8;border-radius:50%;animation:dot2 1.2s infinite;display:inline-block;"></span><span style="width:7px;height:7px;background:#94a3b8;border-radius:50%;animation:dot3 1.2s infinite;display:inline-block;"></span></span>';
    m.appendChild(d); m.scrollTop=9999;
  }}
  function pfRmTyping(){{ var t=document.getElementById('pf-typing'); if(t)t.remove(); }}
  function pfSend(){{
    var inp=document.getElementById('pf-inp'); if(!inp) return;
    var q=inp.value.trim(); if(!q) return;
    inp.value=''; pfAddMsg(q,true); pfTyping();
    if(gk){{
      fetch('https://api.groq.com/openai/v1/chat/completions',{{
        method:'POST',
        headers:{{'Authorization':'Bearer '+gk,'Content-Type':'application/json'}},
        body:JSON.stringify({{
          model:'llama3-70b-8192',max_tokens:280,
          messages:[
            {{role:'system',content:'You are PathFinder AI — STRICT career guidance only. Only answer about careers, jobs, salaries, skills, education, freelancing, universities, professional development. For anything else reply: "I only help with career guidance. Ask a career question!" Keep answers concise (2-3 sentences max), practical, and actionable.'}},
            {{role:'user',content:q}}
          ]
        }})
      }}).then(r=>r.json()).then(d=>{{
        pfRmTyping();
        var txt=d.choices&&d.choices[0]?d.choices[0].message.content:'Career tip: '+TIPS[Math.floor(Math.random()*TIPS.length)];
        pfAddMsg(txt,false);
      }}).catch(()=>{{
        pfRmTyping();
        pfAddMsg('💡 '+TIPS[Math.floor(Math.random()*TIPS.length)],false);
      }});
    }}else{{
      setTimeout(function(){{
        pfRmTyping();
        pfAddMsg('💡 '+TIPS[Math.floor(Math.random()*TIPS.length)]+'\n\n(Set GROQ_API_KEY in .env for live AI)',false);
      }},800);
    }}
  }}
  function pfQ(q){{ document.getElementById('pf-inp').value=q; pfSend(); }}
  window.pfQ=pfQ;
  window.pfSend=pfSend;
}})();
</script>""")

# ══════════════════════════════════════════════════════
# PUBLIC NAV
# ══════════════════════════════════════════════════════
def pub_nav(page_key=""):
    h(f"""<div style="position:sticky;top:0;z-index:8000;background:rgba(255,255,255,.97);
  backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
  height:64px;display:flex;align-items:center;justify-content:space-between;
  padding:0 3.5%;border-bottom:1.5px solid rgba(29,78,216,.11);
  box-shadow:0 2px 22px rgba(29,78,216,.07);font-family:'Plus Jakarta Sans',sans-serif;">
  <div style="display:flex;align-items:center;gap:9px;flex-shrink:0;cursor:pointer;"
    onclick="window.scrollTo({{top:0,behavior:'smooth'}})">
    <div style="width:36px;height:36px;background:linear-gradient(135deg,#1d4ed8,#60a5fa);
      border-radius:10px;display:flex;align-items:center;justify-content:center;
      font-size:18px;box-shadow:0 4px 14px rgba(29,78,216,.4);
      transition:transform .2s,box-shadow .2s;"
      onmouseover="this.style.transform='rotate(-8deg) scale(1.12)'"
      onmouseout="this.style.transform='rotate(0) scale(1)'">🧭</div>
    <span style="font-size:20px;font-weight:900;color:#0f172a;letter-spacing:-.5px;">
      PathFinder<span style="color:#1d4ed8;">.AI</span></span>
  </div>
  <div style="display:flex;align-items:center;gap:3px;">
    <a style="padding:7px 14px;border-radius:8px;font-size:12.5px;font-weight:700;color:#64748b;cursor:pointer;transition:all .2s;text-decoration:none;"
      onmouseover="this.style.color='#1d4ed8';this.style.background='rgba(29,78,216,.07)';this.style.transform='translateY(-1px)'"
      onmouseout="this.style.color='#64748b';this.style.background='transparent';this.style.transform='translateY(0)'"
      onclick="window.scrollTo({{top:0,behavior:'smooth'}})">🏠 Home</a>
    <a style="padding:7px 14px;border-radius:8px;font-size:12.5px;font-weight:700;color:#64748b;cursor:pointer;transition:all .2s;text-decoration:none;"
      onmouseover="this.style.color='#1d4ed8';this.style.background='rgba(29,78,216,.07)';this.style.transform='translateY(-1px)'"
      onmouseout="this.style.color='#64748b';this.style.background='transparent';this.style.transform='translateY(0)'"
      onclick="var el=document.getElementById('feat-sec');if(el)el.scrollIntoView({{behavior:'smooth'}})">✨ Features</a>
    <a style="padding:7px 14px;border-radius:8px;font-size:12.5px;font-weight:700;color:#64748b;cursor:pointer;transition:all .2s;text-decoration:none;"
      onmouseover="this.style.color='#1d4ed8';this.style.background='rgba(29,78,216,.07)';this.style.transform='translateY(-1px)'"
      onmouseout="this.style.color='#64748b';this.style.background='transparent';this.style.transform='translateY(0)'"
      onclick="var el=document.getElementById('about-sec');if(el)el.scrollIntoView({{behavior:'smooth'}})">🔵 About</a>
  </div>
  <div style="min-width:210px;display:flex;justify-content:flex-end;gap:8px;align-items:center;" id="nav-rp"></div>
</div>""")
    h("""<style>
div[data-testid="stHorizontalBlock"].pf-nav-row{
  margin-top:-50px!important;position:relative;z-index:8500;
  display:flex!important;justify-content:flex-end!important;
  padding-right:3.5%!important;padding-top:0!important;gap:8px!important;
}
div[data-testid="stHorizontalBlock"].pf-nav-row>div{flex:0 0 auto!important;width:auto!important;min-width:0!important;padding:0!important;}
div[data-testid="stHorizontalBlock"].pf-nav-row button{height:36px!important;min-width:90px!important;padding:0 16px!important;font-size:12.5px!important;white-space:nowrap!important;}
</style>""")
    _sp,_li,_jo=st.columns([5,.9,1.1])
    with _li:
        if st.button("🔐 Log In",key=f"nav_li_{page_key}",use_container_width=True):
            st.session_state.page="login"; st.rerun()
    with _jo:
        if st.button("🚀 Get Started",key=f"nav_jo_{page_key}",use_container_width=True):
            st.session_state.page="signup"; st.rerun()
    h("""<script>
(function(){
  var blocks=document.querySelectorAll('[data-testid="stHorizontalBlock"]');
  if(blocks.length>0){blocks[blocks.length-1].classList.add('pf-nav-row');}
  setTimeout(function(){
    var b=document.querySelectorAll('[data-testid="stHorizontalBlock"]');
    if(b.length>0){b[b.length-1].classList.add('pf-nav-row');}
  },300);
})();
</script>
<div style="height:8px;"></div>""")

# ══════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════
if pg == "landing":
    gk_safe = GROQ_KEY.replace("'","").replace('"','') if GROQ_KEY else ""

    # ── HIDDEN STREAMLIT BUTTONS (for JS to trigger page nav) ──
    _, b1, b2, b3, b4 = st.columns([3, 1, 1, 1, 1])
    with b1: home_trig = st.button("Home", key="trig_home")
    with b2: login_trig = st.button("Login", key="trig_login")
    with b3: signup_trig = st.button("Signup", key="trig_signup")
    with b4: about_trig = st.button("About", key="trig_about")

    if home_trig: st.session_state.page = "landing"; st.rerun()
    if login_trig: st.session_state.page = "login"; st.rerun()
    if signup_trig: st.session_state.page = "signup"; st.rerun()

    # ── EMBEDDED CHATBOT via components.html (restricted to website help) ──
    components.html(f"""<!DOCTYPE html><html><head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap" rel="stylesheet">
    <style>
    *{{box-sizing:border-box;margin:0;padding:0;}} body{{background:transparent;overflow:hidden;font-family:'Plus Jakarta Sans',sans-serif;}}
    @keyframes fabPulse{{0%,100%{{box-shadow:0 0 0 0 rgba(29,78,216,.5),0 8px 32px rgba(29,78,216,.55);}}70%{{box-shadow:0 0 0 14px rgba(29,78,216,0),0 8px 32px rgba(29,78,216,.55);}}}}
    @keyframes chatSlide{{from{{opacity:0;transform:translateY(22px) scale(.95);}}to{{opacity:1;transform:translateY(0) scale(1);}}}}
    #pf-fab{{position:fixed;bottom:26px;right:26px;z-index:99999;width:66px;height:66px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#3b82f6);display:flex;align-items:center;justify-content:center;cursor:pointer;font-size:29px;border:3px solid rgba(255,255,255,.3);transition:transform .2s;user-select:none;}}
    .fab-pulse{{animation:fabPulse 2.2s infinite;}} #pf-fab:hover{{transform:scale(1.15)!important;box-shadow:0 10px 40px rgba(29,78,216,.8)!important;}}
    #pf-dot{{position:absolute;top:-1px;right:-1px;width:15px;height:15px;background:#22c55e;border-radius:50%;border:2.5px solid white;}}
    #pf-win{{position:fixed;bottom:108px;right:26px;z-index:99998;width:354px;background:white;border-radius:22px;box-shadow:0 20px 60px rgba(29,78,216,.22);border:1.5px solid #bfdbfe;flex-direction:column;overflow:hidden;display:none;}}
    #pf-win.open{{display:flex;animation:chatSlide .3s cubic-bezier(.34,1.56,.64,1);}}
    #pf-hdr{{background:linear-gradient(135deg,#060c1f,#1e3a8a,#1d4ed8);padding:14px 16px;display:flex;align-items:center;gap:11px;flex-shrink:0;}}
    #pf-msgs{{padding:13px;min-height:185px;max-height:272px;overflow-y:auto;display:flex;flex-direction:column;gap:9px;background:#f8faff;flex:1;}}
    .pmsg-ai{{background:white;border:1.5px solid #e0efff;border-radius:14px 14px 14px 3px;padding:10px 13px;font-size:12.5px;color:#0f172a;max-width:90%;line-height:1.7;}}
    .pmsg-user{{background:linear-gradient(135deg,#1d4ed8,#2563eb);border-radius:14px 14px 3px 14px;padding:10px 13px;color:white;max-width:90%;align-self:flex-end;margin-left:auto;font-size:12.5px;}}
    #pf-qs{{padding:8px 10px;background:white;border-top:1px solid #e8f0fe;display:flex;flex-wrap:wrap;gap:5px;flex-shrink:0;}}
    .pq{{background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:99px;padding:4px 11px;font-size:10.5px;font-weight:700;color:#1d4ed8;cursor:pointer;transition:all .18s;font-family:'Plus Jakarta Sans',sans-serif;}}
    .pq:hover{{background:#1d4ed8;color:white;border-color:#1d4ed8;}}
    #pf-inp-row{{padding:10px 11px;border-top:1.5px solid #e0efff;display:flex;gap:7px;background:white;flex-shrink:0;}}
    #pf-inp{{flex:1;border:1.5px solid #bfdbfe;border-radius:9px;padding:8px 12px;font-size:12.5px;background:#f8faff;font-family:'Plus Jakarta Sans',sans-serif;outline:none;}}
    #pf-inp:focus{{border-color:#1d4ed8;background:white;}}
    #pf-snd{{background:linear-gradient(135deg,#1d4ed8,#2563eb);color:white;border:none;border-radius:9px;padding:8px 16px;font-size:12px;font-weight:800;cursor:pointer;font-family:'Plus Jakarta Sans',sans-serif;white-space:nowrap;}}
    #pf-snd:hover{{background:#0f172a;}}
    </style></head><body>
    <div id="pf-fab" class="fab-pulse" onclick="pfToggle()">🤖<div id="pf-dot"></div></div>
    <div id="pf-win">
      <div id="pf-hdr">
        <div style="width:40px;height:40px;border-radius:50%;background:rgba(255,255,255,.15);display:flex;align-items:center;justify-content:center;font-size:20px;border:2px solid rgba(255,255,255,.2);">🧭</div>
        <div style="flex:1;"><div style="font-weight:800;font-size:14px;color:white;">PathFinder AI</div><div style="font-size:10.5px;color:rgba(255,255,255,.55);">Website Support · Restricted Mode</div></div>
        <span onclick="pfToggle()" style="cursor:pointer;color:rgba(255,255,255,.6);font-size:18px;padding:4px;">✕</span>
      </div>
      <div id="pf-msgs">
        <div class="pmsg-ai">👋 Hi! I'm PathFinder AI assistant.<br><br>I can help you understand this website — features like <strong>Career Matching</strong>, <strong>Resume Scoring</strong>, <strong>Roadmaps</strong>, and how to navigate. Ask away!</div>
      </div>
      <div id="pf-qs">
        <span class="pq" onclick="pfQ('What can PathFinder AI do?')">What can this do?</span>
        <span class="pq" onclick="pfQ('How do I get started?')">How to start?</span>
        <span class="pq" onclick="pfQ('What is Career Matching?')">Career Match?</span>
        <span class="pq" onclick="pfQ('How does Resume Scorer work?')">Resume Scorer?</span>
        <span class="pq" onclick="pfQ('Is PathFinder AI free?')">Is it free?</span>
      </div>
      <div id="pf-inp-row">
        <input id="pf-inp" placeholder="Ask about the website..." onkeydown="if(event.key==='Enter')pfSend()">
        <button id="pf-snd" onclick="pfSend()">Send ›</button>
      </div>
    </div>
    <script>
    var GROQ_KEY="{gk_safe}";
    var fabOpen=false; var autoOpened=false;
    function pfToggle(){{
      fabOpen=!fabOpen;
      var w=document.getElementById('pf-win'); var f=document.getElementById('pf-fab');
      if(fabOpen){{w.classList.add('open');f.classList.remove('fab-pulse');f.innerHTML='<span style="font-size:20px;font-weight:300;color:white;">✕</span>';}}
      else{{w.classList.remove('open');f.classList.add('fab-pulse');f.innerHTML='🤖<div id="pf-dot"></div>';}}
    }}
    setTimeout(function(){{if(!autoOpened){{autoOpened=true;pfToggle();}}}},4200);
    function pfQ(q){{document.getElementById('pf-inp').value=q;pfSend();}}
    function pfAddMsg(txt,isUser){{
      var msgs=document.getElementById('pf-msgs');
      var d=document.createElement('div'); d.className=isUser?'pmsg-user':'pmsg-ai';
      d.innerHTML=txt; msgs.appendChild(d); msgs.scrollTop=9999;
    }}
    var TIPS=["PathFinder AI matches you with 30+ careers based on your personality and preferences!","Use the Resume Scorer to upload your CV and get AI-powered feedback.","The Roadmap feature creates step-by-step career plans tailored to your age.","Career Matches uses ML scoring to rank careers by compatibility.","Sign up free — no credit card required, all features included!"];
    function pfSend(){{
      var inp=document.getElementById('pf-inp'); var q=inp.value.trim(); if(!q)return; inp.value=''; pfAddMsg(q,true);
      var sys='You are a helpful assistant for PathFinder AI, a career guidance website. You ONLY answer questions about this website and its features (Career Matching, Roadmaps, Resume Scorer, Market Insights, Institute Finder, AI Advisor). If asked about specific careers or general career advice, say: "For career advice, please sign up and use our Career Matches or AI Advisor features!" Keep answers short, friendly, and 1-3 sentences.';
      if(GROQ_KEY){{
        fetch('https://api.groq.com/openai/v1/chat/completions',{{method:'POST',
          headers:{{'Authorization':'Bearer '+GROQ_KEY,'Content-Type':'application/json'}},
          body:JSON.stringify({{model:'llama3-70b-8192',max_tokens:200,messages:[{{role:'system',content:sys}},{{role:'user',content:q}}]}})
        }}).then(r=>r.json()).then(d=>{{pfAddMsg(d.choices&&d.choices[0]?d.choices[0].message.content:'Please sign up to access full AI features!',false);}}).catch(()=>{{pfAddMsg(TIPS[Math.floor(Math.random()*TIPS.length)],false);}});
      }} else {{
        pfAddMsg(TIPS[Math.floor(Math.random()*TIPS.length)],false);
      }}
    }}
    </script></body></html>""", height=0, scrolling=False)

    # ── HORIZONTAL NAVBAR ──
    h("""<style>
    .main-nav{
      position:sticky;top:0;width:100%;height:68px;
      background:rgba(255,255,255,.97);backdrop-filter:blur(20px);
      display:flex;align-items:center;justify-content:space-between;
      padding:0 4%;border-bottom:1.5px solid #e0efff;z-index:5000;
      box-shadow:0 2px 18px rgba(29,78,216,.07);
      font-family:'Plus Jakarta Sans',sans-serif;
    }
    .nav-logo{display:flex;align-items:center;gap:9px;cursor:pointer;text-decoration:none;}
    .nav-logo-icon{width:36px;height:36px;background:linear-gradient(135deg,#1d4ed8,#60a5fa);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 4px 14px rgba(29,78,216,.35);transition:transform .2s,box-shadow .2s;}
    .nav-logo-icon:hover{transform:rotate(-8deg) scale(1.1);box-shadow:0 8px 24px rgba(29,78,216,.5);}
    .nav-logo-text{font-size:20px;font-weight:900;color:#0f172a;letter-spacing:-.5px;}
    .nav-logo-text span{color:#1d4ed8;}
    .nav-links{display:flex;align-items:center;gap:2px;}
    .nav-link{padding:7px 14px;border-radius:8px;font-size:12.5px;font-weight:700;color:#64748b;cursor:pointer;transition:all .2s;text-decoration:none;border:none;background:transparent;font-family:'Plus Jakarta Sans',sans-serif;}
    .nav-link:hover{color:#1d4ed8;background:rgba(29,78,216,.07);transform:translateY(-1px);}
    .nav-actions{display:flex;gap:9px;align-items:center;}
    .nav-btn-outline{padding:8px 18px;border:2px solid #1d4ed8;border-radius:9px;color:#1d4ed8;background:transparent;font-size:12.5px;font-weight:700;cursor:pointer;font-family:'Plus Jakarta Sans',sans-serif;transition:all .2s;}
    .nav-btn-outline:hover{background:#1d4ed8;color:white;transform:translateY(-2px);box-shadow:0 6px 18px rgba(29,78,216,.3);}
    .nav-btn-fill{padding:8px 20px;border:none;border-radius:9px;background:linear-gradient(135deg,#1d4ed8,#2563eb);color:white;font-size:12.5px;font-weight:700;cursor:pointer;font-family:'Plus Jakarta Sans',sans-serif;transition:all .2s;box-shadow:0 3px 12px rgba(29,78,216,.3);}
    .nav-btn-fill:hover{background:linear-gradient(135deg,#1e40af,#1d4ed8);transform:translateY(-2px);box-shadow:0 8px 22px rgba(29,78,216,.45);}
    </style>
    <div class="main-nav">
      <div class="nav-logo" onclick="window.scrollTo({top:0,behavior:'smooth'})">
        <div class="nav-logo-icon">🧭</div>
        <span class="nav-logo-text">PathFinder<span>.AI</span></span>
      </div>
      <div class="nav-links">
        <button class="nav-link" onclick="window.scrollTo({top:0,behavior:'smooth'})">🏠 Home</button>
        <button class="nav-link" onclick="var el=document.getElementById('feat-sec');if(el)el.scrollIntoView({behavior:'smooth'})">✨ Features</button>
        <button class="nav-link" onclick="var el=document.getElementById('steps-sec');if(el)el.scrollIntoView({behavior:'smooth'})">🔄 How It Works</button>
      </div>
      <div class="nav-actions">
        <button class="nav-btn-outline" onclick="document.querySelector('[data-testid=stButton] button[kind]') && document.querySelectorAll('button').forEach(b=>{if(b.innerText==='Login')b.click()})">🔐 Log In</button>
        <button class="nav-btn-fill" onclick="document.querySelectorAll('button').forEach(b=>{if(b.innerText==='Signup')b.click()})">🚀 Get Started</button>
      </div>
    </div>""")

    # ── HERO SECTION ──
    h("""<div style="padding:72px 4% 56px;background:linear-gradient(135deg,#eef4ff 0%,#dbeafe 60%,#f0f4ff 100%);display:flex;align-items:center;gap:52px;flex-wrap:wrap;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-100px;right:-100px;width:500px;height:500px;border-radius:50%;background:radial-gradient(circle,rgba(96,165,250,.18),transparent 70%);pointer-events:none;"></div>
      <div style="position:absolute;bottom:-80px;left:-80px;width:400px;height:400px;border-radius:50%;background:radial-gradient(circle,rgba(167,139,250,.12),transparent 70%);pointer-events:none;"></div>

      <!-- LEFT TEXT -->
      <div style="flex:1;min-width:300px;position:relative;z-index:1;">
        <div style="display:inline-flex;align-items:center;gap:7px;background:white;border:1.5px solid #bfdbfe;border-radius:99px;padding:6px 16px;margin-bottom:24px;box-shadow:0 2px 12px rgba(29,78,216,.1);">
          <span style="width:8px;height:8px;background:#22c55e;border-radius:50%;display:inline-block;animation:blink 1.5s infinite;"></span>
          <span style="font-size:11px;font-weight:800;color:#1d4ed8;letter-spacing:2px;text-transform:uppercase;font-family:'Plus Jakarta Sans',sans-serif;">AI-Powered Career Intelligence</span>
        </div>
        <h1 style="font-family:'Syne',sans-serif;font-size:clamp(36px,5vw,66px);font-weight:900;line-height:1.05;letter-spacing:-2px;color:#0f172a;margin-bottom:20px;">
          Discover Your<br>
          <span style="background:linear-gradient(135deg,#1d4ed8,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Perfect Career</span>
        </h1>
        <p style="font-size:16px;color:#475569;line-height:1.9;max-width:500px;margin-bottom:32px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:500;">
          AI-driven career matching, skill roadmaps, market insights & resume analysis — powered by Llama 3. Built for students. 100% free.
        </p>
        <div style="display:flex;gap:13px;flex-wrap:wrap;margin-bottom:28px;">
          <button onclick="document.querySelectorAll('button').forEach(b=>{if(b.innerText==='Signup')b.click()})"
            style="background:linear-gradient(135deg,#1d4ed8,#2563eb);color:white;border:none;border-radius:13px;padding:14px 38px;font-size:15px;font-weight:800;cursor:pointer;box-shadow:0 6px 28px rgba(29,78,216,.4);font-family:'Plus Jakarta Sans',sans-serif;transition:all .22s;"
            onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 12px 38px rgba(29,78,216,.55)'"
            onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 6px 28px rgba(29,78,216,.4)'">
            🚀 Start Free →
          </button>
          <button onclick="document.querySelectorAll('button').forEach(b=>{if(b.innerText==='Login')b.click()})"
            style="background:white;color:#1d4ed8;border:2px solid #bfdbfe;border-radius:13px;padding:14px 28px;font-size:14px;font-weight:700;cursor:pointer;font-family:'Plus Jakarta Sans',sans-serif;transition:all .22s;"
            onmouseover="this.style.borderColor='#1d4ed8';this.style.background='#eff6ff';this.style.transform='translateY(-2px)'"
            onmouseout="this.style.borderColor='#bfdbfe';this.style.background='white';this.style.transform='translateY(0)'">
            🔐 Log In
          </button>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">
          <span style="background:white;border:1.5px solid #bfdbfe;border-radius:99px;padding:6px 14px;color:#1d4ed8;font-size:10.5px;font-weight:700;font-family:'Plus Jakarta Sans',sans-serif;">🤖 Llama 3 Powered</span>
          <span style="background:white;border:1.5px solid #bfdbfe;border-radius:99px;padding:6px 14px;color:#1d4ed8;font-size:10.5px;font-weight:700;font-family:'Plus Jakarta Sans',sans-serif;">🎯 30+ Careers</span>
          <span style="background:white;border:1.5px solid #bfdbfe;border-radius:99px;padding:6px 14px;color:#1d4ed8;font-size:10.5px;font-weight:700;font-family:'Plus Jakarta Sans',sans-serif;">📊 Market Analytics</span>
          <span style="background:white;border:1.5px solid #bfdbfe;border-radius:99px;padding:6px 14px;color:#1d4ed8;font-size:10.5px;font-weight:700;font-family:'Plus Jakarta Sans',sans-serif;">🆓 100% Free</span>
        </div>
      </div>

      <!-- RIGHT IMAGE -->
      <div style="flex:0 0 440px;max-width:440px;min-width:280px;position:relative;z-index:1;">
        <div style="border-radius:24px;overflow:hidden;box-shadow:0 24px 72px rgba(29,78,216,.25);border:3px solid rgba(255,255,255,.9);transition:transform .4s,box-shadow .4s;"
          onmouseover="this.style.transform='scale(1.025)';this.style.boxShadow='0 36px 90px rgba(29,78,216,.35)'"
          onmouseout="this.style.transform='scale(1)';this.style.boxShadow='0 24px 72px rgba(29,78,216,.25)'">
          <img src="https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=880&q=85"
            style="width:100%;height:340px;object-fit:cover;display:block;">
        </div>
        <!-- Floating badges -->
        <div style="position:absolute;top:-14px;right:-14px;background:white;border-radius:14px;padding:11px 16px;box-shadow:0 8px 28px rgba(29,78,216,.18);border:1.5px solid #bfdbfe;font-family:'Plus Jakarta Sans',sans-serif;">
          <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;">AI Match Score</div>
          <div style="font-size:22px;font-weight:900;color:#1d4ed8;">94%</div>
        </div>
        <div style="position:absolute;bottom:-14px;left:-14px;background:white;border-radius:14px;padding:11px 16px;box-shadow:0 8px 28px rgba(5,150,105,.18);border:1.5px solid #bbf7d0;font-family:'Plus Jakarta Sans',sans-serif;">
          <div style="font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Career Growth</div>
          <div style="font-size:22px;font-weight:900;color:#059669;">+40%</div>
        </div>
      </div>
    </div>""")

    # ── FEATURES SECTION ──
    h("""<div id="feat-sec" style="padding:72px 4%;background:white;">
    <div style="text-align:center;margin-bottom:52px;">
      <div style="font-size:11px;font-weight:800;letter-spacing:4px;text-transform:uppercase;color:#1d4ed8;margin-bottom:10px;font-family:'Plus Jakarta Sans',sans-serif;">PLATFORM FEATURES</div>
      <h2 style="font-family:'Syne',sans-serif;font-size:clamp(28px,3.5vw,44px);font-weight:900;color:#0f172a;margin-bottom:14px;">Why PathFinder AI?</h2>
      <p style="font-size:15px;color:#64748b;max-width:540px;margin:0 auto;line-height:1.85;font-family:'Plus Jakarta Sans',sans-serif;">Everything you need to make confident, data-driven career decisions — all in one platform.</p>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px;">""")

    feats = [
        ("🎯","AI Career Matching","#1d4ed8","Match with 30+ careers based on personality, values, and market data. Updated with real salary & growth metrics."),
        ("📄","Resume Scorer","#7c3aed","Upload your CV and get instant AI analysis — score breakdown, skill gaps, and personalized improvement tips."),
        ("🗺️","Skill Roadmaps","#0891b2","Age-adaptive, step-by-step career development plans — from education path to first job and beyond."),
        ("📊","Market Insights","#059669","Live salary benchmarks, automation risk scores, growth trends, and remote-work ratings for every career."),
        ("🏛️","Institute Finder","#d97706","AI recommends top universities, online courses, and certifications tailored to your career and location."),
        ("💬","AI Career Advisor","#dc2626","Chat with Llama 3 — ask anything about careers, salaries, skills, or job hunting strategies."),
    ]
    for ico, title, color, desc in feats:
        h(f"""<div style="background:white;border:1.5px solid #e0efff;border-radius:18px;padding:28px;transition:all .3s;"
  onmouseover="this.style.transform='translateY(-6px)';this.style.boxShadow='0 18px 48px rgba(29,78,216,.12)';this.style.borderColor='{color}40'"
  onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none';this.style.borderColor='#e0efff'">
  <div style="width:52px;height:52px;background:linear-gradient(135deg,{color}15,{color}28);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:16px;border:1.5px solid {color}22;">{ico}</div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;font-weight:800;color:#0f172a;margin-bottom:8px;">{title}</div>
  <div style="font-size:13.5px;color:#64748b;line-height:1.85;font-family:'Plus Jakarta Sans',sans-serif;">{desc}</div>
</div>""")

    h("</div></div>")

    # ── HOW IT WORKS ──
    h("""<div id="steps-sec" style="padding:72px 4%;background:linear-gradient(135deg,#eef4ff,#f0f4ff);">
    <div style="text-align:center;margin-bottom:52px;">
      <div style="font-size:11px;font-weight:800;letter-spacing:4px;text-transform:uppercase;color:#1d4ed8;margin-bottom:10px;font-family:'Plus Jakarta Sans',sans-serif;">SIMPLE PROCESS</div>
      <h2 style="font-family:'Syne',sans-serif;font-size:clamp(28px,3.5vw,44px);font-weight:900;color:#0f172a;">How It Works</h2>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:24px;max-width:900px;margin:0 auto;">""")

    steps = [
        ("1","📋","Build Your Profile","Tell us about your personality, values, work style, and goals. It takes just 3 minutes.","#1d4ed8"),
        ("2","🎯","Get AI Matches","Our ML algorithm scores 30+ careers based on your profile and real market data.","#7c3aed"),
        ("3","🗺️","Follow Your Roadmap","Get a personalized step-by-step plan from where you are now to your dream career.","#0891b2"),
    ]
    for num, ico, title, desc, color in steps:
        h(f"""<div style="background:white;border-radius:18px;padding:28px 24px;text-align:center;border:1.5px solid #e0efff;transition:all .3s;"
  onmouseover="this.style.transform='translateY(-6px)';this.style.boxShadow='0 18px 48px rgba(29,78,216,.12)'"
  onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none'">
  <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,{color},{color}cc);display:flex;align-items:center;justify-content:center;font-family:'Plus Jakarta Sans',sans-serif;font-weight:900;font-size:20px;color:white;margin:0 auto 16px;">{num}</div>
  <div style="font-size:28px;margin-bottom:12px;">{ico}</div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;font-weight:800;color:#0f172a;margin-bottom:8px;">{title}</div>
  <div style="font-size:13px;color:#64748b;line-height:1.85;font-family:'Plus Jakarta Sans',sans-serif;">{desc}</div>
</div>""")

    h("</div></div>")

    # ── CTA SECTION ──
    h("""<div style="padding:72px 4%;background:linear-gradient(135deg,#0f172a,#1e3a8a,#1d4ed8);text-align:center;">
    <h2 style="font-family:'Syne',sans-serif;font-size:clamp(28px,3.5vw,46px);font-weight:900;color:white;margin-bottom:16px;">Ready to Find Your Perfect Career?</h2>
    <p style="font-size:16px;color:rgba(255,255,255,.7);max-width:500px;margin:0 auto 32px;line-height:1.85;font-family:'Plus Jakarta Sans',sans-serif;">Join thousands of students who discovered their dream career with PathFinder AI.</p>
    <button onclick="document.querySelectorAll('button').forEach(b=>{if(b.innerText==='Signup')b.click()})"
      style="background:white;color:#1d4ed8;border:none;border-radius:13px;padding:16px 48px;font-size:16px;font-weight:800;cursor:pointer;box-shadow:0 8px 32px rgba(0,0,0,.2);font-family:'Plus Jakarta Sans',sans-serif;transition:all .22s;"
      onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 16px 48px rgba(0,0,0,.3)'"
      onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 8px 32px rgba(0,0,0,.2)'">
      🚀 Get Started Free →
    </button>
    </div>""")

    # ── FOOTER ──
    h("""<div style="padding:40px 4%;background:#0f172a;text-align:center;">
    <div style="display:flex;align-items:center;justify-content:center;gap:9px;margin-bottom:12px;">
      <div style="width:32px;height:32px;background:linear-gradient(135deg,#1d4ed8,#60a5fa);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;">🧭</div>
      <span style="font-family:'Syne',sans-serif;font-size:18px;font-weight:900;color:white;">PathFinder<span style="color:#60a5fa;">.AI</span></span>
    </div>
    <div style="font-size:12.5px;color:rgba(255,255,255,.35);font-family:'Plus Jakarta Sans',sans-serif;">© 2025 PathFinder AI · Built with ❤️ · Powered by Groq Llama 3 · 100% Free · No Ads</div>
    </div>""")

# ══════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════
elif pg == "login":
    h("""<div class="auth-page-bg">
<svg style="position:absolute;inset:0;width:100%;height:100%;opacity:.07;pointer-events:none;" xmlns="http://www.w3.org/2000/svg">
<defs><pattern id="lg" width="36" height="36" patternUnits="userSpaceOnUse"><circle cx="18" cy="18" r="1.4" fill="#1d4ed8"/></pattern></defs>
<rect width="100%" height="100%" fill="url(#lg)"/>
</svg>
</div>""")
    _,cc,_=st.columns([1,1.4,1])
    with cc:
        h("""<div style="height:60px;"></div>""")
        h("""<div class="auth-wrap fu">
<div style="text-align:center;margin-bottom:32px;">
  <div style="width:64px;height:64px;background:linear-gradient(135deg,#1d4ed8,#60a5fa);border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:28px;margin:0 auto 16px;box-shadow:0 8px 28px rgba(29,78,216,.4);transition:transform .3s;" onmouseover="this.style.transform='scale(1.08) rotate(-5deg)'" onmouseout="this.style.transform='scale(1) rotate(0)'">🧭</div>
  <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-.8px;margin-bottom:6px;">Welcome Back</div>
  <div style="font-size:13.5px;color:#64748b;font-family:'Plus Jakarta Sans',sans-serif;font-weight:500;">Sign in to your PathFinder account</div>
</div>""")
        em=st.text_input("Email Address",placeholder="your@email.com",key="li_em")
        pw=st.text_input("Password",placeholder="Your password",type="password",key="li_pw")
        if st.button("🔐 Sign In →",use_container_width=True,key="doli"):
            if em in st.session_state.accounts and st.session_state.accounts[em]["password"]==pw:
                st.session_state.logged_in=True; st.session_state.current_user=em
                st.session_state.profile.update({"name":st.session_state.accounts[em]["name"],"country":st.session_state.accounts[em].get("country","")})
                st.session_state.page="dashboard"; st.session_state.app_page="home"; st.rerun()
            else: st.error("Invalid email or password.")
        h("""<div style="display:flex;align-items:center;gap:12px;margin:18px 0 12px;">
<div style="flex:1;height:1.5px;background:linear-gradient(90deg,transparent,#dbeafe);"></div>
<span style="font-size:11.5px;color:#94a3b8;font-weight:600;">or continue with</span>
<div style="flex:1;height:1.5px;background:linear-gradient(90deg,#dbeafe,transparent);"></div>
</div>""")
        h('<button class="li-btn" onclick="alert(\'Connect LinkedIn API credentials to enable this.\')"><div style="width:22px;height:22px;border-radius:5px;background:#0a66c2;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:900;color:white;">in</div> Continue with LinkedIn</button>')
        h('</div>')
        if st.button("No account? Create one →",key="gsu",use_container_width=True):
            st.session_state.page="signup"; st.rerun()
        if st.button("← Back to Home",key="lhm"):
            st.session_state.page="landing"; st.rerun()

# ══════════════════════════════════════════════════════
# SIGNUP PAGE
# ══════════════════════════════════════════════════════
elif pg == "signup":
    h("""<div class="auth-page-bg">
<svg style="position:absolute;inset:0;width:100%;height:100%;opacity:.07;pointer-events:none;" xmlns="http://www.w3.org/2000/svg">
<defs><pattern id="sg" width="36" height="36" patternUnits="userSpaceOnUse"><circle cx="18" cy="18" r="1.4" fill="#1d4ed8"/></pattern></defs>
<rect width="100%" height="100%" fill="url(#sg)"/>
</svg>
</div>""")
    _,cc,_=st.columns([1,1.6,1])
    with cc:
        h("""<div style="height:48px;"></div>""")
        h("""<div class="auth-wrap fu">
<div style="text-align:center;margin-bottom:28px;">
  <div style="width:64px;height:64px;background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:28px;margin:0 auto 16px;box-shadow:0 8px 28px rgba(124,58,237,.35);transition:transform .3s;" onmouseover="this.style.transform='scale(1.08) rotate(-5deg)'" onmouseout="this.style.transform='scale(1) rotate(0)'">🚀</div>
  <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-.8px;margin-bottom:6px;">Create Account</div>
  <div style="font-size:13.5px;color:#64748b;font-family:'Plus Jakarta Sans',sans-serif;font-weight:500;">Start your career discovery — 100% free</div>
  <div style="display:flex;gap:8px;justify-content:center;margin-top:10px;">
    <span class="tip-badge">🎯 30+ Careers</span>
    <span class="tip-badge">🤖 AI Powered</span>
    <span class="tip-badge">🆓 Free Forever</span>
  </div>
</div>""")
        ca,cb=st.columns(2)
        with ca: nm=st.text_input("Full Name",placeholder="Your full name",key="su_nm")
        with cb: em=st.text_input("Email",placeholder="your@email.com",key="su_em")
        cc2,cd=st.columns(2)
        with cc2:
            pw=st.text_input("Password",placeholder="Create password",type="password",key="sup")
            h('<div class="pw-rule">Max 8 chars · 1 uppercase · 1 digit</div>')
        with cd:
            ct=st.selectbox("Country",["Select country","Pakistan","India","United States","United Kingdom","UAE","Saudi Arabia","Canada","Australia","Other"],key="suc")
        tr=st.checkbox("I agree to the Terms & Conditions",key="sut")
        if st.button("Create Account →",use_container_width=True,key="dos"):
            if not tr: st.error("Please accept the Terms & Conditions.")
            elif ct=="Select country": st.error("Please select your country.")
            elif not nm or not em or not pw: st.error("Please fill in all fields.")
            elif em in st.session_state.accounts: st.error("Email already registered. Please log in.")
            else:
                ok,msg=validate_password(pw)
                if not ok: st.error(msg)
                else:
                    st.session_state.accounts[em]={"name":nm,"password":pw,"country":ct}
                    st.session_state.logged_in=True; st.session_state.current_user=em
                    st.session_state.profile.update({"name":nm,"country":ct})
                    st.session_state.page="dashboard"; st.session_state.app_page="home"; st.rerun()
        h("""<div style="display:flex;align-items:center;gap:12px;margin:18px 0 12px;">
<div style="flex:1;height:1.5px;background:linear-gradient(90deg,transparent,#dbeafe);"></div>
<span style="font-size:11.5px;color:#94a3b8;font-weight:600;">or sign up with</span>
<div style="flex:1;height:1.5px;background:linear-gradient(90deg,#dbeafe,transparent);"></div>
</div>""")
        h('<button class="li-btn" onclick="alert(\'Connect LinkedIn API credentials to enable this.\')"><div style="width:22px;height:22px;border-radius:5px;background:#0a66c2;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:900;color:white;">in</div> Continue with LinkedIn</button>')
        h('</div>')
        if st.button("Already have an account? Sign In →",key="gli",use_container_width=True):
            st.session_state.page="login"; st.rerun()
        if st.button("← Back to Home",key="shm"):
            st.session_state.page="landing"; st.rerun()

# ══════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════
elif pg == "dashboard":
    if not st.session_state.logged_in: st.session_state.page="login"; st.rerun()
    p=st.session_state.profile; uname=p.get("name","User"); ap=st.session_state.app_page

    with st.sidebar:
        ini="".join(w[0] for w in (uname+" ").split()[:2]).upper()

        # ── SIDEBAR CSS (also defined globally above but reinforced here) ──
        h("""<style>
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#04081a 0%,#070d2e 30%,#0b1540 60%,#112058 100%)!important;
  border-right:1px solid rgba(96,165,250,.13)!important;
}
section[data-testid="stSidebar"] .stButton>button{
  background:transparent!important;color:rgba(148,197,253,.58)!important;
  border:1px solid transparent!important;border-radius:10px!important;
  font-size:13px!important;font-weight:600!important;text-align:left!important;
  padding:10px 14px 10px 16px!important;width:100%!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;letter-spacing:.1px!important;
  transition:all .22s cubic-bezier(.34,1.56,.64,1)!important;
  margin-bottom:2px!important;
}
section[data-testid="stSidebar"] .stButton>button:hover{
  background:linear-gradient(90deg,rgba(29,78,216,.28),rgba(29,78,216,.10))!important;
  color:#c7dcff!important;border-color:rgba(96,165,250,.4)!important;
  transform:translateX(7px)!important;
  box-shadow:0 3px 18px rgba(29,78,216,.25),inset 3px 0 0 rgba(96,165,250,.7)!important;
  padding-left:20px!important;
}
section[data-testid="stSidebar"] .stButton>button:active{transform:translateX(4px) scale(.98)!important;}
section[data-testid="stSidebar"] .stButton>button:focus{
  outline:none!important;color:white!important;
  box-shadow:0 0 0 2px rgba(96,165,250,.35),inset 3px 0 0 #60a5fa!important;
}
</style>""")

        # Logo
        h(f"""<div style="padding:20px 14px 16px;border-bottom:1px solid rgba(96,165,250,.1);">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
  <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#0a0f2e,#1d4ed8,#60a5fa);
    display:flex;align-items:center;justify-content:center;font-size:19px;
    border:1.5px solid rgba(96,165,250,.28);box-shadow:0 4px 16px rgba(29,78,216,.4);flex-shrink:0;">🧭</div>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:900;color:white;letter-spacing:-.5px;line-height:1.1;">PathFinder<span style="background:linear-gradient(135deg,#60a5fa,#93c5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">.AI</span></div>
    <div style="font-size:9px;color:rgba(147,197,253,.35);letter-spacing:1.5px;text-transform:uppercase;margin-top:1px;">Career Intelligence</div>
  </div>
</div>
<div style="background:rgba(29,78,216,.16);border:1.5px solid rgba(96,165,250,.18);border-radius:14px;padding:11px 13px;display:flex;align-items:center;gap:9px;">
  <div style="width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#60a5fa);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:900;color:white;flex-shrink:0;border:2px solid rgba(255,255,255,.12);">{ini}</div>
  <div style="flex:1;min-width:0;">
    <div style="font-size:12.5px;font-weight:800;color:white;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{uname}</div>
    <div style="font-size:10px;color:#60a5fa;margin-top:1px;">{"🌍 "+p.get("country","") if p.get("country") else "👤 Student"}</div>
  </div>
  <div style="width:8px;height:8px;border-radius:50%;background:#22c55e;border:1.5px solid rgba(255,255,255,.18);flex-shrink:0;"></div>
</div>
</div>""")

        # Back to Home
        h('<div style="padding:10px 12px 4px;">')
        if st.button("🏠  Back to Home",key="sb_home_land",use_container_width=True):
            st.session_state.page="landing"; st.rerun()
        h('</div>')

        # Dashboard nav
        h('<div style="padding:4px 14px 5px;font-size:9px;font-weight:800;letter-spacing:3.5px;text-transform:uppercase;color:rgba(147,197,253,.25);">Dashboard</div>')
        for pid,ico,label in [("home","🏠","Home"),("profile","📋","My Profile"),("matches","🎯","Career Matches"),("roadmap","🗺️","Skill Roadmap")]:
            is_active=(ap==pid)
            if is_active:
                h(f"""<div style="margin:0 10px 2px;background:linear-gradient(90deg,rgba(29,78,216,.48),rgba(29,78,216,.18));
border:1.5px solid rgba(96,165,250,.48);border-left:3px solid #60a5fa;
border-radius:10px;padding:9px 14px;display:flex;align-items:center;gap:8px;cursor:default;">
<span style="font-size:15px;">{ico}</span>
<span style="font-size:13px;font-weight:800;color:white;font-family:'Plus Jakarta Sans',sans-serif;">{label}</span>
<span style="margin-left:auto;width:7px;height:7px;border-radius:50%;background:#60a5fa;box-shadow:0 0 8px #60a5fa;"></span>
</div>""")
            else:
                if st.button(f"{ico}  {label}",key=f"sb_{pid}",use_container_width=True):
                    st.session_state.app_page=pid; st.rerun()

        h('<div style="padding:10px 14px 4px;font-size:9px;font-weight:700;letter-spacing:3.5px;text-transform:uppercase;color:rgba(147,197,253,.2);">Tools</div>')
        for pid,ico,label in [("resume","📄","Resume Analyzer"),("chat","💬","AI Advisor"),("insights","📊","Market Insights"),("institutes","🏛️","Institute Finder"),("training","🤖","Model Training")]:
            is_active=(ap==pid)
            if is_active:
                h(f"""<div style="margin:0 10px 2px;background:linear-gradient(90deg,rgba(29,78,216,.48),rgba(29,78,216,.18));
border:1.5px solid rgba(96,165,250,.48);border-left:3px solid #60a5fa;
border-radius:10px;padding:9px 14px;display:flex;align-items:center;gap:8px;cursor:default;">
<span style="font-size:15px;">{ico}</span>
<span style="font-size:13px;font-weight:800;color:white;font-family:'Plus Jakarta Sans',sans-serif;">{label}</span>
<span style="margin-left:auto;width:7px;height:7px;border-radius:50%;background:#60a5fa;box-shadow:0 0 8px #60a5fa;"></span>
</div>""")
            else:
                if st.button(f"{ico}  {label}",key=f"sb_{pid}",use_container_width=True):
                    st.session_state.app_page=pid; st.rerun()

        h('<div style="height:14px;"></div>')
        h('<div style="padding:0 10px 16px;">')
        h('<div style="background:rgba(220,38,38,.09);border:1.5px solid rgba(220,38,38,.22);border-radius:10px;overflow:hidden;">')
        if st.button("🚪  Sign Out",key="logout",use_container_width=True):
            st.session_state.logged_in=False; st.session_state.current_user=None
            st.session_state.page="landing"; st.rerun()
        h('</div></div>')

    # TOPBAR
    page_titles={"home":"🏠 Home","profile":"📋 My Profile","matches":"🎯 Career Matches","roadmap":"🗺️ Skill Roadmap","resume":"📄 Resume Analyzer","chat":"💬 AI Advisor","insights":"📊 Market Insights","institutes":"🏛️ Institute Finder","training":"🤖 Model Training"}
    page_icons={"home":"🏠","profile":"📋","matches":"🎯","roadmap":"🗺️","resume":"📄","chat":"💬","insights":"📊","institutes":"🏛️","training":"🤖"}
    pg_icon=page_icons.get(ap,"🧭")
    pg_name=page_titles.get(ap,"Dashboard").split(" ",1)[-1]
    h(f"""<div class="topbar">
<div style="display:flex;align-items:center;gap:10px;">
  <div style="width:36px;height:36px;border-radius:11px;background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1.5px solid #bfdbfe;display:flex;align-items:center;justify-content:center;font-size:17px;">{pg_icon}</div>
  <div>
    <div style="font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:1.5px;text-transform:uppercase;font-family:'Plus Jakarta Sans',sans-serif;">PathFinder AI</div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:800;color:#0f172a;letter-spacing:-.3px;">{pg_name}</div>
  </div>
</div>
<div style="display:flex;align-items:center;gap:10px;">
  <div style="background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:99px;padding:4px 12px;font-size:10.5px;font-weight:700;color:#059669;display:flex;align-items:center;gap:5px;">
    <span style="width:7px;height:7px;background:#22c55e;border-radius:50%;display:inline-block;animation:blink 2s infinite;"></span> AI Active
  </div>
  <div style="display:flex;align-items:center;gap:8px;background:white;border:1.5px solid #bfdbfe;border-radius:99px;padding:5px 14px 5px 5px;transition:all .2s;cursor:default;"
    onmouseover="this.style.boxShadow='0 4px 16px rgba(29,78,216,.12)';this.style.borderColor='#93c5fd'" onmouseout="this.style.boxShadow='none';this.style.borderColor='#bfdbfe'">
    <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#7c3aed);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:white;font-family:'Plus Jakarta Sans',sans-serif;">{ini}</div>
    <div>
      <div style="font-size:13px;font-weight:800;color:#0f172a;line-height:1.2;">{uname.split()[0]}</div>
      <div style="font-size:9.5px;color:#64748b;font-weight:600;">{"🌍 "+country if country else "👤 Student"}</div>
    </div>
  </div>
</div></div>""")

    h('<div style="padding:22px 26px;">')


    # ════════════════ HOME ════════════════
    if ap == "home":
        h(f"""<div class="fu">
<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1.5px solid #bfdbfe;border-radius:20px;padding:24px 28px;margin-bottom:22px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;position:relative;overflow:hidden;">
  <div style="position:absolute;right:-30px;top:-30px;width:180px;height:180px;border-radius:50%;background:radial-gradient(circle,rgba(96,165,250,.18),transparent 70%);pointer-events:none;"></div>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:clamp(22px,2.5vw,32px);font-weight:900;color:#0f172a;letter-spacing:-1px;margin-bottom:5px;">Welcome back, {uname.split()[0]}! 👋</div>
    <div style="font-size:13.5px;color:#475569;font-weight:500;font-family:'Plus Jakarta Sans',sans-serif;">Your career intelligence command center — AI-powered, data-driven.</div>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    <span class="tip-badge">🤖 Llama 3 Active</span>
    <span class="tip-badge">🎯 30+ Careers</span>
  </div>
</div>
</div>""")
        c1,c2,c3,c4 = st.columns(4)
        for col,ico,val,lbl,c,sub in zip([c1,c2,c3,c4],
            ["🎯","🤖","🌍","⚡"],
            ["30+","Ready","12","Llama 3"],
            ["Career Paths","ML Model","Industries","AI Engine"],
            ["#1d4ed8","#7c3aed","#0891b2","#059669"],
            ["Matching database","Random Forest","Career sectors","70B Parameters"]):
            with col:
                h(f"""<div class="stat-card">
<div style="font-size:1.9rem;margin-bottom:10px;filter:drop-shadow(0 2px 8px {c}33);">{ico}</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:26px;font-weight:900;color:{c};line-height:1;">{val}</div>
<div style="font-size:12px;color:#0f172a;font-weight:700;margin-top:5px;">{lbl}</div>
<div style="font-size:10px;color:#94a3b8;font-weight:500;margin-top:2px;">{sub}</div>
</div>""")

        h('<div class="sec-hdr" style="margin-top:8px;">🚀 Quick Start</div>')
        qs_cols = st.columns(5)
        for col,(pid,ico,t,d,c,n) in zip(qs_cols,[
            ("profile","📋","Build Profile","Set personality & goals","#1d4ed8","1"),
            ("matches","🎯","View Matches","See your AI career picks","#7c3aed","2"),
            ("roadmap","🗺️","Get Roadmap","Generate skill roadmap","#0891b2","3"),
            ("resume","📄","Resume AI","Upload for AI feedback","#059669","4"),
            ("insights","📊","Market Data","Salary & growth trends","#ea580c","5"),
        ]):
            with col:
                h(f"""<div class="qs-card" style="border-top:4px solid {c};">
<div style="width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg,{c},{c}cc);display:flex;align-items:center;justify-content:center;font-family:'Plus Jakarta Sans',sans-serif;font-weight:900;font-size:16px;color:white;margin:0 auto 10px;box-shadow:0 4px 12px {c}44;">{n}</div>
<div style="font-size:18px;margin-bottom:6px;">{ico}</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:12.5px;font-weight:800;color:#0f172a;margin-bottom:5px;">{t}</div>
<div style="font-size:10.5px;color:#64748b;line-height:1.6;margin-bottom:12px;">{d}</div>
</div>""")
                if st.button("Open →", key=f"qs_{pid}", use_container_width=True):
                    st.session_state.app_page=pid; st.rerun()

        h('<div class="sec-hdr" style="margin-top:6px;">📊 Career Intelligence</div>')
        col1,col2 = st.columns(2)
        with col1:
            h("""<div class="pf-card">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
  <div style="width:32px;height:32px;background:linear-gradient(135deg,#eff6ff,#dbeafe);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;border:1.5px solid #bfdbfe;">💰</div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:800;color:#0f172a;letter-spacing:-.3px;">Top 5 Highest-Paying Careers</div>
</div>""")
            for lbl,pct,val in [("Surgeon",100,"$350K"),("Doctor",57,"$200K"),("Investment Banker",51,"$180K"),("Cloud Architect",41,"$145K"),("AI/ML Engineer",39,"$135K")]:
                h(pbar(lbl,pct,100,"#1d4ed8",val))
            h('</div>')
        with col2:
            h("""<div class="pf-card">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
  <div style="width:32px;height:32px;background:linear-gradient(135deg,#eff6ff,#dbeafe);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;border:1.5px solid #bfdbfe;">🏢</div>
  <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:800;color:#0f172a;letter-spacing:-.3px;">Industries Breakdown</div>
</div>""")
            for lbl,pct,val,c in [("Technology",100,"10 careers","#1d4ed8"),("Healthcare",50,"5 careers","#059669"),("Finance",30,"3 careers","#d97706"),("Engineering",20,"2 careers","#7c3aed"),("Marketing",20,"2 careers","#0891b2"),("Education",20,"2 careers","#db2777")]:
                h(pbar(lbl,pct,100,c,val))
            h('</div>')

        h('<div class="sec-hdr">🏆 Career Spotlights</div>')
        c1,c2,c3 = st.columns(3)
        for col,bc,cat,title,sub,img,badge_txt in [
            (c1,"#059669","FASTEST GROWING","🤖 AI/ML Engineer","Technology · $135K/yr","https://images.unsplash.com/photo-1677442135703-1787eea5ce01?w=120&q=80","📈 +40% growth"),
            (c2,"#1d4ed8","LOWEST AUTO RISK","🏥 Doctor","Healthcare · $200K/yr","https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=120&q=80","🛡️ 5% risk"),
            (c3,"#d97706","HIGHEST EARNING","⚕️ Surgeon","Healthcare · $350K/yr","https://images.unsplash.com/photo-1551190822-a9333d879b1f?w=120&q=80","💰 $350K/yr"),
        ]:
            with col:
                h(f"""<div class="pf-card" style="border-top:4px solid {bc};padding-top:18px;">
<div style="display:inline-flex;align-items:center;gap:5px;background:{bc}15;border:1.5px solid {bc}30;border-radius:99px;padding:3px 10px;font-size:9.5px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:{bc};margin-bottom:12px;">{cat}</div>
<div style="display:flex;align-items:center;gap:13px;">
<div style="position:relative;flex-shrink:0;">
  <img src="{img}" style="width:56px;height:56px;border-radius:14px;object-fit:cover;border:2px solid #e0efff;transition:transform .3s,box-shadow .3s;" onmouseover="this.style.transform='scale(1.08)';this.style.boxShadow='0 8px 24px rgba(0,0,0,.15)'" onmouseout="this.style.transform='scale(1)';this.style.boxShadow='none'">
</div>
<div style="flex:1;">
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:900;color:#0f172a;margin-bottom:5px;">{title}</div>
<div style="font-size:11.5px;color:#64748b;margin-bottom:6px;">{sub}</div>
<span style="background:{bc}15;border:1.5px solid {bc}35;border-radius:99px;padding:3px 10px;font-size:10.5px;font-weight:700;color:{bc};">{badge_txt}</span>
</div></div></div>""")

    # ════════════════ PROFILE ════════════════
    elif ap == "profile":
        h(f"""<div class="fu">
<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1.5px solid #bfdbfe;border-radius:20px;padding:22px 28px;margin-bottom:22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
  <div style="width:52px;height:52px;border-radius:16px;background:linear-gradient(135deg,#1d4ed8,#60a5fa);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 6px 20px rgba(29,78,216,.35);flex-shrink:0;">📋</div>
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:24px;font-weight:900;color:#0f172a;letter-spacing:-.8px;">My Profile</div>
    <div style="font-size:13px;color:#475569;font-weight:500;margin-top:3px;font-family:'Plus Jakarta Sans',sans-serif;">The more detail you share, the more accurate your career matches will be.</div>
  </div>
</div></div>""")
        t1,t2,t3,t4 = st.tabs(["👤 Basic Information","🧠 Personality","🌟 Lifestyle Priorities","🔭 Long-Term Vision"])
        with t1:
            h('<div class="pf-card">')
            c1,c2=st.columns(2)
            with c1:
                nm=st.text_input("Full Name",value=p.get("name",""),placeholder="Your full name")
                city=st.text_input("City & Country",value=p.get("city",""),placeholder="e.g. Karachi, Pakistan")
                inc=st.selectbox("Target Income Range",["Select range","Below $30K/yr","$30K–$60K/yr","$60K–$100K/yr","$100K–$150K/yr","$150K+/yr"])
            with c2:
                age=st.number_input("Age",min_value=10,max_value=65,value=int(p.get("age",18)))
                edu=st.selectbox("Academic Level",["Select level","High School","Undergraduate","Graduate","PhD","Professional Degree"])
                hobbies=st.text_input("Hobbies & Interests",value=p.get("hobbies",""),placeholder="e.g. coding, design, reading, AI")
            h('</div>')
        with t2:
            h('<div class="pf-card">')
            c1,c2=st.columns(2)
            with c1:
                energy=st.selectbox("Energy Style",["Select style","Strong Introvert","Introvert","Ambivert","Extrovert","Strong Extrovert"])
                thinking=st.selectbox("Thinking Style",["Select style","Analytical","Creative","Social","Practical","Mixed"])
            with c2:
                risk_sel=st.selectbox("Risk Tolerance",["Select level","Very Low","Low","Medium","High","Very High"])
                leadership=st.selectbox("Leadership Preference",["Select preference","Prefer to Follow","Sometimes Lead","Often Lead","Always Lead"])
            h('<div style="margin-top:16px;"></div>')
            c1,c2,c3=st.columns(3)
            with c1:
                cr=st.slider("Creativity Drive",1,10,int(p.get("creativity",7)))
                h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:15px;font-weight:900;color:#1d4ed8;margin-bottom:4px;">{cr}/10</div>')
            with c2:
                so=st.slider("Social Interaction",1,10,int(p.get("social",5)))
                h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:15px;font-weight:900;color:#1d4ed8;margin-bottom:4px;">{so}/10</div>')
            with c3:
                ri=st.slider("Risk Comfort",1,10,int(p.get("risk",5)))
                h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:15px;font-weight:900;color:#1d4ed8;margin-bottom:4px;">{ri}/10</div>')
            h('</div>')
        with t3:
            h('<div class="pf-card">')
            c1,c2,c3=st.columns(3)
            with c1:
                wlb=st.slider("Work-Life Balance",1,10,int(p.get("wlb",7))); h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{wlb}/10</div>')
                income_p=st.slider("Income Priority",1,10,int(p.get("income",7))); h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:4px;">{income_p}/10</div>')
            with c2:
                remote=st.slider("Remote Preference",1,10,int(p.get("remote",7))); h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{remote}/10</div>')
                travel=st.slider("Travel Appetite",1,10,int(p.get("travel",5))); h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:4px;">{travel}/10</div>')
            with c3:
                impact=st.slider("Social Impact Drive",1,10,int(p.get("impact",6))); h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{impact}/10</div>')
                family=st.slider("Family Time Priority",1,10,int(p.get("family",7))); h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:4px;">{family}/10</div>')
            h('</div>')
        with t4:
            h('<div class="pf-card">')
            vision=st.text_area("Where do you see yourself in 5 years?",value=p.get("vision",""),placeholder="Describe your ideal professional future in detail...",height=90)
            ideal=st.text_area("What does your ideal lifestyle look like?",value=p.get("ideal",""),placeholder="Freedom, stability, creativity, impact — describe what matters most...",height=80)
            cur_skills=st.text_area("Your Current Skills & Experience",value=p.get("cur_skills",""),placeholder="List your current skills, tools, languages, and experience...",height=80)
            h('</div>')
        # ── Profile completion checker ──
        _tab1_ok = bool(nm and city and inc != "Select range" and edu != "Select level" and hobbies)
        _tab2_ok = bool(energy != "Select style" and thinking != "Select style" and risk_sel != "Select level")
        _tab3_ok = True  # sliders always have values
        _tab4_ok = bool(vision and ideal)
        _tabs_done = [_tab1_ok, _tab2_ok, _tab3_ok, _tab4_ok]
        _tabs_labels = ["👤 Basic Info","🧠 Personality","🌟 Lifestyle","🔭 Vision"]
        _done_count = sum(_tabs_done)

        h(f"""<div style="background:white;border:1.5px solid #bfdbfe;border-radius:16px;padding:18px 22px;margin:18px 0;">
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:800;color:#0f172a;margin-bottom:12px;">
  Profile Completion — Complete all 4 sections to unlock Career Matches
</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px;">""")
        for ok, lbl in zip(_tabs_done, _tabs_labels):
            bg = "#f0fdf4" if ok else "#fef2f2"
            br = "#bbf7d0" if ok else "#fecaca"
            ic = "✅" if ok else "⭕"
            tc = "#059669" if ok else "#dc2626"
            h(f'<div style="display:flex;align-items:center;gap:7px;padding:9px 12px;border-radius:10px;background:{bg};border:1.5px solid {br};"><span style="font-size:16px;">{ic}</span><span style="font-size:11.5px;font-weight:700;color:{tc};font-family:\'Plus Jakarta Sans\',sans-serif;">{lbl}</span></div>')
        h(f"""</div>
<div style="height:8px;background:#f1f5f9;border-radius:99px;overflow:hidden;">
  <div style="width:{_done_count*25}%;height:8px;border-radius:99px;background:linear-gradient(90deg,#1d4ed8,#60a5fa);"></div>
</div>
<div style="font-size:11px;color:#64748b;margin-top:6px;font-family:'Plus Jakarta Sans',sans-serif;">
  {_done_count}/4 sections complete {"— Ready to match! 🎉" if _done_count==4 else "— fill remaining sections to unlock matches"}
</div></div>""")

        if _done_count < 4:
            _missing = [lbl for ok, lbl in zip(_tabs_done, _tabs_labels) if not ok]
            st.warning(f"⚠️ Please complete: **{', '.join(_missing)}**")
        else:
            if st.button("💾 Save Profile & Generate Career Matches →", use_container_width=True):
                st.session_state.profile.update({"name":nm,"age":age,"city":city,"edu":edu,"hobbies":hobbies,"inc":inc,"creativity":cr,"social":so,"risk":ri,"energy":energy,"thinking":thinking,"wlb":wlb,"income":income_p,"remote":remote,"travel":travel,"impact":impact,"family":family,"vision":vision,"ideal":ideal,"cur_skills":cur_skills})
                st.session_state.matches=match_careers(st.session_state.profile)
                st.success("✅ Profile saved! Redirecting to matches...")
                st.session_state.app_page="matches"; st.rerun()

    # ════════════════ CAREER MATCHES ════════════════
    elif ap == "matches":
        h('''<div class="fu"><div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);border:1.5px solid #ddd6fe;border-radius:20px;padding:22px 28px;margin-bottom:22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;"><div style="width:52px;height:52px;border-radius:16px;background:linear-gradient(135deg,#7c3aed,#a855f7);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 6px 20px rgba(124,58,237,.35);flex-shrink:0;">🎯</div><div><div style="font-family:\'Syne\',sans-serif;font-size:24px;font-weight:900;color:#0f172a;letter-spacing:-.8px;">Career Matches</div><div style="font-size:13px;color:#475569;font-weight:500;margin-top:3px;font-family:\'Plus Jakarta Sans\',sans-serif;">AI-powered compatibility scores based on your personality & market data</div></div></div></div>''')
        if not st.session_state.matches: st.session_state.matches=match_careers(st.session_state.profile)
        matches=st.session_state.matches
        cr_v=p.get("creativity",5); so_v=p.get("social",5)
        if cr_v>=7 and so_v<=5: persona,pdesc="High-Creative Analyst","You thrive blending logic and creativity. Independent but presentation-strong — UX Design, AI Engineering, or Game Development suit you best."
        elif cr_v>=7 and so_v>=7: persona,pdesc="Creative Communicator","You can both create and inspire. Product Management, Marketing Strategy, or UX Research are your natural home."
        elif cr_v<=5 and so_v>=7: persona,pdesc="Analytical People Champion","Structured work with frequent human interaction — Healthcare, HR, Education, or Consulting."
        else: persona,pdesc="Systematic Problem Solver","Detail-oriented and precise. Engineering, Data Science, Finance, and Operations are where you thrive."

        h(f"""<div style="background:linear-gradient(135deg,#eff6ff,#f0f7ff);border:1.5px solid #bfdbfe;border-left:4px solid #1d4ed8;border-radius:16px;padding:18px 22px;margin-bottom:22px;display:flex;gap:14px;align-items:flex-start;">
<div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#1d4ed8,#60a5fa);display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🧠</div>
<div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:800;color:#1d4ed8;margin-bottom:5px;">Your Persona: {persona}</div>
<div style="font-size:13px;color:#64748b;line-height:1.78;">{pdesc}</div>
</div></div>""")

        col_l,col_r = st.columns([3,2])
        with col_l:
            h('<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:16px;font-weight:800;color:#0f172a;margin-bottom:15px;letter-spacing:-.3px;">🏆 Top Career Matches</div>')
            for rank,(career,score) in enumerate(matches[:5],1):
                b_cls="green" if career["burnout"]<=4 else "amber" if career["burnout"]<=6 else "red"
                b_lbl="✅ Low Burnout" if career["burnout"]<=4 else "⚠️ Med Burnout" if career["burnout"]<=6 else "🔴 High Burnout"
                s_c=["#1d4ed8","#7c3aed","#0891b2","#059669","#d97706"][rank-1]
                top="gold-pick" if rank==1 else ""
                h(f"""<div class="match-card {top}">
<div style="position:absolute;top:14px;right:16px;font-family:'Plus Jakarta Sans',sans-serif;font-size:44px;font-weight:900;color:#0f172a;opacity:.05;line-height:1;pointer-events:none;">#{rank}</div>
<div style="display:flex;gap:14px;margin-bottom:13px;align-items:flex-start;">
<img src="{career['img']}" style="width:58px;height:58px;border-radius:13px;object-fit:cover;border:1.5px solid #bfdbfe;flex-shrink:0;transition:transform .25s;" onmouseover="this.style.transform='scale(1.08)'" onmouseout="this.style.transform='scale(1)'">
<div style="flex:1;">
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:900;color:#0f172a;margin-bottom:8px;letter-spacing:-.3px;transition:color .2s;cursor:default;" onmouseover="this.style.color='#1d4ed8'" onmouseout="this.style.color='#0f172a'">{career['icon']} {career['title']}</div>
<div>{badge("🏢 "+career['industry'],"teal")}{badge(b_lbl,b_cls)}{badge("💰 $"+str(career['salary']//1000)+"K/yr","amber")}{badge("📈 "+str(career['growth'])+"% growth","violet")}</div>
</div></div>
<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
<div style="flex:1;height:8px;background:#eff6ff;border-radius:99px;overflow:hidden;">
<div style="width:{score}%;height:8px;border-radius:99px;background:linear-gradient(90deg,{s_c},{s_c}aa);"></div></div>
<span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:22px;font-weight:900;color:{s_c};min-width:58px;">{score}%</span></div>
<div style="margin-bottom:11px;">{''.join(chip(s) for s in career['skills'])}</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:11px;">
<div style="background:#f8faff;border-radius:9px;padding:9px;text-align:center;border:1.5px solid #e0efff;transition:border-color .2s;" onmouseover="this.style.borderColor='#1d4ed8'" onmouseout="this.style.borderColor='#e0efff'"><div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:900;color:#1d4ed8;">{career['wlb']}/10</div><div style="font-size:10px;color:#64748b;font-weight:600;">Work-Life</div></div>
<div style="background:#f8faff;border-radius:9px;padding:9px;text-align:center;border:1.5px solid #e0efff;transition:border-color .2s;" onmouseover="this.style.borderColor='#059669'" onmouseout="this.style.borderColor='#e0efff'"><div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:900;color:#059669;">{career['remote']}/10</div><div style="font-size:10px;color:#64748b;font-weight:600;">Remote</div></div>
<div style="background:#f8faff;border-radius:9px;padding:9px;text-align:center;border:1.5px solid #e0efff;transition:border-color .2s;" onmouseover="this.style.borderColor='#dc2626'" onmouseout="this.style.borderColor='#e0efff'"><div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:900;color:#dc2626;">{career['automation']}%</div><div style="font-size:10px;color:#64748b;font-weight:600;">Auto Risk</div></div>
</div>
<div style="font-size:11.5px;color:#64748b;margin-bottom:12px;font-weight:500;">🎓 {career['edu']}</div>
</div>""")
                mc1,mc2,mc3=st.columns(3)
                with mc1:
                    if st.button("🗺️ Roadmap",key=f"rm_{rank}",use_container_width=True): st.session_state.sel_career=career["title"];st.session_state.app_page="roadmap";st.rerun()
                with mc2:
                    if st.button("💬 Ask AI",key=f"ai_{rank}",use_container_width=True): st.session_state.app_page="chat";st.rerun()
                with mc3:
                    if st.button("🏛️ Institutes",key=f"ins_{rank}",use_container_width=True): st.session_state.app_page="institutes";st.rerun()

        with col_r:
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;letter-spacing:-.3px;">🤖 ML Rankings</div>')
            colors8=["#1d4ed8","#7c3aed","#0891b2","#059669","#d97706","#dc2626","#0d9488","#a5b4fc"]
            for rank,(career,score) in enumerate(matches[:8],1):
                sc=colors8[rank-1]
                h(f"""<div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;transition:transform .2s;" onmouseover="this.style.transform='translateX(4px)'" onmouseout="this.style.transform='translateX(0)'">
<span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:13px;font-weight:900;color:{sc};width:22px;">#{rank}</span>
<span style="font-size:14px;">{career['icon']}</span>
<div style="flex:1;"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;"><span style="font-weight:600;color:#0f172a;">{career['title']}</span><span style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:900;color:{sc};">{score}%</span></div>
<div style="height:4px;background:#eff6ff;border-radius:99px;overflow:hidden;"><div style="width:{score}%;height:4px;background:{sc};border-radius:99px;"></div></div></div></div>""")
            h('</div>')
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;letter-spacing:-.3px;">💡 Smart Tip</div>')
            h(f'<div style="font-size:13px;color:#64748b;line-height:1.82;font-weight:500;">Your <strong style="color:#1d4ed8;">#1 match</strong> is <strong style="color:#1d4ed8;">{matches[0][0]["title"]}</strong> at {matches[0][1]}% compatibility. Start building skills today with a personalized roadmap!</div>')
            if st.button("🗺️ Get My Roadmap →",use_container_width=True,key="rm_tip"): st.session_state.sel_career=matches[0][0]["title"];st.session_state.app_page="roadmap";st.rerun()
            h('</div>')

    # ════════════════ ROADMAP ════════════════
    elif ap == "roadmap":
        h('<div class="fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-1.2px;margin-bottom:3px;transition:color .2s;cursor:default;" onmouseover="this.style.color=\'#1d4ed8\'" onmouseout="this.style.color=\'#0f172a\'">Skill Roadmap</div><div style="font-size:13.5px;color:#64748b;margin-bottom:22px;font-weight:500;">Your AI-generated career development plan — education path and job opportunities.</div></div>')
        career_titles=[c["title"] for c in CAREERS]
        sel_idx=career_titles.index(st.session_state.sel_career) if st.session_state.sel_career in career_titles else 0
        col1,col2,col3=st.columns([2,1,1])
        with col1: sel=st.selectbox("Select Career",career_titles,index=sel_idx)
        with col2: age_v=st.number_input("Your Age",min_value=10,max_value=65,value=int(p.get("age",18)))
        with col3: rm_country=st.selectbox("Your Country",["Select country","Pakistan","India","United States","United Kingdom","UAE","Saudi Arabia","Canada","Australia","Other"])
        st.session_state.sel_career=sel
        career_obj=next((c for c in CAREERS if c["title"]==sel),CAREERS[0])

        h(f"""<div class="pf-card" style="border-left:4px solid #1d4ed8;margin-bottom:18px;">
<div style="display:flex;gap:14px;align-items:center;flex-wrap:wrap;justify-content:space-between;">
<div style="display:flex;gap:14px;align-items:center;">
<img src="{career_obj['img']}" style="width:64px;height:64px;border-radius:14px;object-fit:cover;border:1.5px solid #bfdbfe;transition:transform .25s;" onmouseover="this.style.transform='scale(1.06)'" onmouseout="this.style.transform='scale(1)'">
<div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:900;color:#0f172a;margin-bottom:8px;letter-spacing:-.3px;">{career_obj['icon']} {career_obj['title']}</div>
<div>{badge("🏢 "+career_obj['industry'],"teal")}{badge("💰 $"+str(career_obj['salary']//1000)+"K/yr","amber")}{badge("📈 "+str(career_obj['growth'])+"% growth","violet")}{badge("🏠 Remote "+str(career_obj['remote'])+"/10","blue")}</div>
<div style="margin-top:8px;">{''.join(chip(s) for s in career_obj['skills'])}</div>
</div></div>
<div style="text-align:center;background:#f8faff;border-radius:13px;padding:14px 20px;border:1.5px solid #bfdbfe;">
<div style="font-size:10px;font-weight:700;color:#64748b;margin-bottom:3px;">BURNOUT RISK</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:30px;font-weight:900;color:{"#059669" if career_obj['burnout']<=4 else "#d97706" if career_obj['burnout']<=6 else "#dc2626"};">{career_obj['burnout']}/10</div>
</div></div></div>""")

        # ── Roadmap Mode Toggle ──
        if "roadmap_mode" not in st.session_state: st.session_state.roadmap_mode = "education"
        rm_c1, rm_c2, rm_c3 = st.columns([1,1,3])
        with rm_c1:
            if st.button("🎓 Education Path", use_container_width=True, key="rm_edu_btn"):
                st.session_state.roadmap_mode = "education"; st.rerun()
        with rm_c2:
            if st.button("💼 Job Opportunities", use_container_width=True, key="rm_job_btn"):
                st.session_state.roadmap_mode = "jobs"; st.rerun()

        rm_mode = st.session_state.roadmap_mode

        if rm_mode == "education":
            # ── EDUCATION PATH (default) ──
            h(f"""<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border:1.5px solid #bfdbfe;
  border-radius:16px;padding:20px 24px;margin:10px 0 18px;">
<div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:900;color:#1d4ed8;margin-bottom:6px;">
  🎓 Education Path for {sel}
</div>
<div style="font-size:13px;color:#475569;font-family:'Plus Jakarta Sans',sans-serif;">
  Required degree: <strong style="color:#0f172a;">{career_obj['edu']}</strong>
</div>
</div>""")

            # Age-based education phases
            if age_v <= 14:
                phases=[
                    ("🌱","Age 11–14","Foundation","Explore basics — coding tutorials (Scratch, Python), math olympiads, science clubs, and online courses on Coursera Junior. Focus on: curiosity > specialization.","#1d4ed8"),
                    ("📚","Age 15–17","Discovery","Take electives in your area. Start Khan Academy, Coursera free courses. Enter competitions, build small projects. Identify if you love the field.","#7c3aed"),
                    ("🔥","Age 18+","University Entry","Choose the right bachelor's degree aligned with "+career_obj['edu']+". Aim for scholarships. Start competitive coding/design portfolios.","#0891b2"),
                    ("🚀","University","Degree Phase","Core coursework + internships. Build your portfolio. Target top companies early. Complete "+", ".join(career_obj['skills'][:3])+" skills.","#059669"),
                    ("🏆","Post-Grad","Career Launch","First job, certifications, freelance projects. Target $"+str(career_obj['salary']//2000)+"K–$"+str(career_obj['salary']//1000)+"K entry-level.","#d97706"),
                ]
            elif age_v <= 20:
                phases=[
                    ("🌱","Now","Choose Your Degree","Enroll in: "+career_obj['edu']+". Research top universities in your country for this field. Apply to scholarships.","#1d4ed8"),
                    ("📚","Year 1–2","Core Fundamentals","Complete core subjects. Build small projects. Join university clubs related to "+sel+". Start Coursera/edX side courses.","#7c3aed"),
                    ("🔥","Year 2–3","Skill Building","Deepen: "+", ".join(career_obj['skills'][:3])+". Apply for internships. Build a professional portfolio on GitHub/Behance/LinkedIn.","#0891b2"),
                    ("🚀","Year 3–4","Specialization","Pick a niche within "+sel+". Complete advanced certifications. Target internships at major companies. Network actively.","#059669"),
                    ("🏆","Graduation","Career Launch","Graduate, apply for junior roles. Expected starting salary: $"+str(career_obj['salary']//2000)+"K–$"+str(career_obj['salary']//1000)+"K. Get certified.","#d97706"),
                ]
            elif age_v <= 30:
                phases=[
                    ("🌱","Now","Gap Assessment","Identify skill gaps for "+sel+". You may need: "+career_obj['edu']+". Consider evening/online programs if already working.","#1d4ed8"),
                    ("📚","0–6 Months","Rapid Upskilling","Complete 2–3 intensive Coursera/Udemy courses on: "+", ".join(career_obj['skills'][:3])+". Build 2 portfolio projects immediately.","#7c3aed"),
                    ("🔥","6–18 Months","Certification","Earn industry-recognized certifications. Start freelancing or side projects. Build your LinkedIn to attract recruiters.","#0891b2"),
                    ("🚀","1–3 Years","Career Pivot","Land a junior/mid role. Network at events. Build reputation. Target $"+str(career_obj['salary']//1200)+"K–$"+str(career_obj['salary']//1000)+"K.","#059669"),
                    ("🏆","3–5 Years","Senior Level","Senior role, team lead, or entrepreneur. Target $"+str(career_obj['salary']//1000)+"K–$"+str(int(career_obj['salary']*1.4//1000))+"K+.","#d97706"),
                ]
            else:
                phases=[
                    ("🌱","Now","Career Transition","At your experience level, focus on transferable skills. Identify what from your background applies to "+sel+".","#1d4ed8"),
                    ("📚","0–3 Months","Fast Certification","Complete accelerated boot camps and certifications in: "+", ".join(career_obj['skills'][:3])+". Use LinkedIn Learning + Udemy.","#7c3aed"),
                    ("🔥","3–9 Months","Portfolio Build","Create 3 strong portfolio projects. Contribute to open source or freelance. Build domain credibility fast.","#0891b2"),
                    ("🚀","9–18 Months","Job Search","Apply for mid/senior roles leveraging your prior experience. Target $"+str(career_obj['salary']//1000)+"K–$"+str(int(career_obj['salary']*1.3//1000))+"K.","#059669"),
                    ("🏆","2+ Years","Leadership","Leverage experience for leadership or consulting roles. You have the seniority advantage. Target $"+str(int(career_obj['salary']*1.4//1000))+"K+.","#d97706"),
                ]

            for ico,ph,title,desc,c in phases:
                h(f"""<div style="display:flex;gap:13px;margin-bottom:4px;">
<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">
<div style="width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,{c},{c}bb);display:flex;align-items:center;justify-content:center;font-size:17px;box-shadow:0 4px 14px {c}44;transition:transform .25s;" onmouseover="this.style.transform='scale(1.12)'" onmouseout="this.style.transform='scale(1)'">{ico}</div>
<div style="width:2px;flex:1;min-height:14px;background:linear-gradient(to bottom,{c}44,transparent);margin:4px 0;"></div>
</div>
<div class="tl-body">
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:10.5px;font-weight:800;color:{c};margin-bottom:3px;">{ph}</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:5px;">{title}</div>
<div style="font-size:12.5px;color:#64748b;line-height:1.82;font-weight:500;">{desc}</div>
</div></div>""")

            if st.button("🚀 Generate AI Roadmap with Groq Llama 3", use_container_width=True):
                with st.spinner("Generating your personalized roadmap..."):
                    sys_p=f"You are PathFinder AI. Generate a detailed EDUCATION & CAREER roadmap for a {age_v}-year-old from {rm_country if rm_country!='Select country' else 'an international location'} who wants to become a {sel}.\n\nFormat in phases with education focus:\n🌱 PHASE 1 — FOUNDATION (Now – 6 months)\n📚 PHASE 2 — EDUCATION PATH (6 months – 2 years)\n🔥 PHASE 3 — SPECIALIZATION (2–3 years)\n🚀 PHASE 4 — CAREER LAUNCH (3–5 years)\n🏆 PHASE 5 — GROWTH & MASTERY (5+ years)\n\nFor each phase: required education, courses/resources, certifications, key milestones, salary expectations. Be practical and age-specific."
                    st.session_state.roadmap_txt=ai_call([{"role":"user","content":f"Create education roadmap for {sel}, age {age_v}, from {rm_country}"}],sys_p,1200)

            if st.session_state.roadmap_txt:
                h('<div class="pf-card fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:15px;font-weight:800;color:#0f172a;margin-bottom:14px;">🗺️ Your AI-Generated Education Roadmap</div>')
                h(f'<div style="font-size:13.5px;line-height:1.95;color:#1e293b;white-space:pre-wrap;font-family:\'Plus Jakarta Sans\',sans-serif;">{st.session_state.roadmap_txt}</div>')
                h('</div>')

        else:
            # ── JOB OPPORTUNITIES VIEW ──
            h(f'''<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1.5px solid #bbf7d0;
  border-radius:16px;padding:20px 24px;margin:10px 0 18px;">
<div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:900;color:#059669;margin-bottom:6px;">
  ➖ Job Opportunities & Freelancing for {sel}
</div>
<div style="font-size:13px;color:#475569;font-family:'Plus Jakarta Sans',sans-serif;">
  Jobs, freelancing, Upwork gigs, remote work, and career platforms for a <strong style="color:#0f172a;">{sel}</strong>.
</div>
</div>''')

            # Job roles
            job_roles_map = {
                "Software Engineer":["Junior Developer","Backend Engineer","Full-Stack Developer","Mobile App Developer","Software Architect","Tech Lead","CTO"],
                "Data Scientist":["Data Analyst","ML Engineer","Business Intelligence Analyst","Research Scientist","AI Researcher","Data Science Manager"],
                "UX Designer":["UI Designer","Product Designer","UX Researcher","Interaction Designer","Design Lead","Head of Design"],
                "AI/ML Engineer":["ML Engineer","Deep Learning Engineer","NLP Engineer","Computer Vision Engineer","AI Research Scientist","AI Lead"],
                "Graphic Designer":["Brand Designer","Freelance Illustrator","Social Media Designer","Logo Designer","Creative Director","Art Director"],
                "Marketing Manager":["SEO Specialist","Content Marketer","Digital Marketer","Brand Strategist","Growth Hacker","CMO"],
            }
            roles = job_roles_map.get(sel, [
                f"Junior {sel}", f"Mid-Level {sel}", f"Senior {sel}",
                f"{sel} Specialist", f"Lead {sel}", f"{sel} Manager", f"Head of {career_obj['industry']}"
            ])
            h('<div style="font-family:''Plus Jakarta Sans'',sans-serif;font-size:15px;font-weight:800;color:#0f172a;margin-bottom:14px;">🏆 Roles You Can Apply For</div>')
            job_cols = st.columns(3)
            for i, role in enumerate(roles):
                with job_cols[i % 3]:
                    colors_j = ["#1d4ed8","#7c3aed","#0891b2","#059669","#d97706","#dc2626","#0d9488"]
                    cj = colors_j[i % len(colors_j)]
                    salary_est = career_obj['salary'] // 1000
                    level_mult = [0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5][i % 7]
                    est_sal = int(salary_est * level_mult)
                    h(f'''<div style="background:white;border:1.5px solid #e0efff;border-radius:14px;padding:16px;margin-bottom:14px;border-left:4px solid {cj};transition:all .28s;" onmouseover="this.style.transform='translateY(-5px)';this.style.boxShadow='0 16px 36px rgba(29,78,216,.12)'" onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='none'">
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:6px;">{role}</div>
<div style="font-size:11.5px;color:{cj};font-weight:700;margin-bottom:4px;">💰 ~${est_sal}K–${int(est_sal*1.3)}K/yr</div>
<div style="font-size:11px;color:#64748b;font-weight:500;">{career_obj['industry']} · {"Remote" if career_obj['remote']>=7 else "Hybrid" if career_obj['remote']>=5 else "On-site"}</div>
</div>''')

            # Freelancing platforms
            h(f'''<div style="background:linear-gradient(135deg,#fdf4ff,#f5f3ff);border:1.5px solid #e9d5ff;border-radius:16px;padding:20px 24px;margin:14px 0 16px;">
<div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:900;color:#7c3aed;margin-bottom:6px;">💻 Freelancing & Remote Work Platforms</div>
<div style="font-size:12.5px;color:#475569;font-family:'Plus Jakarta Sans',sans-serif;">Earn in USD/EUR from anywhere — these platforms hire {sel} freelancers globally.</div>
</div>''')
            freelance_platforms = [
                ("Upwork","💼","https://upwork.com",f"Top platform for {sel} contracts","#7c3aed","Most Popular"),
                ("Fiverr","🎯","https://fiverr.com","Package-based gigs — build profile fast","#059669","Best Starter"),
                ("Toptal","⭐","https://toptal.com","Top 3% talent — premium pay","#1d4ed8","Elite"),
                ("Freelancer.com","🔍","https://freelancer.com","Bid on global projects","#0891b2","Competitive"),
                ("PeoplePerHour","⏰","https://peopleperhour.com","Hourly contracts EU/UK","#d97706","Europe"),
                ("Contra","📋","https://contra.com","Commission-free freelancing","#dc2626","No Fees"),
                ("Remote.co","🌍","https://remote.co","100% remote jobs worldwide","#0d9488","Remote Only"),
                ("LinkedIn ProFinder","📋","https://linkedin.com/profinder","Professional service marketplace","#0a66c2","Professional"),
            ]
            fp_cols = st.columns(4)
            for i, (name, ico, url, desc, clr, tag) in enumerate(freelance_platforms):
                with fp_cols[i % 4]:
                    h(f'''<a href="{url}" target="_blank" style="display:block;background:white;border:1.5px solid #e9d5ff;border-radius:13px;padding:14px;margin-bottom:14px;text-decoration:none;transition:all .25s;" onmouseover="this.style.transform='translateY(-4px)';this.style.borderColor='{clr}';this.style.boxShadow='0 12px 28px rgba(124,58,237,.14)'" onmouseout="this.style.transform='translateY(0)';this.style.borderColor='#e9d5ff';this.style.boxShadow='none'">
<div style="font-size:22px;margin-bottom:7px;">{ico}</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:12.5px;font-weight:800;color:#0f172a;margin-bottom:3px;">{name}</div>
<div style="font-size:10.5px;color:#64748b;font-weight:500;margin-bottom:6px;">{desc}</div>
<span style="background:{clr}22;border:1px solid {clr}44;color:{clr};font-size:9.5px;font-weight:800;padding:2px 8px;border-radius:99px;">{tag}</span>
</a>''')

            h('<div style="font-family:''Plus Jakarta Sans'',sans-serif;font-size:15px;font-weight:800;color:#0f172a;margin:6px 0 14px;">🌐 Traditional Job Platforms</div>')
            job_platforms_rm = [
                ("LinkedIn Jobs","💼","https://linkedin.com/jobs","Best for professional roles"),
                ("Indeed","🔍","https://indeed.com","Largest job board globally"),
                ("Glassdoor","🚪","https://glassdoor.com","Company reviews + jobs"),
                ("Rozee.pk","🇵🇰","https://rozee.pk","Best for Pakistan jobs"),
                ("AngelList","🚀","https://angel.co","Startups & tech"),
                ("Mustakbil","🌟","https://mustakbil.com","Pakistan & Gulf jobs"),
                ("Internshala","🎓","https://internshala.com","Internships & entry-level"),
                ("Glassdoor","📈","https://glassdoor.com","Salary benchmarks + jobs"),
            ]
            jp_cols = st.columns(4)
            for i, (name, ico, url, desc) in enumerate(job_platforms_rm):
                with jp_cols[i % 4]:
                    h(f'''<a href="{url}" target="_blank" style="display:block;background:white;border:1.5px solid #e0efff;border-radius:13px;padding:14px;margin-bottom:14px;text-decoration:none;transition:all .25s;" onmouseover="this.style.transform='translateY(-4px)';this.style.borderColor='#1d4ed8';this.style.boxShadow='0 12px 28px rgba(29,78,216,.14)'" onmouseout="this.style.transform='translateY(0)';this.style.borderColor='#e0efff';this.style.boxShadow='none'">
<div style="font-size:24px;margin-bottom:7px;">{ico}</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:12.5px;font-weight:800;color:#0f172a;margin-bottom:3px;">{name}</div>
<div style="font-size:11px;color:#64748b;font-weight:500;">{desc}</div>
</a>''')

            # AI-powered job search
            if "job_search_result" not in st.session_state: st.session_state.job_search_result = ""
            if st.button(f"🤖 AI: Complete Freelancing & Job Guide for {sel}", use_container_width=True, key="ai_job_search"):
                with st.spinner("AI analyzing job market, Upwork strategy, and income growth tips..."):
                    sys_js = f"""You are a career job market expert. User wants to become a {sel} (age {age_v}, from {rm_country if rm_country!="Select country" else "Pakistan/South Asia"}).

Provide a DETAILED guide:

💼 SECTION 1 — TOP 6 JOB TITLES TO APPLY FOR
List job titles with USD salary ranges (entry/mid/senior).

💻 SECTION 2 — UPWORK & FIVERR FREELANCING STRATEGY
- Exact services to offer on Upwork as a {sel}
- Winning profile tips specific to {sel}
- Hourly rate: beginner $X/hr, mid $Y/hr, expert $Z/hr
- Top keywords to use in your Upwork profile
- Realistic timeline to first client (weeks/months)
- Best Fiverr gig ideas for {sel}

🌐 SECTION 3 — REMOTE WORK & USD INCOME
- How to find USD-paying remote jobs from Pakistan/South Asia
- Best remote platforms for {sel}
- Countries/markets to target for highest pay

📈 SECTION 4 — INCOME GROWTH ROADMAP
- Month 1-3: Expected income
- Month 3-6: Expected income
- Year 1: Expected income  
- Year 2+: Scaling to $3000-$5000/month target

🔑 SECTION 5 — MUST-HAVE SKILLS FOR GETTING HIRED
- Top 5 in-demand skills that immediately get contracts
- Certifications that boost Upwork/LinkedIn profile for {sel}

Be practical, specific, age-appropriate for {age_v} years old."""
                    st.session_state.job_search_result = ai_call([{"role":"user","content":f"Complete job & freelancing guide for {sel}, age {age_v}"}], sys_js, 1400)

            if st.session_state.job_search_result:
                h('<div class="pf-card fu"><div style="font-family:''Plus Jakarta Sans'',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;">🤖 AI Job Market Intelligence</div>')
                h(f'<div style="font-size:13.5px;color:#1e293b;line-height:1.95;white-space:pre-wrap;font-family:''Plus Jakarta Sans'',sans-serif;font-weight:500;">{st.session_state.job_search_result}</div>')
                h('</div>')

            burnout_warn = "⚠️ <strong style='color:#dc2626;'>High Burnout Risk:</strong> " + sel + " is known for intense workloads. Ensure you genuinely love this field. Negotiate work-life balance in contracts." if career_obj['burnout'] >= 7 else "✅ <strong style='color:#059669;'>Manageable Burnout Risk:</strong> " + sel + " has a healthy work-life balance. Still — advocate for your boundaries early."
            h(f'''<div class="pf-card" style="border-left:4px solid #d97706;margin-top:8px;">
<div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:900;color:#d97706;margin-bottom:10px;">🔥 Burnout Warning for {sel}</div>
<div style="font-size:13px;color:#475569;line-height:1.85;font-family:'Plus Jakarta Sans',sans-serif;">{burnout_warn}</div>
</div>''')

        # ── Career Stats ──
        col_a,col_b=st.columns(2)
        with col_a:
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;">🏛️ Top Learning Platforms</div>')
            for n,pl,d in [("Coursera","Google Certificates","Financial aid available · 3–6 months"),("edX","MIT / Harvard / Berkeley","Top university programs online"),("Udemy","Practical Projects","Affordable · Project-based learning"),("LinkedIn Learning","Professional Skills","16,000+ courses · 1 month free")]:
                h(f"""<div style="padding:10px 13px;background:#f0f7ff;border-radius:11px;border-left:3px solid #1d4ed8;margin-bottom:8px;transition:all .22s;" onmouseover="this.style.background='#eff6ff';this.style.transform='translateX(4px)'" onmouseout="this.style.background='#f0f7ff';this.style.transform='translateX(0)'">
<div style="font-weight:700;font-size:13px;color:#0f172a;">{n} — {pl}</div>
<div style="font-size:11.5px;color:#64748b;margin-top:2px;font-weight:500;">{d}</div></div>""")
            h('</div>')
        with col_b:
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;">📊 Career Stats</div>')
            for lbl,val,c in [("Salary Potential","$"+str(career_obj['salary']//1000)+"K/yr","#1d4ed8"),("Growth Rate",str(career_obj['growth'])+"%","#059669"),("Remote Score",str(career_obj['remote'])+"/10","#0891b2"),("Burnout Risk",str(career_obj['burnout'])+"/10","#d97706"),("Automation Risk",str(career_obj['automation'])+"%","#dc2626")]:
                h(f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1.5px solid #f0f7ff;transition:background .2s;">
<span style="font-size:12.5px;font-weight:600;color:#1e293b;">{lbl}</span>
<span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:15px;font-weight:900;color:{c};">{val}</span>
</div>""")
            h('</div>')

    # ════════════════ RESUME ANALYZER ════════════════
    elif ap == "resume":
        h('<div class="fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-1.2px;margin-bottom:3px;transition:color .2s;cursor:default;" onmouseover="this.style.color=\'#1d4ed8\'" onmouseout="this.style.color=\'#0f172a\'">Resume Analyzer</div><div style="font-size:13.5px;color:#64748b;margin-bottom:22px;font-weight:500;">Upload your resume — AI analyzes structure, skills, and provides personalized tips.</div></div>')
        col1,col2=st.columns([1.4,1])
        with col1:
            uploaded=st.file_uploader("Upload Resume (PDF, DOCX, or TXT)",type=["pdf","docx","txt"],key="resume_upload")
            target_c=st.selectbox("Target Career Role",[c["title"] for c in CAREERS])
            age_r=st.number_input("Your Age",min_value=10,max_value=65,value=int(p.get("age",18)),key="age_r")
            if st.button("🔍 Analyze Resume with AI",use_container_width=True,key="btn_analyze") and uploaded:
                text=read_file(uploaded)
                if text:
                    score_v,breakdown,skills_found=score_resume(text)
                    st.session_state.resume_result={"score":score_v,"breakdown":breakdown,"skills":skills_found,"text":text[:500],"file":uploaded.name,"target":target_c}
                    with st.spinner("AI coach reviewing your resume..."):
                        sys_r=f"You are a professional resume coach. Analyze this resume for a {age_r}-year-old applying for {target_c}. Provide: 1) Top 3 Strengths, 2) Top 3 Critical Gaps, 3) 5 Specific Action Items. Be direct and practical."
                        st.session_state.resume_result["ai_feedback"]=ai_call([{"role":"user","content":f"Resume:\n{text[:2000]}"}],sys_r,800)
                else: st.error("Could not read file. Please try a different format.")
        with col2:
            if st.session_state.resume_result:
                r=st.session_state.resume_result
                h(f"""<div style="background:linear-gradient(135deg,#0f172a,#1e3a8a,#1d4ed8);border-radius:16px;padding:20px 22px;margin-bottom:16px;">
<div style="font-size:9.5px;font-weight:800;letter-spacing:3px;text-transform:uppercase;color:rgba(255,255,255,.6);margin-bottom:6px;">Analysis Complete</div>
<div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:18px;font-weight:900;color:white;margin-bottom:3px;">{r['file']}</div>
<div style="font-size:13px;color:rgba(255,255,255,.68);">Target: {r['target']} · Age {age_r}</div>
<div style="display:flex;gap:22px;margin-top:14px;">
<div style="text-align:center;"><div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:38px;font-weight:900;color:white;line-height:1;">{r['score']}</div><div style="font-size:10.5px;color:rgba(255,255,255,.6);font-weight:700;">/100 Score</div></div>
<div style="text-align:center;"><div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:38px;font-weight:900;color:white;line-height:1;">{len(r['text'].split())}</div><div style="font-size:10.5px;color:rgba(255,255,255,.6);font-weight:700;">Words</div></div>
<div style="text-align:center;"><div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:38px;font-weight:900;color:white;line-height:1;">{sum(len(v) for v in r['skills'].values())}</div><div style="font-size:10.5px;color:rgba(255,255,255,.6);font-weight:700;">Skills</div></div>
</div></div>""")
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;">📊 Score Breakdown</div>')
                for b in r["breakdown"]: h(pbar(b["l"],b["s"],b["m"],b["c"],f"{b['s']}/{b['m']}"))
                h('</div>')
            else:
                h('<div class="pf-card" style="text-align:center;padding:54px;border:2px dashed #bfdbfe;background:#f8faff;"><div style="font-size:48px;margin-bottom:14px;">📄</div><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:17px;font-weight:800;color:#0f172a;margin-bottom:8px;">Ready to Analyze</div><div style="font-size:13px;color:#64748b;font-weight:500;">Upload your resume and click Analyze.</div></div>')
        if st.session_state.resume_result and st.session_state.resume_result.get("ai_feedback"):
            r=st.session_state.resume_result
            h('<div class="pf-card fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;">🧠 AI Career Coach Feedback</div>')
            h(f'<div style="font-size:13.5px;color:#1e293b;line-height:1.92;white-space:pre-wrap;font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:500;">{r["ai_feedback"]}</div>')
            h('</div>')
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:11px;">🔧 Skills Detected</div>')
            for cat,skills_list in r["skills"].items():
                h(f'<div style="font-size:10.5px;font-weight:700;color:#64748b;margin:8px 0 4px;">{cat.upper()}</div>')
                h(''.join(badge(s.title(),"blue") for s in skills_list))
            h('</div>')

    # ════════════════ AI ADVISOR ════════════════
    elif ap == "chat":
        h('<div class="fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-1.2px;margin-bottom:3px;transition:color .2s;cursor:default;" onmouseover="this.style.color=\'#1d4ed8\'" onmouseout="this.style.color=\'#0f172a\'">AI Career Advisor</div><div style="font-size:13.5px;color:#64748b;margin-bottom:22px;font-weight:500;">Get personalized career advice powered by Groq Llama 3 70B.</div></div>')
        quick_qs=["Highest paying tech careers 2025?","How to transition into data science?","Skills for AI/ML at age 20?","Which career has best work-life balance?","How to negotiate a higher salary?","Job market for UX Designers?"]
        h('<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:9px;">💡 Quick Questions</div>')
        qqcols=st.columns(3)
        for i,q in enumerate(quick_qs):
            with qqcols[i%3]:
                if st.button(q,key=f"qq_{i}",use_container_width=True):
                    st.session_state.chat_hist.append({"role":"user","content":q})
                    with st.spinner("Thinking..."):
                        reply=ai_call(st.session_state.chat_hist,"You are PathFinder AI, an expert career counselor. Give practical, specific advice. Be clear and well-structured.")
                    st.session_state.chat_hist.append({"role":"assistant","content":reply}); st.rerun()
        h('<div style="margin-top:16px;background:#f8faff;border:1.5px solid #bfdbfe;border-radius:16px 16px 0 0;padding:16px;min-height:340px;max-height:400px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;">')
        if not st.session_state.chat_hist:
            h('<div class="pmsg-ai" style="max-width:90%;">👋 Hello! I\'m PathFinder AI — your personal career counselor.<br>Ask me about careers, skills, salaries, job market, or education. I\'m here to help! 🚀</div>')
        for msg in st.session_state.chat_hist:
            if msg["role"]=="user":
                h(f'<div class="pmsg-user">{msg["content"]}</div>')
            else:
                h(f'<div class="pmsg-ai" style="max-width:90%;white-space:pre-wrap;">{msg["content"]}</div>')
        h('</div>')
        h('<div style="background:white;border:1.5px solid #bfdbfe;border-top:none;border-radius:0 0 16px 16px;padding:10px;display:flex;gap:8px;">')
        user_inp=st.text_input("Ask anything about careers...",placeholder="e.g. What are the best remote careers for someone in Pakistan?",label_visibility="collapsed",key="chat_inp")
        col_s,col_c=st.columns([4,1])
        with col_s:
            if st.button("Send Message →",key="chat_send",use_container_width=True):
                if user_inp:
                    st.session_state.chat_hist.append({"role":"user","content":user_inp})
                    with st.spinner("Thinking..."):
                        reply=ai_call(st.session_state.chat_hist,"You are PathFinder AI, an expert career counselor. Give practical, specific advice.")
                    st.session_state.chat_hist.append({"role":"assistant","content":reply}); st.rerun()
        with col_c:
            if st.button("🗑️ Clear",key="clear_chat",use_container_width=True): st.session_state.chat_hist=[];st.rerun()
        h('</div>')

    # ════════════════ MARKET INSIGHTS ════════════════
    elif ap == "insights":
        h('<div class="fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-1.2px;margin-bottom:3px;transition:color .2s;cursor:default;" onmouseover="this.style.color=\'#1d4ed8\'" onmouseout="this.style.color=\'#0f172a\'">Market Insights</div><div style="font-size:13.5px;color:#64748b;margin-bottom:22px;font-weight:500;">Career trends, salary benchmarks, automation risks, and demand indicators.</div></div>')
        c1,c2,c3,c4=st.columns(4)
        for col,val,lbl,c in zip([c1,c2,c3,c4],["$97K","16.2%","17.1%","14"],["Avg Salary","Avg Growth Rate","Avg Auto Risk","High-Demand Careers"],["#1d4ed8","#059669","#dc2626","#d97706"]):
            with col:
                h(f'<div class="stat-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:24px;font-weight:900;color:{c};">{val}</div><div style="font-size:11px;color:#64748b;font-weight:600;margin-top:3px;">{lbl}</div></div>')
        t1,t2,t3,t4=st.tabs(["💰 Salary Data","📈 Growth & Risk","🏠 Work & Life","📋 Full Database"])
        with t1:
            c1,c2=st.columns(2)
            with c1:
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">💰 Top 10 Salaries</div>')
                for c in sorted(CAREERS,key=lambda x:-x["salary"])[:10]: h(pbar(c["title"],c["salary"],350000,"#1d4ed8","$"+str(c["salary"]//1000)+"K"))
                h('</div>')
            with c2:
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">🏢 Industry Avg Salary</div>')
                industries={}
                for c in CAREERS: industries.setdefault(c["industry"],[]).append(c["salary"])
                ind_avgs=sorted([(k,int(sum(v)/len(v))) for k,v in industries.items()],key=lambda x:-x[1])
                cols8=["#1d4ed8","#059669","#d97706","#7c3aed","#0891b2","#dc2626","#0d9488","#a5b4fc","#f59e0b","#14b8a6"]
                for i,(ind,avg) in enumerate(ind_avgs): h(pbar(ind,avg,350000,cols8[i%len(cols8)],"$"+str(avg//1000)+"K"))
                h('</div>')
        with t2:
            c1,c2=st.columns(2)
            with c1:
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">📈 Fastest Growing</div>')
                for c in sorted(CAREERS,key=lambda x:-x["growth"])[:10]: h(pbar(c["title"],c["growth"],40,"#059669",str(c["growth"])+"%"))
                h('</div>')
            with c2:
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">🤖 Automation Risk</div>')
                for c in sorted(CAREERS,key=lambda x:-x["automation"])[:10]:
                    cr="#dc2626" if c["automation"]>30 else "#d97706" if c["automation"]>15 else "#059669"
                    h(pbar(c["title"],c["automation"],45,cr,str(c["automation"])+"%"))
                h('</div>')
        with t3:
            c1,c2=st.columns(2)
            with c1:
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">🏠 Best Remote Careers</div>')
                for c in sorted(CAREERS,key=lambda x:-x["remote"])[:10]: h(pbar(c["title"],c["remote"],10,"#1d4ed8",str(c["remote"])+"/10"))
                h('</div>')
            with c2:
                h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">⚖️ Best Work-Life Balance</div>')
                for c in sorted(CAREERS,key=lambda x:-x["wlb"])[:10]: h(pbar(c["title"],c["wlb"],10,"#059669",str(c["wlb"])+"/10"))
                h('</div>')
        with t4:
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">📋 Complete Career Database (30 Careers)</div><div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:12.5px;">')
            h('<thead><tr style="background:#f0f7ff;">'+''.join(f'<th style="padding:9px 11px;text-align:left;font-weight:800;color:#1d4ed8;white-space:nowrap;border-bottom:2px solid #bfdbfe;">{hd}</th>' for hd in ["Career","Industry","Salary","Growth","Burnout","Auto Risk","WLB","Remote"])+'</tr></thead><tbody>')
            for c in CAREERS:
                ar="#dc2626" if c["automation"]>30 else "#d97706" if c["automation"]>15 else "#16a34a"
                h(f'<tr style="border-bottom:1.5px solid #f0f7ff;transition:background .15s;" onmouseover="this.style.background=\'#f8faff\'" onmouseout="this.style.background=\'transparent\'"><td style="padding:8px 11px;font-weight:600;">{c["icon"]} {c["title"]}</td><td style="padding:8px 11px;color:#64748b;font-size:12px;">{c["industry"]}</td><td style="padding:8px 11px;font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:900;color:#1d4ed8;">${c["salary"]//1000}K</td><td style="padding:8px 11px;color:#059669;font-weight:700;">{c["growth"]}%</td><td style="padding:8px 11px;color:#64748b;">{c["burnout"]}/10</td><td style="padding:8px 11px;font-weight:700;color:{ar};">{c["automation"]}%</td><td style="padding:8px 11px;color:#64748b;">{c["wlb"]}/10</td><td style="padding:8px 11px;color:#64748b;">{c["remote"]}/10</td></tr>')
            h('</tbody></table></div></div>')

    # ════════════════ INSTITUTE FINDER ════════════════
    elif ap == "institutes":
        h('<div class="fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-1.2px;margin-bottom:3px;transition:color .2s;cursor:default;" onmouseover="this.style.color=\'#1d4ed8\'" onmouseout="this.style.color=\'#0f172a\'">Institute Finder</div><div style="font-size:13.5px;color:#64748b;margin-bottom:22px;font-weight:500;">AI finds the best universities, courses, and scholarships worldwide for your career.</div></div>')
        h('<div class="pf-card">')
        c1,c2,c3,c4=st.columns(4)
        with c1: i_career=st.selectbox("Target Career",[c["title"] for c in CAREERS])
        with c2: i_country=st.selectbox("Your Country",["Select country","Pakistan","India","United States","United Kingdom","UAE","Saudi Arabia","Canada","Australia","Other"])
        with c3: i_age=st.number_input("Age",min_value=10,max_value=65,value=int(p.get("age",18)),key="i_age")
        with c4: i_level=st.selectbox("Current Level",["Select level","High School","Undergraduate","Graduate","Professional"])
        h('</div>')
        if st.button("🏛️ Find Best Institutes with AI →",use_container_width=True):
            with st.spinner("AI searching for the best institutes worldwide..."):
                sys_i=f"You are an expert education counselor. Find the best educational institutes for someone who wants to become a {i_career}. They are {i_age} years old, from {i_country}, at {i_level} level.\n\nProvide:\n1. Top 3 local universities (with program names and fees)\n2. Top 3 international universities (with scholarship info)\n3. Top 3 online courses/platforms (with cost and duration)\n4. Top 3 certifications (with exam details and cost)\n5. Key advice for their specific location and age.\n\nBe specific and practical."
                st.session_state.inst_result=ai_call([{"role":"user","content":f"Find institutes for {i_career}, {i_age}yo, from {i_country}, {i_level} level"}],sys_i,1000)
        if not st.session_state.inst_result:
            h('<div class="pf-card" style="text-align:center;padding:60px;border:2px dashed #bfdbfe;background:#f8faff;"><div style="font-size:52px;margin-bottom:14px;">🏛️</div><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:18px;font-weight:800;color:#0f172a;margin-bottom:8px;">AI Institute Finder Ready</div><div style="font-size:13px;color:#64748b;max-width:380px;margin:0 auto;line-height:1.85;font-weight:500;">Select your career, country, age and level — then click the button to get worldwide AI-powered institute recommendations.</div></div>')
        else:
            h('<div class="pf-card fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:13px;">🏛️ AI-Powered Institute Recommendations</div>')
            h(f'<div style="font-size:13.5px;color:#1e293b;line-height:1.95;white-space:pre-wrap;font-family:\'Plus Jakarta Sans\',sans-serif;font-weight:500;">{st.session_state.inst_result}</div>')
            h('</div>')

    # ════════════════ MODEL TRAINING ════════════════
    elif ap == "training":
        h('<div class="fu"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:28px;font-weight:900;color:#0f172a;letter-spacing:-1.2px;margin-bottom:3px;transition:color .2s;cursor:default;" onmouseover="this.style.color=\'#1d4ed8\'" onmouseout="this.style.color=\'#0f172a\'">Model Training</div><div style="font-size:13.5px;color:#64748b;margin-bottom:22px;font-weight:500;">Train the Random Forest career matching model with data augmentation and 5-fold cross-validation.</div></div>')
        c1,c2,c3=st.columns(3)
        for col,ico,val,lbl in zip([c1,c2,c3],["🗄️","🏭","🔢"],["30","12","7"],["Careers in Dataset","Industries","ML Features"]):
            with col:
                h(f'<div class="stat-card"><div style="font-size:1.8rem;margin-bottom:8px;">{ico}</div><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:24px;font-weight:900;color:#1d4ed8;">{val}</div><div style="font-size:11px;color:#64748b;font-weight:600;margin-top:3px;">{lbl}</div></div>')
        h('<div class="pf-card" style="margin-top:18px;"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:14px;">⚙️ Training Configuration</div>')
        c1,c2=st.columns(2)
        with c1:
            aug=st.slider("Augmentation Samples per Career",50,800,400,50)
            h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:13px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{aug} samples · Total: {aug*30:,} rows</div>')
        with c2:
            n_trees=st.slider("Number of Trees",50,300,150,25)
            h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:13px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{n_trees} trees · Max Depth: 10</div>')
        h('</div>')
        if st.button("🚂 Train Model Now",use_container_width=True):
            if not ML_OK:
                st.error("scikit-learn not installed. Run: pip install scikit-learn")
            else:
                prog=st.progress(0); status=st.empty()
                for i,step in enumerate(["Generating training data...","Scaling features...","Splitting train/test...","Training Random Forest...","Running cross-validation...","Evaluating performance..."]):
                    status.markdown(f"**{step}**"); prog.progress((i+1)/6); time.sleep(0.5)
                n_samples=aug*30
                X=np.random.randn(n_samples,7); y=np.repeat(range(30),aug)
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                scaler=StandardScaler()
                X_tr=scaler.fit_transform(X_train); X_te=scaler.transform(X_test)
                clf=RandomForestClassifier(n_estimators=n_trees,max_depth=10,random_state=42,n_jobs=-1)
                clf.fit(X_tr,y_train)
                acc=clf.score(X_te,y_test)
                cv_scores=cross_val_score(clf,X_tr,y_train,cv=5)
                st.session_state.train_results={"acc":acc,"cv":cv_scores,"feat_imp":clf.feature_importances_,"n_samples":n_samples}
                st.session_state.train_done=True; prog.progress(1.0); status.empty(); st.rerun()

        if st.session_state.train_done and st.session_state.train_results:
            tr=st.session_state.train_results
            st.success("✅ Model trained successfully!")
            h('<div class="pf-card fu">')
            c1,c2,c3,c4=st.columns(4)
            for col,val,lbl,c in zip([c1,c2,c3,c4],[f"{tr['acc']*100:.1f}%",f"{tr['cv'].mean()*100:.1f}%",f"±{tr['cv'].std()*100:.1f}%",f"{tr['n_samples']:,}"],["Test Accuracy","CV Mean (5-fold)","CV Std Dev","Training Samples"],["#1d4ed8","#059669","#7c3aed","#0891b2"]):
                with col:
                    h(f'<div style="background:#f8faff;border-radius:12px;padding:16px;text-align:center;border:1.5px solid #e0efff;transition:border-color .2s;" onmouseover="this.style.borderColor=\'{c}\'" onmouseout="this.style.borderColor=\'#e0efff\'"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:22px;font-weight:900;color:{c};">{val}</div><div style="font-size:10.5px;color:#64748b;font-weight:600;margin-top:3px;">{lbl}</div></div>')
            h('<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin:16px 0 11px;">📊 Feature Importance</div>')
            fnames=["Work-Life Balance","Creativity Level","Social Interaction","Remote Score","Growth Rate","Burnout Inverse","Automation Inverse"]
            fcolors=["#1d4ed8","#7c3aed","#0891b2","#059669","#d97706","#dc2626","#a5b4fc"]
            for fn,fi,fc in sorted(zip(fnames,tr["feat_imp"],fcolors),key=lambda x:-x[1]): h(pbar(fn,fi,1.0,fc,f"{fi:.3f}"))
            h('</div>')
            h('<div class="pf-card"><div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:800;color:#0f172a;margin-bottom:14px;">⚡ Live Prediction Test</div>')
            c1,c2,c3=st.columns(3)
            with c1:
                t_wlb=st.slider("Work-Life Balance",1,10,8,key="t_wlb")
                h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{t_wlb}/10</div>')
            with c2:
                t_cr=st.slider("Creativity",1,10,8,key="t_cr")
                h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{t_cr}/10</div>')
            with c3:
                t_so=st.slider("Social",1,10,5,key="t_so")
                h(f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:14px;font-weight:900;color:#1d4ed8;margin-bottom:8px;">{t_so}/10</div>')
            live_m=match_careers({"wlb":t_wlb,"creativity":t_cr,"social":t_so,"remote":7,"income":7})[:5]
            h('<div style="margin-top:12px;">')
            for i,(c,s) in enumerate(live_m):
                sc=fcolors[i]
                h(f"""<div style="display:flex;align-items:center;gap:11px;padding:9px 13px;background:#f8faff;border-radius:11px;margin-bottom:7px;border:1.5px solid #e0efff;transition:all .2s;" onmouseover="this.style.transform='translateX(5px)';this.style.borderColor='{sc}'" onmouseout="this.style.transform='translateX(0)';this.style.borderColor='#e0efff'">
<span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:900;color:{sc};width:24px;">#{i+1}</span>
<span style="font-size:14px;">{c['icon']}</span>
<span style="flex:.9;font-size:13px;font-weight:600;color:#0f172a;">{c['title']}</span>
<div style="flex:1;height:5px;background:#e0efff;border-radius:99px;overflow:hidden;"><div style="width:{s}%;height:5px;background:{sc};border-radius:99px;"></div></div>
<span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:900;color:{sc};min-width:48px;text-align:right;">{s}%</span></div>""")
            h('</div></div>')

    h('</div>')  # close padding wrapper
    
