# CAREER-GUIDANCE-PLATFORM


# PathFinder AI: Career Intelligence Platform
**AI-powered career guidance for students**

PathFinder AI is an intelligent career navigation system that utilizes Machine Learning and Large Language Models to assist students in career selection. The platform provides data-driven career matching, personalized roadmaps, and automated resume analysis within a unified interface.

# My Contributions
**AI Integration & Logic:** Implemented Meta Llama 3 (70B) via Groq API to generate age-adaptive skill roadmaps and educational guidance.

**Career Matching Engine:** Developed the core prediction logic to match student personality profiles with high-probability career paths.

**User Interface Design:** Built the entire web dashboard using Streamlit, focusing on a professional technical aesthetic.

Data Resiliency: Engineered a 3-gate fallback system for the Institute Finder using BeautifulSoup4 for live scraping with CSV and hardcoded backups.

# Core Features
**Career Matching:** Predicts top 5 career matches based on user input and lifestyle preferences.

**Skill Roadmap:** Generates 5-phase execution plans including education, job preparation, and freelancing.

**Resume Analyzer:** Performs text extraction and provides AI-driven coaching on strengths and gaps.

**Market Insights:** Displays salary data, growth rates, and automation risk metrics via interactive charts.

**Institute Finder:** Locates universities and scholarships through real-time web scraping and AI filtering.

# Technical Stack

**Language:** Python

**Web Framework:** Streamlit

**AI/LLM:** Groq API (Llama 3 70B)

**Machine Learning:** scikit-learn (RandomForest, StandardScaler)

**Data Processing:** Pandas, NumPy

**Extraction Tools:** PyPDF2, docx2txt

**Web Scraping:** BeautifulSoup4, Requests

**Visualization:** Plotly

# Setup and Installation
**Install required dependencies:**
pip install streamlit groq python-dotenv numpy scikit-learn pandas PyPDF2 docx2txt requests beautifulsoup4 plotly

**Configure environment variables:**
Create a .env file and add GROQ_API_KEY=your_key_here

**Launch the application:**
streamlit run main.py

# Team

**Tahira Muhammad Javed: Developer — AI Integration, Career Matching, UI Design**

**Maheen Raza: Developer — ML Pipeline, Resume Analyzer, Market Insights**
