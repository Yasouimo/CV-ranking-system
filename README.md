# 🎯 CV Ranking System

A comprehensive AI-powered CV analysis tool that evaluates resumes against job descriptions using multiple AI providers (Gemini, OpenAI GPT, and xAI Grok). Get detailed feedback, section-wise scoring, and actionable improvement suggestions.

# 🌟 Features

- Multi-AI Provider Support: Choose between Google Gemini, OpenAI GPT, or xAI Grok
- File Format Support: Upload PDF, DOCX, or TXT files
- Section-wise Analysis: Detailed breakdown of CV sections with individual scores
- Comprehensive Scoring: Overall CV match score against job requirements
- Actionable Feedback: Specific improvement suggestions for each section
- Secure API Key Handling: Users provide their own API keys (not stored)
- Modern UI: Clean, intuitive Streamlit interface
- Real-time Analysis: Instant AI-powered CV evaluation

# 🚀 Quick Start

## Prerequisites

- Python 3.7 or higher
- API key from at least one of the supported providers

## Installation
### Clone or create the project:
```bash
git clone https://github.com/Yasouimo/CV-ranking-system.git
```
### Install required packages:
```bash
pip install -r requirements.txt
```
### Run the application:
```bash
streamlit run cv_ranking_app.py
```

## 🔑 Getting API Keys
### Google Gemini
- Visit Google AI Studio
- Sign in with your Google account
- Click "Create API Key"
- Copy your API key

### OpenAI
- Visit OpenAI Platform
- Sign in to your OpenAI account
- Click "Create new secret key"
- Copy your API key

### xAI (Grok)
- Visit xAI Console
- Sign in with your account
- Navigate to API Keys section
- Generate a new API key

## 📖 Usage Guide
### Step 1: Configure AI Settings
- Select your preferred AI provider (Gemini, OpenAI, or xAI)
- Choose the specific model you want to use
- Enter your API key securely

### Step 2: Upload Your CV
- Option A: Upload a file (PDF, DOCX, or TXT)
- Option B: Copy and paste your CV content directly

### Step 3: Add Job Description
Paste the complete job description including:

- Job requirements
- Required skills
- Responsibilities
- Qualifications
