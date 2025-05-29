import streamlit as st 
import google.generativeai as genai
import openai
import requests
import PyPDF2
import docx
import io
import json
import anthropic
import cohere

# Configure the page
st.set_page_config(
    page_title="CV Ranking System",
    page_icon="📊",
    layout="wide"
)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def call_gemini_api(api_key, model_name, cv_text, job_description):
    """Call Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = create_analysis_prompt(cv_text, job_description)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API Error: {str(e)}")

def call_openai_api(api_key, model_name, cv_text, job_description):
    """Call OpenAI API"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        prompt = create_analysis_prompt(cv_text, job_description)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert HR professional and recruiter."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API Error: {str(e)}")

def call_anthropic_api(api_key, model_name, cv_text, job_description):
    """Call Anthropic Claude API"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        prompt = create_analysis_prompt(cv_text, job_description)
        
        response = client.messages.create(
            model=model_name,
            max_tokens=3000,
            temperature=0.7,
            system="You are an expert HR professional and recruiter.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Anthropic API Error: {str(e)}")

def call_cohere_api(api_key, model_name, cv_text, job_description):
    """Call Cohere API"""
    try:
        co = cohere.Client(api_key)
        
        prompt = f"You are an expert HR professional and recruiter.\n\n{create_analysis_prompt(cv_text, job_description)}"
        
        response = co.generate(
            model=model_name,
            prompt=prompt,
            max_tokens=3000,
            temperature=0.7
        )
        return response.generations[0].text
    except Exception as e:
        raise Exception(f"Cohere API Error: {str(e)}")

def call_xai_api(api_key, model_name, cv_text, job_description):
    """Call xAI (Grok) API"""
    try:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = create_analysis_prompt(cv_text, job_description)
        
        data = {
            "messages": [
                {"role": "system", "content": "You are an expert HR professional and recruiter."},
                {"role": "user", "content": prompt}
            ],
            "model": model_name,
            "stream": False,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"xAI API Error: {str(e)}")

def call_mistral_api(api_key, model_name, cv_text, job_description):
    """Call Mistral AI API"""
    try:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = create_analysis_prompt(cv_text, job_description)
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert HR professional and recruiter."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 3000
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Mistral API Error: {str(e)}")

def call_perplexity_api(api_key, model_name, cv_text, job_description):
    """Call Perplexity API"""
    try:
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = create_analysis_prompt(cv_text, job_description)
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert HR professional and recruiter."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 3000
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Perplexity API Error: {str(e)}")

def call_together_api(api_key, model_name, cv_text, job_description):
    """Call Together AI API"""
    try:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = create_analysis_prompt(cv_text, job_description)
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert HR professional and recruiter."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 3000
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Together AI API Error: {str(e)}")

def create_analysis_prompt(cv_text, job_description):
    """Create the analysis prompt for any AI model"""
    return f"""
    As an expert HR professional and recruiter, analyze the following CV against the job description.
    
    JOB DESCRIPTION:
    {job_description}
    
    CV CONTENT:
    {cv_text}
    
    Please provide a comprehensive analysis in the following JSON format:
    {{
        "overall_score": <score out of 100>,
        "sections": [
            {{
                "section_name": "<section name>",
                "content": "<section content summary>",
                "score": <score out of 10>,
                "feedback": "<detailed feedback>",
                "improvements": ["<improvement 1>", "<improvement 2>"]
            }}
        ],
        "strengths": ["<strength 1>", "<strength 2>"],
        "weaknesses": ["<weakness 1>", "<weakness 2>"],
        "missing_skills": ["<missing skill 1>", "<missing skill 2>"],
        "overall_recommendation": "<detailed recommendation>"
    }}
    
    Break down the CV into these sections: Personal Information, Summary/Objective, Experience, Education, Skills, and Additional Sections.
    Provide specific, actionable feedback for each section.
    Make sure to return valid JSON format.
    """

def analyze_cv_with_ai(provider, api_key, model_name, cv_text, job_description):
    """Analyze CV using selected AI provider"""
    if provider == "Gemini":
        return call_gemini_api(api_key, model_name, cv_text, job_description)
    elif provider == "OpenAI":
        return call_openai_api(api_key, model_name, cv_text, job_description)
    elif provider == "Anthropic (Claude)":
        return call_anthropic_api(api_key, model_name, cv_text, job_description)
    elif provider == "Cohere":
        return call_cohere_api(api_key, model_name, cv_text, job_description)
    elif provider == "xAI (Grok)":
        return call_xai_api(api_key, model_name, cv_text, job_description)
    elif provider == "Mistral AI":
        return call_mistral_api(api_key, model_name, cv_text, job_description)
    elif provider == "Perplexity":
        return call_perplexity_api(api_key, model_name, cv_text, job_description)
    elif provider == "Together AI":
        return call_together_api(api_key, model_name, cv_text, job_description)
    else:
        raise Exception("Unsupported AI provider")

def parse_analysis_response(response_text):
    """Parse the AI response and extract JSON"""
    try:
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
    except:
        pass
    return None

def display_score_meter(score, title):
    """Display a score meter"""
    color = "red" if score < 50 else "orange" if score < 75 else "green"
    st.metric(title, f"{score}/100")
    st.progress(score / 100)

def get_model_options(provider):
    """Get available models for each provider"""
    models = {
        "Gemini": [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ],
        "OpenAI": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini"
        ],
        "Anthropic (Claude)": [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ],
        "Cohere": [
            "command-r-plus",
            "command-r",
            "command",
            "command-nightly",
            "command-light"
        ],
        "xAI (Grok)": [
            "grok-beta",
            "grok-vision-beta"
        ],
        "Mistral AI": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b"
        ],
        "Perplexity": [
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct"
        ],
        "Together AI": [
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
        ]
    }
    return models.get(provider, [])

def get_api_info(provider):
    """Get API information for each provider"""
    info = {
        "Gemini": {
            "url": "https://makersuite.google.com/app/apikey",
            "description": "Get your Gemini API key from Google AI Studio"
        },
        "OpenAI": {
            "url": "https://platform.openai.com/api-keys", 
            "description": "Get your OpenAI API key from OpenAI Platform"
        },
        "Anthropic (Claude)": {
            "url": "https://console.anthropic.com/",
            "description": "Get your Claude API key from Anthropic Console"
        },
        "Cohere": {
            "url": "https://dashboard.cohere.ai/api-keys",
            "description": "Get your Cohere API key from Cohere Dashboard"
        },
        "xAI (Grok)": {
            "url": "https://console.x.ai/",
            "description": "Get your xAI API key from xAI Console"
        },
        "Mistral AI": {
            "url": "https://console.mistral.ai/",
            "description": "Get your Mistral API key from Mistral Console"
        },
        "Perplexity": {
            "url": "https://www.perplexity.ai/settings/api",
            "description": "Get your Perplexity API key from Perplexity Settings"
        },
        "Together AI": {
            "url": "https://api.together.xyz/settings/api-keys",
            "description": "Get your Together AI API key from Together Platform"
        }
    }
    return info.get(provider, {"url": "", "description": ""})

def main():
    st.title("🎯 CV Ranking System")
    st.markdown("Upload your CV and job description to get AI-powered analysis and improvement suggestions")
    
    # Sidebar for AI configuration and inputs
    with st.sidebar:
        st.header("🤖 AI Configuration")
        
        # AI Provider Selection
        ai_provider = st.selectbox(
            "Choose AI Provider",
            ["Gemini", "OpenAI", "Anthropic (Claude)", "Cohere", "xAI (Grok)", "Mistral AI", "Perplexity", "Together AI"],
            help="Select your preferred AI provider"
        )
        
        # Model Selection based on provider
        available_models = get_model_options(ai_provider)
        selected_model = st.selectbox(
            "Choose Model",
            available_models,
            help=f"Select the {ai_provider} model to use"
        )
        
        # API Key Input
        api_key = st.text_input(
            f"Enter your {ai_provider} API Key",
            type="password",
            help=f"Your {ai_provider} API key will be used securely and not stored"
        )
        
        # API Key help text
        api_info = get_api_info(ai_provider)
        if api_info["url"]:
            st.info(f"{api_info['description']}: {api_info['url']}")
        
        st.divider()
        
        st.header("📁 Upload CV")
        uploaded_file = st.file_uploader(
            "Choose your CV file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT"
        )
        
        st.header("💼 Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            placeholder="Enter the complete job description including requirements, skills, and responsibilities..."
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 CV Content")
        cv_text = ""
        
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    cv_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    cv_text = extract_text_from_docx(uploaded_file)
                elif uploaded_file.type == "text/plain":
                    cv_text = str(uploaded_file.read(), "utf-8")
                
                st.text_area("Extracted CV Text", cv_text, height=300)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            cv_text = st.text_area(
                "Or paste your CV content here",
                height=300,
                placeholder="Paste your CV content here if you didn't upload a file..."
            )
    
    with col2:
        st.header("🔍 Analysis Configuration")
        
        # Display current configuration
        st.info(f"**AI Provider:** {ai_provider}\n**Model:** {selected_model}")
        
        if not api_key:
            st.warning(f"Please enter your {ai_provider} API key in the sidebar to proceed.")
        
        if st.button("🚀 Analyze CV", type="primary", use_container_width=True):
            if not api_key:
                st.error(f"Please provide your {ai_provider} API key")
            elif not cv_text.strip():
                st.error("Please upload a CV file or paste CV content")
            elif not job_description.strip():
                st.error("Please provide a job description")
            else:
                with st.spinner(f"Analyzing CV with {ai_provider} {selected_model}..."):
                    try:
                        analysis_response = analyze_cv_with_ai(
                            ai_provider, api_key, selected_model, cv_text, job_description
                        )
                        
                        if analysis_response:
                            parsed_analysis = parse_analysis_response(analysis_response)
                            
                            if parsed_analysis:
                                st.session_state.analysis = parsed_analysis
                                st.success("Analysis completed successfully!")
                            else:
                                st.session_state.raw_analysis = analysis_response
                                st.warning("Analysis completed, but couldn't parse structured data. Showing raw response.")
                                
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    
    # Display analysis results
    if 'analysis' in st.session_state:
        analysis = st.session_state.analysis
        
        st.divider()
        st.header("📊 Analysis Results")
        
        # Overall Score
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("Overall CV Score")
            display_score_meter(analysis['overall_score'], "Match Score")
        
        # Section Analysis
        st.subheader("📋 Section-wise Analysis")
        
        for section in analysis['sections']:
            with st.expander(f"{section['section_name']} - Score: {section['score']}/10"):
                if 'content' in section and section['content']:
                    st.write("**Content Summary:**")
                    st.text(section['content'][:200] + "..." if len(section['content']) > 200 else section['content'])
                
                st.write("**Feedback:**")
                st.write(section['feedback'])
                
                st.write("**Improvements:**")
                for improvement in section['improvements']:
                    st.write(f"• {improvement}")
        
        # Strengths and Weaknesses
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("✅ Strengths")
            for strength in analysis['strengths']:
                st.success(f"✓ {strength}")
        
        with col2:
            st.subheader("⚠️ Areas for Improvement")
            for weakness in analysis['weaknesses']:
                st.warning(f"⚡ {weakness}")
        
        # Missing Skills
        st.subheader("🔧 Missing Skills")
        if analysis['missing_skills']:
            for skill in analysis['missing_skills']:
                st.info(f"📌 {skill}")
        else:
            st.success("No major skills missing!")
        
        # Overall Recommendation
        st.subheader("💡 Overall Recommendation")
        st.write(analysis['overall_recommendation'])
    
    elif 'raw_analysis' in st.session_state:
        st.divider()
        st.header("📊 Analysis Results")
        st.write(st.session_state.raw_analysis)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>CV Ranking System - Multi-AI Provider Support</p>
        <p>Supports 8 AI Providers: Gemini, OpenAI, Claude, Cohere, xAI Grok, Mistral, Perplexity & Together AI</p>
        <p>Upload your CV and get instant feedback to improve your job application success rate!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
