import os
import re
import docx
import pickle
import json
import PyPDF2
import streamlit as st
import hashlib
import sqlite3
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import urllib.parse
import http.client

# Run with: streamlit run app.py

# Load pre-trained models and initialize Gemini API
try:
    svc_model = pickle.load(open('models/svc_model.pkl', 'rb'))
    tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    le = pickle.load(open('models/label_encoder.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY not found in environment variables")
    st.stop()

genai.configure(api_key=gemini_api_key)


def init_database():
    """Initialize SQLite database for user authentication and session storage"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Create users table for authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create user sessions table for tracking analysis history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            resume_category TEXT,
            analysis_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()


# Authentication Functions
def hash_password(password):
    """Simple password hashing using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password, hashed_password):
    """Verify password against stored hash"""
    return hash_password(password) == hashed_password


def create_user(username, email, password):
    """Create new user account"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        hashed_password = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def authenticate_user(username, password):
    """Authenticate user login credentials"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, username, password_hash FROM users 
        WHERE username = ?
    ''', (username,))
    user = cursor.fetchone()
    conn.close()

    if user and verify_password(password, user[2]):
        return {"id": user[0], "username": user[1]}
    return None


def save_user_session(user_id, resume_category, analysis_score):
    """Save analysis session to database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_sessions (user_id, resume_category, analysis_score)
        VALUES (?, ?, ?)
    ''', (user_id, resume_category, analysis_score))
    conn.commit()
    conn.close()


def get_user_history(user_id):
    """Retrieve user's analysis history"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT resume_category, analysis_score, created_at 
        FROM user_sessions 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history


def show_login_page():
    """Display login and registration interface"""
    st.markdown("<h1 style='text-align: center;'>Job Finder & Resume Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            Developed by <a href="https://github.com/tasmiaaaa">Tasmia Hussain</a> and 
            <a href="https://github.com/Mitu-Dev">Shila Rani Deb Mitu</a>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if username and password:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.success(f"Welcome, {username}!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials!")
                    else:
                        st.error("Please fill in all fields!")

        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Register"):
                    if all([new_username, new_email, new_password, confirm_password]):
                        if new_password != confirm_password:
                            st.error("Passwords don't match!")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters!")
                        elif create_user(new_username, new_email, new_password):
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error("Username or email already exists!")
                    else:
                        st.error("Please fill in all fields!")


# Resume Processing Functions
def clean_resume(txt):
    """Clean and preprocess resume text for ML model"""
    clean_text = re.sub('http\S+\s', ' ', txt)  # Remove URLs
    clean_text = re.sub('RT|cc', ' ', clean_text)  # Remove RT and cc
    clean_text = re.sub('#\S+\s', ' ', clean_text)  # Remove hashtags
    clean_text = re.sub('@\S+', '  ', clean_text)  # Remove mentions
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ',
                        clean_text)  # Remove punctuation
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)  # Remove non-ASCII
    clean_text = re.sub('\s+', ' ', clean_text)  # Remove extra whitespace
    return clean_text


def extract_text_from_pdf(file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None


def extract_text_from_docx(file):
    """Extract text from uploaded DOCX file"""
    try:
        doc = docx.Document(file)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return None


def extract_text_from_txt(file):
    """Extract text from uploaded TXT file with encoding handling"""
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            file.seek(0)  # Reset file pointer
            text = file.read().decode('latin-1')
        except Exception as e:
            st.error(f"Error reading TXT file: {str(e)}")
            return None
    return text


def handle_file_upload(uploaded_file):
    """Process uploaded file and extract text based on file type"""
    if not uploaded_file:
        return None

    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload PDF, DOCX, or TXT files only.")
        return None


def predict_resume_category(input_resume):
    """Predict resume category using pre-trained ML model"""
    try:
        cleaned_text = clean_resume(input_resume)
        vectorized_text = tfidf.transform([cleaned_text])
        vectorized_text = vectorized_text.toarray()
        predicted_category = svc_model.predict(vectorized_text)
        predicted_category_name = le.inverse_transform(predicted_category)
        return predicted_category_name[0]
    except Exception as e:
        st.error(f"Error predicting resume category: {str(e)}")
        return "Unknown"


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse structured JSON response from Gemini API"""
    try:
        # Clean up response text
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        response_dict = json.loads(cleaned_text)

        # Return structured response with defaults
        return {
            'score': int(response_dict.get('score', 0)),
            'strengths': response_dict.get('strengths', ["Unable to determine strengths"]),
            'weaknesses': response_dict.get('weaknesses', ["Unable to determine weaknesses"]),
            'missing_skills': response_dict.get('missing_skills', []),
            'suggestions': response_dict.get('suggestions', ["No suggestions available"]),
            'formatting_feedback': response_dict.get('formatting_feedback', []),
            'summary': response_dict.get('summary', "Analysis not available")
        }
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response: {str(e)}")
        return {
            'score': 0,
            'strengths': ["Unable to analyze"],
            'weaknesses': ["Response parsing failed"],
            'missing_skills': [],
            'suggestions': ["Please try again"],
            'formatting_feedback': [],
            'summary': "Analysis failed due to parsing error"
        }


def analyze_resume_with_gemini(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Analyze resume using Gemini AI and return structured feedback"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Create analysis prompt
        prompt = f"""
        You are an expert HR professional. Analyze this resume against the job description.
        If job description is empty, analyze the resume independently.

        Job Description:
        {job_description if job_description.strip() else "No specific job description provided - general analysis"}

        Resume:
        {resume_text}

        Provide analysis as valid JSON only (no extra text or markdown):

        {{
            "score": [number 0-100],
            "strengths": ["strength 1", "strength 2", "strength 3"],
            "weaknesses": ["weakness 1", "weakness 2"],
            "missing_skills": ["skill 1", "skill 2"],
            "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
            "formatting_feedback": ["feedback 1", "feedback 2"],
            "summary": "Brief 3-4 sentence summary of the analysis"
        }}
        """

        response = model.generate_content(prompt)

        if not response or not response.text:
            st.error("No response received from AI")
            return {
                'score': 0,
                'strengths': [],
                'weaknesses': [],
                'missing_skills': [],
                'suggestions': [],
                'formatting_feedback': [],
                'summary': "No response from AI service"
            }

        return parse_gemini_response(response.text)

    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return {
            'score': 0,
            'strengths': [],
            'weaknesses': [],
            'missing_skills': [],
            'suggestions': [],
            'formatting_feedback': [],
            'summary': f"Analysis failed: {str(e)}"
        }


def search_jobs_api(keywords: str, location: str = "", results_per_page: int = 10) -> Dict:
    """
    Search for jobs using Jobicy API
    Returns job listings or error information
    """
    try:
        if not keywords.strip():
            return {"error": "Keywords cannot be empty"}

        # URL encode parameters
        keywords = urllib.parse.quote(keywords.strip())

        # Build query parameters
        query_params = f"?count={min(results_per_page, 20)}&tag={keywords}"
        if location.strip():
            location = urllib.parse.quote(location.strip())
            query_params += f"&geo={location}"

        headers = {'Accept': "application/json"}

        # Make API request
        conn = http.client.HTTPSConnection("jobicy.com")
        conn.request("GET", f"/api/v2/remote-jobs{query_params}", headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode("utf-8")
        conn.close()

        if response.status == 200:
            data = json.loads(response_data)
            # Format results for display
            formatted_jobs = []
            if 'jobs' in data and isinstance(data['jobs'], list):
                for job in data['jobs']:
                    formatted_jobs.append({
                        'title': job.get('jobTitle', 'N/A'),
                        'company': {'display_name': job.get('companyName', 'N/A')},
                        'location': {'display_name': job.get('jobGeo', 'Remote')},
                        'description': (job.get('jobDescr', 'N/A')[:200] + '...'
                                        if len(job.get('jobDescr', '')) > 200
                                        else job.get('jobDescr', 'N/A')),
                        'redirect_url': job.get('url', '#')
                    })
            return {'results': formatted_jobs}
        else:
            return {"error": f"API request failed with status {response.status}"}

    except Exception as e:
        return {"error": f"Error fetching jobs: {str(e)}"}


def display_score_with_status(score: int):
    """Display resume score with color-coded status"""
    if score >= 80:
        color = "#00FF00"
        status = "Excellent"
    elif score >= 60:
        color = "#FFA500"
        status = "Good"
    elif score >= 40:
        color = "#FFFF00"
        status = "Average"
    else:
        color = "#FF0000"
        status = "Needs Improvement"

    st.subheader(f"Resume Score: {score}/100")
    st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{status}</span>",
                unsafe_allow_html=True)


def main_app():
    """Main application interface after successful login"""
    # Header with user info
    st.markdown("<h1 style='text-align: center;'>AI Powered Job Finder & Resume Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            Developed by <a href="https://github.com/tasmiaaaa">Tasmia Hussain</a> and
            <a href="https://github.com/Mitu-Dev">Shila Rani Deb Mitu</a>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Main navigation tabs
    tab1, tab2, tab3 = st.tabs(["📄 Resume Analysis", "💼 Job Search", "📊 Analysis History"])

    with tab1:
        st.markdown("### Upload your resume and get AI-powered feedback")

        # File upload section
        uploaded_file = st.file_uploader("Choose your resume file", type=["pdf", "docx", "txt"])

        # Job description input
        job_description = st.text_area(
            "Job Description (Optional but recommended)",
            height=150,
            placeholder="Paste the job description here for targeted analysis..."
        )

        if uploaded_file is not None:
            # Extract text from uploaded file
            resume_text = handle_file_upload(uploaded_file)

            if resume_text:
                st.success("✅  Resume text extracted successfully!")

                # Show extracted text in expandable section
                with st.expander("📄 View Extracted Resume Text"):
                    st.text_area("Resume Content", resume_text, height=300, disabled=True)

                # Analysis section
                col1, col2, col3 = st.columns([2, 2, 2])

                with col1:
                    st.subheader("📊 Resume Category")
                    category = predict_resume_category(resume_text)
                    st.markdown(
                        f"**Predicted Category:** <span style='color:#39FF14; font-weight:bold'>{category}</span>",
                        unsafe_allow_html=True)

                with col2:
                    if st.button("🔍 Analyze Resume", use_container_width=True):
                        with st.spinner("Analyzing with AI... This may take a moment"):
                            analysis = analyze_resume_with_gemini(resume_text, job_description)
                            st.session_state['analysis'] = analysis
                            st.session_state['category'] = category

                with col3:
                    # Display score if analysis exists
                    if 'analysis' in st.session_state:
                        display_score_with_status(st.session_state['analysis']['score'])
                        # Save to user history
                        save_user_session(
                            st.session_state.user['id'],
                            st.session_state['category'],
                            st.session_state['analysis']['score']
                        )

                # Display detailed analysis results
                if 'analysis' in st.session_state:
                    analysis = st.session_state['analysis']

                    # Analysis results in tabs
                    analysis_tabs = st.tabs(["💪 Strengths", "⚠️ Areas to Improve", "🎯 Missing Skills",
                                             "💡 Suggestions", "📝 Formatting", "📋 Summary"])

                    with analysis_tabs[0]:
                        st.subheader("Resume Strengths")
                        for i, strength in enumerate(analysis['strengths'], 1):
                            st.write(f"✅ {strength}")

                    with analysis_tabs[1]:
                        st.subheader("Areas for Improvement")
                        for i, weakness in enumerate(analysis['weaknesses'], 1):
                            st.write(f"⚠️ {weakness}")

                    with analysis_tabs[2]:
                        st.subheader("Skills to Consider Adding")
                        if analysis['missing_skills']:
                            for i, skill in enumerate(analysis['missing_skills'], 1):
                                st.write(f"🎯 {skill}")
                        else:
                            st.info("No specific missing skills identified!")

                    with analysis_tabs[3]:
                        st.subheader("Improvement Recommendations")
                        for i, suggestion in enumerate(analysis['suggestions'], 1):
                            st.write(f"💡 {suggestion}")

                    with analysis_tabs[4]:
                        st.subheader("Formatting & Structure Feedback")
                        if analysis['formatting_feedback']:
                            for i, feedback in enumerate(analysis['formatting_feedback'], 1):
                                st.write(f"📝 {feedback}")
                        else:
                            st.info("No specific formatting issues found!")

                    with analysis_tabs[5]:
                        st.subheader("Overall Analysis Summary")
                        st.info(analysis['summary'])

    with tab2:
        st.subheader("🔍 Job Search")
        st.markdown("Find relevant job opportunities based on your skills and preferences")

        # Job search form
        with st.form("job_search_form"):
            col1, col2 = st.columns(2)

            with col1:
                search_keywords = st.text_input(
                    "Job Keywords",
                    placeholder="e.g., Software Engineer, Data Scientist, Marketing Manager"
                )

            with col2:
                search_location = st.text_input(
                    "Location (Optional)",
                    placeholder="e.g., Remote, New York, London"
                )

            num_results = st.slider("Number of Results", 5, 20, 10)
            search_button = st.form_submit_button("🔍 Search Jobs", use_container_width=True)

            if search_button and search_keywords:
                with st.spinner("Searching for jobs..."):
                    job_results = search_jobs_api(search_keywords, search_location, num_results)

                    if "error" in job_results:
                        st.error(f"Job search failed: {job_results['error']}")
                    elif not job_results.get('results'):
                        st.info("No jobs found. Try different keywords or remove location filters.")
                    else:
                        st.session_state['job_results'] = job_results['results']
                        st.success(f"Found {len(job_results['results'])} job opportunities!")

            elif search_button:
                st.warning("Please enter job keywords to search!")

        # Display job results
        if 'job_results' in st.session_state and st.session_state['job_results']:
            st.subheader("🎯 Job Opportunities")

            # Convert to DataFrame for better display
            job_data = []
            for job in st.session_state['job_results']:
                job_data.append({
                    "Job Title": job.get('title', 'N/A'),
                    "Company": job.get('company', {}).get('display_name', 'N/A'),
                    "Location": job.get('location', {}).get('display_name', 'N/A'),
                    "Description": job.get('description', 'N/A'),
                    "Apply": job.get('redirect_url', '#')
                })

            df = pd.DataFrame(job_data)

            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Apply": st.column_config.LinkColumn(
                        "Apply Now",  # This is the display text for the header
                        display_text="Apply Now",  # This is the display text for each link
                    )
                }
            )

    with tab3:
        st.subheader("📊 Your Analysis History")
        history = get_user_history(st.session_state.user['id'])

        if history:
            # Display history as a nice DataFrame
            history_df = pd.DataFrame(history, columns=['Category', 'Score', 'Date'])
            history_df['Date'] = pd.to_datetime(history_df['Date']).dt.strftime('%Y-%m-%d %H:%M')

            st.dataframe(
                history_df,
                column_config={
                    "Category": "Resume Category",
                    "Score": st.column_config.ProgressColumn(
                        "Analysis Score",
                        help="Resume score out of 100",
                        min_value=0,
                        max_value=100,
                    ),
                    "Date": "Analysis Date"
                },
                hide_index=True,
                use_container_width=True
            )

            # Display statistics if there's enough data
            if len(history) > 1:
                avg_score = sum([h[1] for h in history]) / len(history)
                best_score = max([h[1] for h in history])
                total_analyses = len(history)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyses", total_analyses)
                with col2:
                    st.metric("Average Score", f"{avg_score:.1f}")
                with col3:
                    st.metric("Best Score", f"{best_score}")
        else:
            st.info("No analysis history yet. Upload and analyze your first resume to get started!")

    # App information sections
    st.markdown("---")
    with st.expander("How to Use This App"):
        st.markdown("""
        **Step-by-step guide:**

        1. **Upload Resume**: Choose your resume file (PDF, DOCX, or TXT format)
        2. **Add Job Description**: Paste the job description for targeted analysis (optional)
        3. **Analyze**: Click the analyze button to get AI-powered feedback
        4. **Review Results**: Check your strengths, weaknesses, and improvement suggestions
        5. **Search Jobs**: Use the job search feature to find relevant opportunities
        6. **Track Progress**: Monitor your improvement over time in the history section
        """)

    with st.expander("Privacy & Security"):
        st.markdown("""
        **Your data is safe with us:**

        - **Resume Security**: We never store your uploaded resume files
        - **User Data**: Only username, email, and analysis scores are saved
        - **AI Processing**: Resume analysis is done securely through Google's Gemini API
        - **Job Search**: Job listings are fetched from Jobicy API
        - **Data Control**: You can request account deletion anytime

        This tool is designed for educational and career development purposes only.
        """)

    with st.container():
        # Display user info or welcome message
        if 'user' in st.session_state and st.session_state.user:
            st.success(f"Welcome, **{st.session_state.user['username']}**")

        # Place the button in a div with the center-button class
        with st.container():
            if st.button("Logout"):
                if 'user' in st.session_state:
                    del st.session_state.user
                st.rerun()


def main():
    """Application entry point"""
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Powered Job Finder & Resume Optimizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize database
    init_database()

    # Route to appropriate interface based on authentication
    if 'user' not in st.session_state:
        show_login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()