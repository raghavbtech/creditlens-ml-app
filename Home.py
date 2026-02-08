"""
CreditLens — ML Loan Approval Prediction System
Professional refactored version with improved code organization and maintainability
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CreditLens | Loan Approval",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .main-header p {
        font-family: 'Sora', sans-serif;
        font-weight: 300;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
        color: #667eea;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Input labels */
    .stSelectbox label, .stNumberInput label {
        font-family: 'Sora', sans-serif;
        font-weight: 400;
        color: #e0e6ff !important;
        font-size: 0.95rem;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput {
        margin-bottom: 0.8rem;
    }
    
    /* Fix selectbox styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        color: #e0e6ff;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #667eea;
    }
    
    /* Selectbox dropdown menu */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(26, 31, 58, 0.95);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Selectbox options */
    [data-baseweb="popover"] {
        background-color: #1a1f3a !important;
    }
    
    [role="listbox"] {
        background-color: #1a1f3a !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    [role="option"] {
        background-color: #1a1f3a !important;
        color: #e0e6ff !important;
    }
    
    [role="option"]:hover {
        background-color: rgba(102, 126, 234, 0.2) !important;
    }
    
    [aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        color: #e0e6ff;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 1px #667eea;
    }
    
    /* Prediction button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        border: none;
        border-radius: 12px;
        margin-top: 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .result-approved {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    }
    
    .result-rejected {
        border-left: 4px solid #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
    }
    
    .result-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .result-prob {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: #a0aec0;
        margin-top: 1rem;
    }
    
    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-family: 'Sora', sans-serif;
        color: #e0e6ff;
    }
    
    /* Credit score indicator */
    .credit-indicator {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .credit-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .credit-excellent { background: linear-gradient(90deg, #10b981, #059669); }
    .credit-good { background: linear-gradient(90deg, #3b82f6, #2563eb); }
    .credit-fair { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .credit-poor { background: linear-gradient(90deg, #ef4444, #dc2626); }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        font-family: 'Sora', sans-serif;
        color: #a0aec0;
        font-size: 0.85rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (CACHED FOR PERFORMANCE)
# ============================================================================

@st.cache_resource
def load_ml_models():
    """Load pre-trained ML models and scaler from disk."""
    try:
        log_model = joblib.load("log_model.pkl")
        nb_model = joblib.load("nb_model.pkl")
        knn_model = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return log_model, nb_model, knn_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


log_model, nb_model, knn_model, scaler = load_ml_models()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_credit_status(credit_score: float) -> tuple[str, str]:
    """
    Determine credit rating based on score.
    Returns: (status_text, color_hex)
    """
    if credit_score >= 750:
        return "Excellent", "#10b981"
    elif credit_score >= 700:
        return "Good", "#3b82f6"
    elif credit_score >= 650:
        return "Fair", "#f59e0b"
    else:
        return "Needs Improvement", "#ef4444"


def get_dti_status(dti_ratio: float) -> tuple[str, str]:
    """
    Determine DTI (Debt-to-Income) rating based on ratio.
    Returns: (status_text, color_hex)
    """
    if dti_ratio <= 30:
        return "Excellent", "#10b981"
    elif dti_ratio <= 40:
        return "Good", "#3b82f6"
    elif dti_ratio <= 50:
        return "Fair", "#f59e0b"
    else:
        return "High Risk", "#ef4444"


def get_credit_bar_class(credit_score: float) -> str:
    """Get CSS class for credit score progress bar."""
    if credit_score >= 750:
        return "credit-excellent"
    elif credit_score >= 700:
        return "credit-good"
    elif credit_score >= 650:
        return "credit-fair"
    else:
        return "credit-poor"


def prepare_features(
    age: int,
    gender: str,
    marital_status: str,
    dependents: int,
    education: str,
    employment_status: str,
    employer_category: str,
    applicant_income: float,
    coapplicant_income: float,
    savings: float,
    existing_loans: int,
    credit_score: float,
    dti_ratio: float,
    loan_amount: float,
    loan_term: int,
    collateral_value: float,
    loan_purpose: str,
    property_area: str
) -> pd.DataFrame:
    """
    Prepare feature vector for ML model prediction.
    Encodes categorical variables and normalizes features.
    """
    # Categorical encoding
    education_encoded = 1 if education == "Graduate" else 0
    employment_salaried = int(employment_status == "Salaried")
    employment_self_emp = int(employment_status == "Self-employed")
    employment_unemployed = int(employment_status == "Unemployed")
    marital_single = int(marital_status == "Single")
    loan_purpose_car = int(loan_purpose == "Car")
    loan_purpose_education = int(loan_purpose == "Education")
    loan_purpose_home = int(loan_purpose == "Home")
    loan_purpose_personal = int(loan_purpose == "Personal")
    area_semiurban = int(property_area == "Semiurban")
    area_urban = int(property_area == "Urban")
    gender_male = int(gender == "Male")
    employer_govt = int(employer_category == "Government")
    employer_mnc = int(employer_category == "MNC")
    employer_private = int(employer_category == "Private")
    employer_unemployed = int(employer_category == "Unemployed")
    
    # Feature interactions
    dti_squared = dti_ratio ** 2
    credit_score_squared = credit_score ** 2
    
    # Build feature vector (must match training feature order)
    feature_names = [
    'Applicant_Income',
    'Coapplicant_Income',
    'Age',
    'Dependents',
    'Existing_Loans',
    'Savings',
    'Collateral_Value',
    'Loan_Amount',
    'Loan_Term',
    'Education_Level',
    'Employment_Status_Salaried',
    'Employment_Status_Self-employed',
    'Employment_Status_Unemployed',
    'Marital_Status_Single',
    'Loan_Purpose_Car',
    'Loan_Purpose_Education',
    'Loan_Purpose_Home',
    'Loan_Purpose_Personal',
    'Property_Area_Semiurban',
    'Property_Area_Urban',
    'Gender_Male',
    'Employer_Category_Government',
    'Employer_Category_MNC',
    'Employer_Category_Private',
    'Employer_Category_Unemployed',
    'DTI_Ratio_sq',
    'Credit_Score_sq'
    ]

    values = [[
        applicant_income,
        coapplicant_income,
        age,
        dependents,
        existing_loans,
        savings,
        collateral_value,
        loan_amount,
        loan_term,
        education_encoded,
        employment_salaried,
        employment_self_emp,
        employment_unemployed,
        marital_single,
        loan_purpose_car,
        loan_purpose_education,
        loan_purpose_home,
        loan_purpose_personal,
        area_semiurban,
        area_urban,
        gender_male,
        employer_govt,
        employer_mnc,
        employer_private,
        employer_unemployed,
        dti_squared,
        credit_score_squared
    ]]

    return pd.DataFrame(values, columns=feature_names)


# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🔍 CreditLens</h1>
    <p>ML-Powered Loan Approval & Risk Prediction</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION & INFORMATION
# ============================================================================

with st.sidebar:
    st.markdown("### 🤖 Model Selection")
    
    model_choice = st.selectbox(
        "Choose Prediction Model",
        ["Logistic Regression ⭐", "Naive Bayes", "KNN"],
        help="Logistic Regression recommended for best accuracy"
    )
    
    st.markdown("### 📊 About the Models")
    st.markdown("""
    <div class="info-box">
    <strong>Logistic Regression</strong> — Best overall accuracy and probability estimates
    <br><br>
    <strong>Naive Bayes</strong> — Fast predictions with good performance
    <br><br>
    <strong>KNN</strong> — Pattern-based classification
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    <div class="info-box">
    <strong>Improve Your Chances:</strong><br>
    • Maintain credit score above 700<br>
    • Keep DTI ratio below 40%<br>
    • Show stable employment<br>
    • Provide collateral when possible
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer-text">
    Built with ML & ❤️<br>
    Version 2.1 (Refactored)
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN FORM - PERSONAL INFORMATION
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
    
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    
    st.markdown('<p class="section-header">💼 Employment Details</p>', unsafe_allow_html=True)
    
    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )
    employer_category = st.selectbox(
        "Employer Category",
        ["Government", "MNC", "Private", "Unemployed", "Other"]
    )

with col2:
    st.markdown('<p class="section-header">💰 Financial Information</p>', unsafe_allow_html=True)
    
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0.0, value=50000.0, step=1000.0)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0.0, value=0.0, step=1000.0)
    savings = st.number_input("Savings (₹)", min_value=0.0, value=10000.0, step=1000.0)
    existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=20, value=0)
    
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    
    # Display credit score indicator
    credit_status, credit_color = get_credit_status(credit_score)
    credit_percentage = ((credit_score - 300) / 600) * 100
    credit_bar_class = get_credit_bar_class(credit_score)
    
    st.markdown(f"""
    <div style="margin-top: -0.5rem; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="font-family: 'Sora', sans-serif; font-size: 0.8rem; color: #a0aec0;">Rating: <strong style="color: {credit_color};">{credit_status}</strong></span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #a0aec0;">{int(credit_score)}/900</span>
        </div>
        <div class="credit-indicator">
            <div class="credit-bar {credit_bar_class}" style="width: {credit_percentage}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    dti_ratio = st.number_input(
        "Debt-to-Income Ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        help="Percentage of monthly income that goes to debt payments"
    )
    
    # Display DTI indicator
    dti_status, dti_color = get_dti_status(dti_ratio)
    st.markdown(f"""
    <div style="margin-top: -0.5rem; margin-bottom: 1rem;">
        <span style="font-family: 'Sora', sans-serif; font-size: 0.8rem; color: #a0aec0;">Status: <strong style="color: {dti_color};">{dti_status}</strong></span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAN DETAILS
# ============================================================================

st.markdown('<p class="section-header">🏠 Loan Details</p>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0.0, value=100000.0, step=10000.0)

with col4:
    loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=480, value=360)

with col5:
    collateral_value = st.number_input("Collateral Value (₹)", min_value=0.0, value=0.0, step=10000.0)

col6, col7 = st.columns(2)

with col6:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Car", "Education", "Home", "Personal", "Other"]
    )

with col7:
    property_area = st.selectbox(
        "Property Area",
        ["Semiurban", "Urban", "Rural"]
    )

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

if st.button("🔮 Get Loan Approval Prediction", use_container_width=True):
    
    if log_model is None or nb_model is None or knn_model is None:
        st.error("Models failed to load. Please ensure all .pkl files are available.")
    else:
        with st.spinner("Analyzing your application..."):
            # Prepare features for prediction
            features = prepare_features(
                age=age,
                gender=gender,
                marital_status=marital_status,
                dependents=dependents,
                education=education,
                employment_status=employment_status,
                employer_category=employer_category,
                applicant_income=applicant_income,
                coapplicant_income=coapplicant_income,
                savings=savings,
                existing_loans=existing_loans,
                credit_score=credit_score,
                dti_ratio=dti_ratio,
                loan_amount=loan_amount,
                loan_term=loan_term,
                collateral_value=collateral_value,
                loan_purpose=loan_purpose,
                property_area=property_area
            )
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Get prediction from selected model
            if model_choice == "Logistic Regression ⭐":
                prediction = log_model.predict(features_scaled)[0]
                probability = log_model.predict_proba(features_scaled)[0]
            elif model_choice == "Naive Bayes":
                prediction = nb_model.predict(features_scaled)[0]
                probability = nb_model.predict_proba(features_scaled)[0]
            else:  # KNN
                prediction = knn_model.predict(features_scaled)[0]
                probability = knn_model.predict_proba(features_scaled)[0]
            
            # Extract approval probability
            try:
                approval_prob = probability[1] * 100
            except (IndexError, TypeError):
                approval_prob = 0
            
            # Display results
            if prediction == 1:
                st.markdown("""
                <div class="result-card result-approved">
                    <div class="result-title" style="color: #10b981;">✅ Application Approved</div>
                    <p style="color: #e0e6ff; margin: 0.5rem 0;">Congratulations! Your loan application has been approved.</p>
                    <div class="result-prob">Confidence: {:.2f}%</div>
                </div>
                """.format(approval_prob), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card result-rejected">
                    <div class="result-title" style="color: #ef4444;">❌ Application Declined</div>
                    <p style="color: #e0e6ff; margin: 0.5rem 0;">Unfortunately, your application was not approved at this time.</p>
                    <div class="result-prob">Approval Probability: {:.2f}%</div>
                    <p style="color: #a0aec0; margin-top: 1rem; font-size: 0.9rem;">💡 Try improving your credit score or reducing your DTI ratio for better chances.</p>
                </div>
                """.format(approval_prob), unsafe_allow_html=True)
            
            # Display model info
            st.info(f"Prediction made using: **{model_choice}**")
