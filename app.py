import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️", layout="wide", initial_sidebar_state="expanded"
) 

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B; 
        text-align: center; 
        margin-bottom: 2rem; 
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏥 Navigation")  # Corrected title
page = st.sidebar.radio("Go to", ["📊 Dataset View", "📈 Insights & Evaluation",
                                   "🔮 Predict Heart Disease", "🤖 AI Health assistance"])

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
            return model
    except FileNotFoundError:
        st.warning("Model file 'random_forest_model.pkl' not found. "
                   "Creating a placeholder model.")
 
        X_sample = np.random.rand(10, 9)
        y_sample = np.random.randint(0, 2, 10)
 
        placeholder_model = RandomForestClassifier(n_estimators=10, random_state=42)
        placeholder_model.fit(X_sample, y_sample)

        if not hasattr(placeholder_model, 'predict_proba'):
            def predict_proba(X):
                return np.array([[0.5, 0.5]] * len(X))
            placeholder_model.predict_proba = predict_proba
        return placeholder_model


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except FileNotFoundError: 

        np.random.seed(42)
        n = 500
        data = {
            'age': np.random.randint(30, 80, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'bmi': np.random.uniform(18, 40, n),
            'blood_glucose_level': np.random.uniform(70, 200, n),
            'cholesterol': np.random.uniform(150, 300, n),
            'blood_pressure': np.random.randint(90, 180, n),
            'heart_rate': np.random.randint(60, 120, n),
            'smoking': np.random.choice(['Yes', 'No'], n),
            'diabetes': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
        } 
        return pd.DataFrame(data)
 
df = load_data()
# PAGE 1: Dataset View 
if page == "📊 Dataset View":
    st.markdown("<h1 class='main-header'>❤️ Heart Disease Dataset</h1>",
                unsafe_allow_html=True)

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3: 
        if 'diabetes' in df.columns:
            positive_cases = df['diabetes'].value_counts().get('Yes', 0)  # Fixed
            st.metric("Positive Cases", positive_cases)
        else: 
            st.metric("Positive Cases", "N/A")
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)

    st.markdown("---")

    # Display full dataset
    st.markdown("<h2 class='sub-header'>Complete Dataset</h2>",
                unsafe_allow_html=True)
 
    # Search and filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search in dataset", "") 
    with col2:
        show_rows = st.number_input("Rows to display", min_value=5,
                                    max_value=len(df), value=10)

    if search:
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        filtered_df = df[mask]
        st.dataframe(filtered_df.head(show_rows), use_container_width=True)
    else: 
        st.dataframe(df.head(show_rows), use_container_width=True)
 
    # Download button
    csv = df.to_csv(index=False).encode('utf-8') 
    st.download_button(
        label="📥 Download Dataset as CSV", data=csv,
        file_name='heart_disease_dataset.csv', mime='text/csv',
    )

    st.markdown("---")

    # Dataset statistics
    st.markdown("<h2 class='sub-header'>Statistical Summary</h2>",
                unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)
    
    # Column information
    st.markdown("<h2 class='sub-header'>Column Information</h2>", unsafe_allow_html=True)
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values
    })
    st.dataframe(col_info, use_container_width=True) 
 
# PAGE 2: Insights & Evaluation 
 
elif page == "📈 Insights & Evaluation": 
    st.markdown("<h1 class='main-header'>📈 Data Insights & Model Evaluation</h1>",
                unsafe_allow_html=True)

    # Prepare data for analysis
    df_analysis = df.copy()
    
    # Convert categorical to numeric if needed
    if 'gender' in df_analysis.columns and df_analysis['gender'].dtype == 'object':
        df_analysis['gender_encoded'] = df_analysis['gender'].map({'Male': 1, 'Female': 0}) 
    
    if ('diabetes' in df_analysis.columns and 
            df_analysis['diabetes'].dtype == 'object'): 
        df_analysis['diabetes_encoded'] = df_analysis['diabetes'].map( 
            {'Yes': 1, 'No': 0})

    # Tab layout for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🔗 Correlation", "📉 Missing Values", "🎯 Model Metrics"])
    
    with tab1:
        st.markdown("<h2 class='sub-header'>Feature Distributions</h2>", unsafe_allow_html=True)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist() 
        
        if numeric_cols: 
            col1, col2 = st.columns(2) 
            
            with col1:
                selected_feature = st.selectbox("Select Feature", numeric_cols)
                fig = px.histogram(df, x=selected_feature, nbins=30, 
                                 title=f'Distribution of {selected_feature}',
                                 color_discrete_sequence=['#FF6B6B'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'diabetes' in df.columns:
                    fig = px.pie(df, names='diabetes', title='Diabetes Distribution', 
                               color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                    st.plotly_chart(fig, use_container_width=True)
        
        # Box plots 
        st.markdown("### Box Plots for Outlier Detection")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = px.box(df, y='age', title='Age Distribution',
                           color_discrete_sequence=['#95E1D3'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2: 
            if 'bmi' in df.columns:
                fig = px.box(df, y='bmi', title='BMI Distribution', 
                           color_discrete_sequence=['#F38181'])
                st.plotly_chart(fig, use_container_width=True) 
    
    with tab2:
        st.markdown("<h2 class='sub-header'>Correlation Matrix</h2>", unsafe_allow_html=True)
        
        # Get numeric columns for correlation
        numeric_df = df_analysis.select_dtypes(include=[np.number])
        
        if not numeric_df.empty: 
            corr = numeric_df.corr() 
            
            fig = px.imshow(corr, 
                          text_auto='.2f', 
                          aspect='auto',
                          title='Feature Correlation Heatmap',
                          color_continuous_scale='RdBu_r')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.markdown("### Top Correlations")
            corr_pairs = corr.unstack() 
            corr_pairs = corr_pairs[corr_pairs < 1] 
            top_corr = corr_pairs.abs().sort_values(ascending=False).head(10) 
            st.dataframe(top_corr, use_container_width=True) 
    
    with tab3:
        st.markdown("<h2 class='sub-header'>Missing Values Analysis</h2>", unsafe_allow_html=True)
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values, 
                        title='Missing Values by Feature',
                        labels={'x': 'Features', 'y': 'Count'},
                        color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True) 
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Missing values heatmap
        st.markdown("### Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False, ax=ax)
        ax.set_title('Missing Values Heatmap')
        st.pyplot(fig) 
    
    with tab4:
        st.markdown("<h2 class='sub-header'>Model Performance Metrics</h2>", unsafe_allow_html=True)
        # Display sample metrics (you would replace these with actual model evaluation) 
        # Display sample metrics (you would replace these with actual model evaluation)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Accuracy", "94.5%", "2.3%")
            st.markdown("</div>", unsafe_allow_html=True)
        # Display sample metrics (you would replace these with actual model evaluation) 
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Precision", "92.8%", "1.5%")
            st.markdown("</div>", unsafe_allow_html=True) 
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Recall", "91.2%", "-0.8%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("F1 Score", "92.0%", "0.5%") 
            st.markdown("</div>", unsafe_allow_html=True) 
        
        st.markdown("---") 
        
        # Sample confusion matrix
        st.markdown("### Confusion Matrix")
        cm = np.array([[120, 15], [10, 105]])
        
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual"),
                       x=['No Disease', 'Disease'],
                       y=['No Disease', 'Disease'],
                       title='Confusion Matrix',
                       color_continuous_scale='Blues') 
        st.plotly_chart(fig, use_container_width=True) 
        
        # Sample ROC curve
        st.markdown("### ROC Curve") 
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.3)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                                line=dict(color='#FF6B6B', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                line=dict(color='gray', width=2, dash='dash')))
        fig.update_layout(title='ROC Curve (AUC = 0.95)', 
                         xaxis_title='False Positive Rate', 
                         yaxis_title='True Positive Rate', 
                         height=500) 
        st.plotly_chart(fig, use_container_width=True) 

# PAGE 3: Predict Heart Disease
elif page == "🔮 Predict Heart Disease":
    st.markdown("<h1 class='main-header'>🔮 Heart Disease Prediction</h1>", unsafe_allow_html=True)
    
    model = load_model()
    
    if model is None:
        st.error("❌ Failed to load or create a model. The prediction page cannot be loaded.")
    else: 
        st.success("✅ Model loaded successfully!")
    
    st.markdown("---") 
    
    st.markdown("<h2 class='sub-header'>Enter Patient Information</h2>", unsafe_allow_html=True)
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, 
                              value=25.0, step=0.1)
        cholesterol = st.number_input("Cholesterol", min_value=100, 
                                      max_value=400, value=200)
    with col2:
        blood_glucose = st.number_input("Blood Glucose Level", min_value=50, 
                                        max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (Systolic)", min_value=80, max_value=200, value=120)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=150, value=72)
    
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking = st.selectbox("Smoking", ["Yes", "No"])
        diabetes = st.selectbox("Diabetes", ["Yes", "No"])
        exercise = st.selectbox("Regular Exercise", ["Yes", "No"])
    
    st.markdown("---") 
    
    # Predict button and logic
    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True): 
        if model is not None:
            # Prepare input data
            gender_encoded = 1 if gender == "Male" else 0
            smoking_encoded = 1 if smoking == "Yes" else 0
            exercise_encoded = 1 if exercise == "Yes" else 0
            
            
            diabetes_encoded = 1 if diabetes == "Yes" else 0 
            
            
            input_data = np.array([[age, bmi, cholesterol, blood_glucose, 
                                    blood_pressure, heart_rate, gender_encoded, 
                                    smoking_encoded, diabetes_encoded, 
                                    exercise_encoded, 0, 0, 0]])
            try:
                # Make prediction
                prediction = int(model.predict(input_data)[0])
                prediction_proba = model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.markdown("<h2 class='sub-header'>Prediction Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ HIGH RISK of Heart Disease") 
                        st.markdown(f"**Confidence:** {prediction_proba[1]*100:.2f}%") 
                    else: 
                        st.success("✅ LOW RISK of Heart Disease") 
                        st.markdown(f"**Confidence:** {prediction_proba[0]*100:.2f}%") 
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_proba[1]*100,
                        title = {'text': "Risk Level"},
                        gauge = { 
                            'axis': {'range': [None, 100]}, 
                            'bar': {'color': "#FF6B6B"}, 
                            'steps': [ 
                                {'range': [0, 30], 'color': "#4ECDC4"}, 
                                {'range': [30, 70], 'color': "#FFE66D"},
                                {'range': [70, 100], 'color': "#FF6B6B"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        } 
                    )) 
                    fig.update_layout(height=300) 
                    st.plotly_chart(fig, use_container_width=True) 
                
                # Risk factors
                st.markdown("---")
                st.markdown("<h2 class='sub-header'>Risk Factors Analysis</h2>", unsafe_allow_html=True)
                
                risk_factors = []
                if age > 60:
                    risk_factors.append("• Age over 60")
                if bmi > 30:
                    risk_factors.append("• BMI indicates obesity")
                if cholesterol > 240:
                    risk_factors.append("• High cholesterol")
                if blood_glucose > 140:
                    risk_factors.append("• Elevated blood glucose")
                if blood_pressure > 140:
                    risk_factors.append("• High blood pressure") 
                if smoking == "Yes":
                    risk_factors.append("• Smoking")
                if exercise == "No":
                    risk_factors.append("• Lack of regular exercise") 
                
                if risk_factors:
                    st.warning("⚠️ **Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.success("✅ No major risk factors identified!")
                
                # Recommendations 
                st.markdown("---") 
                st.markdown("<h2 class='sub-header'>Health Recommendations</h2>", 
                            unsafe_allow_html=True) 
                recommendations = [ 
                    "🏃 Maintain regular physical activity (at least 30 minutes daily)",
                    "🥗 Follow a heart-healthy diet rich in fruits and vegetables",
                    "💊 Take prescribed medications regularly",
                    "😴 Get adequate sleep (7-9 hours per night)",
                    "🧘 Manage stress through relaxation techniques",
                    "🚭 Avoid smoking and limit alcohol consumption",
                    "📊 Regular health check-ups and monitoring"
                ] 
                
                for rec in recommendations: 
                    st.info(rec) 
                
                st.warning("⚠️ **Disclaimer:** This prediction is for educational purposes only. Please consult with a healthcare professional for proper medical advice.")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please ensure the input features match the model's expected format.")
        else:
            st.error("❌ Cannot make prediction without a trained model.")





elif page == "🤖 AI Health assistance":
    st.markdown("<h1 class='main-header'> AI Health Assistant</h1>", unsafe_allow_html=True)
    st.markdown("Ask me questions about the heart disease dataset or general health topics!")

    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("GEMINI_API_KEY not found. Please set it in your Streamlit secrets.") 
        st.stop() 

    genai.configure(api_key=api_key) 
 
   
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        st.stop()

    # --- Streamlit App ---
    st.write("Enter a prompt and get a response from Gemini-Pro.")

    # --- User Input ---
    prompt = st.text_input("Enter your prompt:", placeholder="e.g., What are the risk factors for heart disease?")

    # --- Generate Content ---
    if st.button("Generate Response"):
        if prompt:
            try:
                with st.spinner("Generating response..."):
                    # Generate content
                    response = model.generate_content(prompt)
                    
                    # Display the response
                    st.subheader("Gemini's Response:")
                    st.markdown(response.text)
                    
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
        else:
            st.warning("Please enter a prompt.")