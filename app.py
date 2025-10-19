import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import plotly.graph_objects as go
import plotly.express as px
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .benign {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .malignant {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_model():
    """Load data and train neural network model"""
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Use only mean values (first 10 features)
    X_mean = X[:, :10]
    
    # Initialize scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mean)
    
    # Build Neural Network Model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    model.fit(
        X_scaled, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Get feature names for mean values
    feature_names = [name.replace(' (mean)', '') for name in data.feature_names[:10]]
    
    return model, scaler, feature_names, X_scaled, y

def preprocess_input(features_dict, scaler):
    """Preprocess input features"""
    features_array = np.array(list(features_dict.values())).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    return features_scaled

def make_prediction(features_scaled, model):
    """Make prediction and get probability"""
    probability = model.predict(features_scaled, verbose=0)[0][0]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def plot_prediction_confidence(probability):
    """Plot prediction confidence"""
    benign_prob = 1 - probability
    malignant_prob = probability
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Benign', 'Malignant'],
            y=[benign_prob, malignant_prob],
            marker=dict(color=['#28a745', '#dc3545']),
            text=[f'{benign_prob*100:.1f}%', f'{malignant_prob*100:.1f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        xaxis_title="Diagnosis",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 1])
    )
    return fig

def plot_input_distribution(features_dict, feature_names, X_mean):
    """Plot input values against typical ranges"""
    feature_values = list(features_dict.values())
    first_feature_idx = 0
    
    fig = go.Figure()
    
    # Add box plot for dataset distribution
    fig.add_trace(go.Box(
        y=X_mean[:, first_feature_idx],
        name="Dataset Range",
        marker=dict(color='rgba(58, 123, 213, 0.3)')
    ))
    
    # Add scatter for input value
    fig.add_trace(go.Scatter(
        x=['Dataset Range'],
        y=[feature_values[first_feature_idx]],
        mode='markers',
        name='Your Input',
        marker=dict(size=15, color='red', symbol='diamond')
    ))
    
    fig.update_layout(
        title=f"Input Value vs Dataset Distribution ({feature_names[first_feature_idx]})",
        yaxis_title="Feature Value",
        height=400,
        showlegend=True
    )
    return fig

def plot_input_features(features_dict, feature_names):
    """Plot all input feature values"""
    feature_values = list(features_dict.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=feature_names,
            y=feature_values,
            marker=dict(color='rgba(58, 123, 213, 0.8)'),
            text=[f'{v:.2f}' for v in feature_values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="All Input Feature Values",
        yaxis_title="Value",
        xaxis_title="Features",
        height=400,
        showlegend=False,
        xaxis_tickangle=-45
    )
    return fig

# Main application
st.title("🔬 Breast Cancer Detection System (Neural Network)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This application uses a Deep Neural Network model "
        "trained on the Wisconsin Breast Cancer dataset to predict whether "
        "a tumor is benign or malignant based on 10 diagnostic features."
    )
    st.markdown("---")
    st.header("Model Information")
    st.write("**Model Type:** Deep Neural Network")
    st.write("**Architecture:** 64 → 32 → 16 → 1 neurons")
    st.write("**Activation:** ReLU + Sigmoid")
    st.write("**Training Data:** Wisconsin Breast Cancer Dataset")
    st.write("**Features:** 10 tumor characteristics (mean values)")
    st.write("**Classes:** Benign / Malignant")

# Load model
model, scaler, feature_names, X_mean, y = load_and_train_model()

# Get model accuracy
train_predictions = model.predict(scaler.transform(X_mean), verbose=0)
train_accuracy = np.mean((train_predictions.flatten() > 0.5).astype(int) == y) * 100

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "Instructions"])

with tab1:
    st.header("Enter Tumor Diagnostic Features (Mean Values)")
    
    # Create input form
    st.write("Enter the mean values of the tumor characteristics:")
    
    col1, col2 = st.columns(2)
    
    features_dict = {}
    
    for i, feat_name in enumerate(feature_names):
        if i < 5:
            with col1:
                features_dict[feat_name] = st.number_input(
                    f"{i+1}. {feat_name}",
                    value=12.0 + i,
                    step=0.1,
                    key=f"feat_{i}"
                )
        else:
            with col2:
                features_dict[feat_name] = st.number_input(
                    f"{i+1}. {feat_name}",
                    value=12.0 + i,
                    step=0.1,
                    key=f"feat_{i}"
                )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔍 Predict Diagnosis", use_container_width=True):
        # Preprocess and predict
        features_scaled = preprocess_input(features_dict, scaler)
        prediction, probability = make_prediction(features_scaled, model)
        
        # Display results
        st.markdown("### Prediction Results")
        
        if prediction == 0:
            diagnosis = "🟢 BENIGN"
            confidence = 1 - probability
            css_class = "benign"
            prognosis = "The tumor is likely benign (non-cancerous)."
        else:
            diagnosis = "🔴 MALIGNANT"
            confidence = probability
            css_class = "malignant"
            prognosis = "The tumor is likely malignant (cancerous). Further medical consultation is recommended."
        
        st.markdown(
            f"""
            <div class="prediction-box {css_class}">
                <h2 style="margin: 0;">{diagnosis}</h2>
                <p style="margin: 10px 0; font-size: 18px;">
                    <strong>Confidence: {confidence*100:.2f}%</strong>
                </p>
                <p style="margin: 10px 0; font-size: 14px;">{prognosis}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Display visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_prediction_confidence(probability),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_input_features(features_dict, feature_names),
                use_container_width=True
            )
        
        # Distribution comparison
        st.markdown("---")
        st.subheader("Input vs Dataset Distribution")
        st.plotly_chart(
            plot_input_distribution(features_dict, feature_names, X_mean),
            use_container_width=True
        )
        
        # Additional metrics
        st.markdown("---")
        st.subheader("Prediction Details")
        col1, col2, col3 = st.columns(3)
        
        benign_prob = 1 - probability
        with col1:
            st.metric("Benign Probability", f"{benign_prob*100:.2f}%")
        with col2:
            st.metric("Malignant Probability", f"{probability*100:.2f}%")
        with col3:
            st.metric("Model Confidence", f"{max(benign_prob, probability)*100:.2f}%")

with tab2:
    st.header("Neural Network Model Information")
    
    st.subheader("Model Architecture")
    st.write("""
    **Neural Network Structure:**
    - Input Layer: 10 neurons (one for each feature)
    - Hidden Layer 1: 64 neurons + ReLU activation + Dropout (30%)
    - Hidden Layer 2: 32 neurons + ReLU activation + Dropout (30%)
    - Hidden Layer 3: 16 neurons + ReLU activation
    - Output Layer: 1 neuron + Sigmoid activation
    """)
    
    st.write(f"**Model Accuracy on Training Data:** {train_accuracy:.2f}%")
    st.write(f"**Total Features Used:** 10 (Mean values only)")
    st.write(f"**Dataset Samples:** {len(X_mean)}")
    st.write(f"**Benign Samples:** {(y == 0).sum()}")
    st.write(f"**Malignant Samples:** {(y == 1).sum()}")
    
    st.subheader("Features Used (Mean Values Only)")
    st.write("This simplified model uses only the mean characteristics:")
    for idx, feat in enumerate(feature_names, 1):
        st.write(f"{idx}. {feat}")
    
    st.subheader("Why Neural Networks?")
    st.info("""
    ✅ **Advantages:**
    - Can capture non-linear relationships
    - Better at handling complex patterns
    - Highly accurate with proper architecture
    - Can process large datasets efficiently
    
    ⚡ **This Model:**
    - Uses dropout for regularization
    - Simple enough to understand
    - Uses only 10 features for clarity
    - Sigmoid activation for binary classification
    """)

with tab3:
    st.header("How to Use This Application")
    
    st.write("""
    ### Step-by-Step Guide:
    
    1. **Enter Tumor Characteristics**: Input the 10 mean diagnostic features
       - Values are typically in the range of 5-35
       - Use actual measurements from medical imaging
    
    2. **Click "Predict Diagnosis"**: The neural network will process your inputs
    
    3. **Review Results**: 
       - See the diagnosis (Benign or Malignant)
       - Check the confidence level (probability)
       - Examine visualizations
    
    4. **Understand Visualizations**:
       - Bar chart shows your input values
       - Confidence chart shows prediction probabilities
       - Distribution chart compares to dataset
    
    ### Important Notes:
    
    ⚠️ **Medical Disclaimer**: This tool is for educational purposes only. 
    It should NOT be used as a substitute for professional medical diagnosis. 
    Always consult with qualified healthcare professionals for actual medical decisions.
    
    ✅ **Best Practices**:
    - Ensure all feature values are realistic
    - Use actual diagnostic measurements from medical imaging
    - Consider multiple diagnostic tools and expert opinion
    - Understand that AI is a support tool, not a replacement for doctors
    
    ### Dataset Information:
    - **Source**: Wisconsin Breast Cancer Database
    - **Total Samples**: 569 patient records
    - **Benign Cases**: 357 (62.7%)
    - **Malignant Cases**: 212 (37.3%)
    - **Features Used**: 10 mean values per sample
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 12px;">
        <p>🔬 Breast Cancer Detection System | Neural Network Model</p>
        <p>Developed with Streamlit, TensorFlow/Keras, and scikit-learn</p>
        <p>⚠️ Not for clinical diagnosis. Always consult healthcare professionals.</p>
    </div>
    """,
    unsafe_allow_html=True
)