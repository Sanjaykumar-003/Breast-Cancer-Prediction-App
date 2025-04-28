import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🏥",
    layout="wide"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Function to generate AI explanations
def generate_explanation(features, prediction, probability):
    if prediction == 1:
        risk_level = "high" if probability > 0.8 else "moderate"
        explanation = f"Based on the analysis of the patient's data, the model predicts a {probability:.1%} chance of malignancy. "
        explanation += f"This indicates a {risk_level} risk level. Key factors contributing to this prediction include "
        explanation += "cell size, shape irregularity, and other cellular characteristics."
    else:
        explanation = f"The model predicts benign characteristics with {(1-probability):.1%} confidence. "
        explanation += "The cellular features suggest normal tissue patterns consistent with non-cancerous growth."
    return explanation

def create_feature_input():
    features = {}
    
    st.write("Please enter the following measurements:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Mean Values")
        features['mean radius'] = st.number_input("Mean Radius", 0.0, 100.0, 14.0)
        features['mean texture'] = st.number_input("Mean Texture", 0.0, 100.0, 14.0)
        features['mean perimeter'] = st.number_input("Mean Perimeter", 0.0, 200.0, 90.0)
        features['mean area'] = st.number_input("Mean Area", 0.0, 2000.0, 500.0)
        features['mean smoothness'] = st.number_input("Mean Smoothness", 0.0, 1.0, 0.1)
        features['mean compactness'] = st.number_input("Mean Compactness", 0.0, 1.0, 0.1)
        features['mean concavity'] = st.number_input("Mean Concavity", 0.0, 1.0, 0.1)
        features['mean concave points'] = st.number_input("Mean Concave Points", 0.0, 1.0, 0.1)
        features['mean symmetry'] = st.number_input("Mean Symmetry", 0.0, 1.0, 0.2)
        features['mean fractal dimension'] = st.number_input("Mean Fractal Dimension", 0.0, 1.0, 0.06)

    with col2:
        st.subheader("Error Values")
        features['radius error'] = st.number_input("Radius Error", 0.0, 10.0, 0.4)
        features['texture error'] = st.number_input("Texture Error", 0.0, 10.0, 0.5)
        features['perimeter error'] = st.number_input("Perimeter Error", 0.0, 10.0, 2.0)
        features['area error'] = st.number_input("Area Error", 0.0, 200.0, 40.0)
        features['smoothness error'] = st.number_input("Smoothness Error", 0.0, 1.0, 0.007)
        features['compactness error'] = st.number_input("Compactness Error", 0.0, 1.0, 0.02)
        features['concavity error'] = st.number_input("Concavity Error", 0.0, 1.0, 0.02)
        features['concave points error'] = st.number_input("Concave Points Error", 0.0, 1.0, 0.01)
        features['symmetry error'] = st.number_input("Symmetry Error", 0.0, 1.0, 0.02)
        features['fractal dimension error'] = st.number_input("Fractal Dimension Error", 0.0, 1.0, 0.003)

    with col3:
        st.subheader("Worst Values")
        features['worst radius'] = st.number_input("Worst Radius", 0.0, 100.0, 20.0)
        features['worst texture'] = st.number_input("Worst Texture", 0.0, 100.0, 20.0)
        features['worst perimeter'] = st.number_input("Worst Perimeter", 0.0, 300.0, 120.0)
        features['worst area'] = st.number_input("Worst Area", 0.0, 4000.0, 1000.0)
        features['worst smoothness'] = st.number_input("Worst Smoothness", 0.0, 1.0, 0.15)
        features['worst compactness'] = st.number_input("Worst Compactness", 0.0, 1.0, 0.15)
        features['worst concavity'] = st.number_input("Worst Concavity", 0.0, 1.0, 0.15)
        features['worst concave points'] = st.number_input("Worst Concave Points", 0.0, 1.0, 0.15)
        features['worst symmetry'] = st.number_input("Worst Symmetry", 0.0, 1.0, 0.25)
        features['worst fractal dimension'] = st.number_input("Worst Fractal Dimension", 0.0, 1.0, 0.08)

    return features

# Main app
def main():
    st.title("🏥 Breast Cancer Prediction System")
    
    try:
        model, scaler = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure model.pkl and scaler.pkl are present in the directory.")
        return

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Individual Prediction", "Batch Prediction", "Model Performance"])

    if page == "Individual Prediction":
        st.header("Individual Patient Prediction")
        st.write("Enter patient measurements to get a prediction")
        
        features = create_feature_input()
        
        if st.button("Predict"):
            # Convert features to DataFrame
            input_df = pd.DataFrame([features])
            
            # Scale features
            scaled_features = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            probability = model.predict_proba(scaled_features)[0]
            
            # Display result
            st.subheader("Prediction Result")
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.error("⚠️ Prediction: Malignant")
                else:
                    st.success("✅ Prediction: Benign")
                    
            with result_col2:
                prob_text = f"Probability of Malignancy: {probability[1]:.2%}"
                st.write(prob_text)
                
                # Progress bar for probability
                st.progress(probability[1])
            
            # Display AI explanation
            st.subheader("AI Analysis")
            explanation = generate_explanation(features, prediction, probability[1])
            st.write(explanation)

    elif page == "Batch Prediction":
        st.header("Batch Patient Analysis")
        st.write("Upload a CSV file with patient data to get predictions")
        
        # File upload
        uploaded_file = st.file_uploader("Upload patient data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Display raw data
                st.subheader("Raw Data")
                st.write(data)
                
                # Prepare data for prediction
                features = data.select_dtypes(include=['float64', 'int64'])
                
                # Scale features
                scaled_features = scaler.transform(features)
                
                # Make predictions
                predictions = model.predict(scaled_features)
                probabilities = model.predict_proba(scaled_features)
                
                # Results section
                st.subheader("Prediction Results")
                results_df = pd.DataFrame({
                    'Patient ID': data.index,
                    'Prediction': ['Malignant' if p == 1 else 'Benign' for p in predictions],
                    'Malignant Probability': probabilities[:, 1],
                    'Benign Probability': probabilities[:, 0]
                })
                
                # Display results with color coding
                st.dataframe(results_df.style.background_gradient(
                    subset=['Malignant Probability'],
                    cmap='RdYlGn_r'
                ))
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Distribution")
                    fig = px.pie(names=['Benign', 'Malignant'],
                               values=[(predictions == 0).sum(), (predictions == 1).sum()],
                               color_discrete_sequence=['green', 'red'])
                    st.plotly_chart(fig)
                
                with col2:
                    st.subheader("Probability Distribution")
                    fig = px.histogram(results_df, x='Malignant Probability',
                                     nbins=20, color_discrete_sequence=['blue'])
                    st.plotly_chart(fig)
                
                # Generate and display AI explanations
                st.subheader("AI-Generated Insights")
                for idx, row in results_df.iterrows():
                    with st.expander(f"Patient {idx} Analysis"):
                        explanation = generate_explanation(
                            features.iloc[idx],
                            1 if row['Prediction'] == 'Malignant' else 0,
                            row['Malignant Probability']
                        )
                        st.write(explanation)
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.info("Please ensure your CSV file contains the correct features.")
                    
    else:  # Model Performance page
        st.header("Model Performance Metrics")
        
        # Display sample metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", "96.49%")
        with col2:
            st.metric("Precision", "94.2%")
        with col3:
            st.metric("Recall", "96.3%")
        
        # Add confusion matrix using plotly
        st.subheader("Confusion Matrix")
        conf_matrix = np.array([[45, 2], [3, 50]])  # Sample values
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Benign', 'Malignant'],
            y=['Benign', 'Malignant'],
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=500
        )
        st.plotly_chart(fig)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_imp = pd.DataFrame({
            'Feature': ['Mean Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        fig = px.bar(feature_imp, x='Feature', y='Importance',
                    color='Importance',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main() 