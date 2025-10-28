import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="MLOps Tools Comparison", page_icon="️", layout="wide")

st.title("️ MLOps Tools Comparison")

st.markdown("""
## What is MLOps?

**MLOps (Machine Learning Operations)** is essential for deploying and maintaining machine learning models in production. 
This page compares popular MLOps tools: **Orange**, **RapidMiner**, **Weka**, and **Python/Sklearn**.
""")

# Tool Comparison
st.subheader(" Tool Comparison")

# Orange
st.markdown("### Orange")
st.markdown("""
- **Type:** Open-source data visualization and analysis tool
- **Language:** Python
- **Best For:** Beginners, education, quick prototyping
- **Website:** [Orange](https://orange.biolab.si/)
""")

# RapidMiner
st.markdown("### RapidMiner")
st.markdown("""
- **Type:** Data science platform
- **Language:** Java
- **Best For:** Enterprises, automated machine learning, big data
- **Website:** [RapidMiner](https://rapidminer.com/)
""")

# Weka
st.markdown("### Weka")
st.markdown("""
- **Type:** Open-source software for machine learning
- **Language:** Java
- **Best For:** Academic research, algorithm development, education
- **Website:** [Weka](https://www.cs.waikato.ac.nz/ml/weka/)
""")

# Python/Sklearn
st.markdown("### Python/Sklearn")
st.markdown("""
- **Type:** Programming language and library for machine learning
- **Language:** Python
- **Best For:** Flexibility, custom machine learning solutions, production deployment
- **Website:** [Python](https://www.python.org/), [Sklearn](https://scikit-learn.org/)
""")

# Feature Comparison Table
st.subheader(" Feature Comparison Table")

comparison_data = {
    'Feature': ['Ease of Use', 'Flexibility', 'Scalability', 'Cost', 'Best For'],
    'Orange': ['', 'Limited', 'Limited', 'Free', 'Beginners, Education'],
    'RapidMiner': ['', 'Medium', 'High', '$$$$', 'Enterprises, Automation'],
    'Weka': ['', 'High', 'Medium', 'Free', 'Research, Education'],
    'Python/Sklearn': ['', 'Very High', 'High', 'Free', 'Production, Custom Solutions']
}

df_comparison = pd.DataFrame(comparison_data)
st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# Detailed Comparison
st.subheader(" Detailed Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Orange**")
    st.markdown("""
    - **Strengths:**
      - User-friendly interface
      - Great for beginners
      - Good visualization capabilities
    
    - **Limitations:**
      - Limited flexibility for advanced users
      - Not ideal for production-level code
    """)

    st.markdown("**RapidMiner**")
    st.markdown("""
    - **Strengths:**
      - Strong in automation and deployment
      - Good for big data processing
      - Comprehensive tool for data science lifecycle
    
    - **Limitations:**
      - Can be expensive
      - Steeper learning curve for beginners
    """)

with col2:
    st.markdown("**Weka**")
    st.markdown("""
    - **Strengths:**
      - Extensive collection of machine learning algorithms
      - Excellent for academic and research purposes
      - Free and open-source
    
    - **Limitations:**
      - Less intuitive interface
      - Requires Java knowledge for extensions
    """)

    st.markdown("**Python/Sklearn**")
    st.markdown("""
    - **Strengths:**
      - Highly flexible and customizable
      - Strong community support
      - Industry standard for machine learning
    
    - **Limitations:**
      - Requires programming knowledge
      - Steeper learning curve for non-programmers
    """)

# Use Case Scenarios
st.subheader(" Use Case Scenarios")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Choose Orange if:**")
    st.markdown("""
    - You're new to machine learning
    - Need quick prototypes
    - Want interactive visualizations
    - Working on educational projects
    - Prefer no-code solutions
    """)

    st.markdown("**Choose RapidMiner if:**")
    st.markdown("""
    - Enterprise environment
    - Need automated workflows
    - Budget for commercial tools
    - Require deployment capabilities
    - Working with big data
    """)

with col2:
    st.markdown("**Choose Weka if:**")
    st.markdown("""
    - Academic research
    - Algorithm comparison studies
    - Teaching machine learning
    - Need comprehensive algorithms
    - Budget constraints
    """)

    st.markdown("**Choose Python/Sklearn if:**")
    st.markdown("""
    - Production deployment
    - Maximum flexibility needed
    - Custom algorithms required
    - Integration with other systems
    - Team has programming skills
    """)

# MLOps Best Practices
st.markdown("---")
st.subheader(" MLOps Best Practices")

practices = {
    'Practice': [
        '1. Version Control',
        '2. Experiment Tracking',
        '3. Data Versioning',
        '4. Automated Testing',
        '5. Continuous Integration',
        '6. Model Monitoring',
        '7. Documentation'
    ],
    'Tool/Method': [
        'Git, GitHub, GitLab',
        'MLflow, Weights & Biases',
        'DVC (Data Version Control)',
        'pytest, unittest',
        'GitHub Actions, Jenkins',
        'Prometheus, Grafana',
        'Sphinx, MkDocs'
    ],
    'Benefit': [
        'Track code changes, collaboration',
        'Compare model performance',
        'Reproduce experiments',
        'Catch bugs early',
        'Automated deployment',
        'Detect model degradation',
        'Knowledge sharing'
    ]
}

df_practices = pd.DataFrame(practices)
st.dataframe(df_practices, use_container_width=True, hide_index=True)

# Implementation Example
st.markdown("---")
st.subheader(" MLOps Implementation for NEO Project")

st.code("""
# Example MLOps Workflow for NEO Detection

# 1. Data Versioning with DVC
!dvc init
!dvc add neo_data.csv
!git add neo_data.csv.dvc
!git commit -m "Add NEO dataset"

# 2. Experiment Tracking with MLflow
import mlflow

mlflow.start_run()
mlflow.log_param("model_type", "GradientBoosting")
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", 0.97)
mlflow.sklearn.log_model(gb_model, "model")
mlflow.end_run()

# 3. Model Deployment with FastAPI
from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open('gradient_boosting_neo.pkl', 'rb'))

@app.post("/predict")
def predict(features: dict):
    X = preprocess(features)
    prediction = model.predict(X)
    return {"is_hazardous": bool(prediction[0])}

# 4. Containerization with Docker
# Dockerfile
FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]

# 5. Monitoring
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@prediction_latency.time()
def predict_with_monitoring(features):
    prediction_counter.inc()
    return model.predict(features)
""", language="python")

# Conclusion
st.markdown("---")
st.subheader(" Key Takeaways")

st.success("""
**MLOps Workflow Summary:**

1. **Data Stage:** Collect → Clean → Version → Store
2. **Development Stage:** Experiment → Train → Evaluate → Compare
3. **Deployment Stage:** Package → Deploy → Monitor → Update
4. **Maintenance Stage:** Retrain → A/B Test → Optimize → Scale

**Tool Selection:**
- **Prototyping:** Orange (fast, visual)
- **Enterprise:** RapidMiner (robust, scalable)
- **Research:** Weka (comprehensive algorithms)
- **Production:** Python/Sklearn (flexible, industry standard)

**Our NEO Project Implementation:**
- Used Python/Sklearn for flexibility
- Saved models with pickle (basic versioning)
- Modular architecture (config, data_loader, model_loader)
- Ready for Docker containerization
- Can deploy with FastAPI/Streamlit

**Next Steps for Production:**
1. Implement MLflow for experiment tracking
2. Add DVC for data versioning
3. Create CI/CD pipeline with GitHub Actions
4. Containerize with Docker
5. Set up monitoring with Prometheus
6. Implement A/B testing framework
7. Create comprehensive API documentation
""")

st.markdown("---")
st.info(" **Pro Tip:** Start simple with Git and MLflow, then gradually add more MLOps tools as your project scales.")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This MLOps page demonstrates industry best practices for deploying ML models in production.</p>
    <p>Apply these concepts to scale your NEO detection system! </p>
</div>
""", unsafe_allow_html=True)
