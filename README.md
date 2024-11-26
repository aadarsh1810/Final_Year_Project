# Supplier News Categorization System

This project leverages **Machine Learning (ML)** and **MLOps principles** to automate the categorization of supplier-related news articles into predefined categories such as Financial, Geopolitical, Natural Disasters, and Regulatory. The system is deployed using **Streamlit** with **Docker** for an interactive, web-based user experience.

---

## 🚀 **Project Objectives**

1. Automate the categorization of supplier news articles to reduce manual effort.
2. Provide real-time data processing and categorization.
3. Enable scalability and reliability through cloud deployment and MLOps practices.
4. Implement robust data security and ensure adaptability through automated retraining pipelines.

---

## 📊 **Project Workflow**

1. **Data Collection**: News articles are scraped from multiple sources and stored securely in **Azure Blob Storage**.
2. **Data Preprocessing**: Text data is cleaned, tokenized, and transformed for model training.
3. **Model Training**: Various machine learning models, including **BERT**, are trained and validated.
4. **Deployment**: The model is containerized using **Docker** and deployed with **Streamlit** for interactive use.
5. **Monitoring & Retraining**: Model performance is monitored using **Azure Application Insights**, with automated retraining pipelines for continuous improvement.

---

## 🛠 **Tools and Technologies**

### **Machine Learning**
- **NLP Models**: BERT, SVM
- **Libraries**: PyTorch, Scikit-learn, Transformers

### **MLOps and Deployment**
- **Docker**: For containerization
- **Streamlit**: Web-based interactive interface
- **Azure**:
  - Azure DevOps: CI/CD pipeline
  - Azure Blob Storage: Data storage
  - Azure Application Insights: Monitoring

---

## 🛠 **Features of the CI/CD Pipeline**
1. **Continuous Integration (CI)**:
   - Code hosted on Azure DevOps Repos with version control.
   - Automated unit testing and model validation triggered on every commit.
2. **Continuous Deployment (CD)**:
   - Containerized model deployed on Streamlit via Azure Virtual Machines.
   - Real-time monitoring and alerts for performance deviations.

---

## 🔑 **Key Challenges and Solutions**
- **Challenge**: Handling large data volumes and ensuring real-time categorization.  
  **Solution**: Scalable storage on Azure Blob and efficient Streamlit deployment.  
- **Challenge**: Data drift affecting model accuracy.  
  **Solution**: Automated retraining pipelines using Azure Machine Learning.

---

## ✅ **Expected Outcomes**
- Real-time categorization of supplier news with reduced manual effort.
- Scalable and secure deployment on Azure.
- Enhanced accuracy and adaptability through continuous monitoring and retraining.

---

