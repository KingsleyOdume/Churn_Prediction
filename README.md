📊 Customer Churn Prediction Model  

A machine learning project that predicts whether a telecom customer is likely to churn (leave the company) or stay.  
This helps businesses identify at-risk customers** and take proactive retention steps.  

---

🔑 Project Highlights  
- ✅ Cleaned and analyzed **5,000+ customer records**  
- ✅ Built and tuned ML classification models with **85% accuracy**  
- ✅ Interactive **Streamlit demo** for live testing  
- ✅ Outcome: Helps businesses **reduce churn & improve customer retention**  

---

 🛠 Tech Stack  
- Python 🐍  
- Pandas & NumPy (data wrangling)  
- Matplotlib & Seaborn (visualization)  
- Scikit-learn (ML models)  
- Streamlit (interactive demo)  

---

 📂 Project Structure  

📁 telecom-churn-prediction
│── app.py # Streamlit demo app
│── churn_model.ipynb # Jupyter Notebook (EDA + model training)
│── telecom_churn.csv # Dataset
│── requirements.txt # Dependencies
│── README.md # Project documentation



---

🚀 How to Run Locally  

1. Clone the repo:  
   
   git@github.com:KingsleyOdume/Churn_Prediction.git
   cd Churn_Prediction


Create virtual environment & install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Open browser at http://localhost:8501

🌐 Live Demo
👉 Try it here on Streamlit


📊 Example Prediction
Input:
Gender: Female
Tenure: 12 months
Contract: Month-to-month
Monthly Charges: $70
Total Charges: $1200

Output:
⚠️ This customer is likely to CHURN. (Risk: 78%)

📈 Model Performance
Accuracy: 85%
Precision: 0.83
Recall: 0.81
F1 Score: 0.82
