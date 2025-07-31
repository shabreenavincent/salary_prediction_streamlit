# 💼 Salary Prediction App (Streamlit)

This is a Machine Learning-based web application built using **Streamlit** that predicts employee salaries (in USD and INR) based on job title, company location, experience level, and other relevant features. The backend is powered by trained ML models like Linear Regression, Decision Tree, and Random Forest, with Random Forest selected as the best performer.

---

## 🚀 Features

- Predicts salary in **USD** and auto-converts to **INR**
- Easy-to-use **Streamlit** interface
- Accepts user input for job-related features
- Compares predictions across multiple ML models
- Includes **exploratory data analysis**, **feature engineering**, and **model evaluation**
- Built with **VS Code**, deployed via **GitHub**, and powered by **Python**

---

## 📂 Tech Stack

- **Python**
- **Pandas, NumPy, Scikit-learn**
- **Matplotlib, Seaborn**
- **Streamlit**
- **Git & GitHub**

---

## 🧠 ML Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor ✅ (best performance)

Each model was trained and tested on a global salary dataset, and evaluated using metrics like MAE, MSE, and R² Score.

---

## 📷 Screenshot

<img width="1920" height="1080" alt="Screenshot 2025-07-28 232335" src="https://github.com/user-attachments/assets/e3772f7e-000b-4cc9-81fe-a22f550a35c9" />
<img width="1920" height="1080" alt="Screenshot 2025-07-28 232345" src="https://github.com/user-attachments/assets/e4226dab-6318-46a7-bfe6-e8ce61951b26" />
<img width="1920" height="1080" alt="Screenshot 2025-07-28 232547" src="https://github.com/user-attachments/assets/9a844662-9f8a-4fcd-9405-2c5819cb86a4" />
<img width="1920" height="1080" alt="Screenshot 2025-07-28 232519" src="https://github.com/user-attachments/assets/08e11ba6-a5dd-4e98-936c-f552459a5970" />
<img width="1920" height="1080" alt="Screenshot 2025-07-28 232614" src="https://github.com/user-attachments/assets/14bc02a5-dad4-4596-8e27-e5b0035892a2" />
<img width="1920" height="1080" alt="Screenshot 2025-07-28 232650" src="https://github.com/user-attachments/assets/c03466b4-d881-4659-be9e-259b03822bd2" />


---

## 📊 Dataset Used

- Dataset: **Global Data Science Job Salaries (2020–2025)**
- Source: [kaggle.com/datasets](https://www.kaggle.com/datasets)
- Features: Job title, experience level, company size, location, employment type, salary in USD, etc.

---

## 📁 Project Structure

```
SALARY_PREDICTION_STREAMLIT/
│
├── salary_prediction_streamlit/       # Main project folder
│   ├── __pycache__/                    # Cache for this folder
│   │
│   ├── assets/                         # UI assets and templates
│   │   ├── animation.json              # Lottie animation file
│   │   ├── job_titles.json             # Job title reference data
│   │   └── template.csv                # Data template for download/upload
│   │
│   ├── data/                           # Dataset folder
│   │   └── DataScience_salaries_2025.csv
│   │
│   ├── models/                         # Trained model files
│   │   ├── encoders.pkl                # Label encoders
│   │   ├── lr_model.pkl                # Linear Regression model
│   │   ├── meta_model.pkl              # Stacking meta-learner
│   │   ├── tree_model.pkl              # Decision Tree model
│   │   └── xgb_model.pkl               # XGBoost model
│   │
│   ├── app.py                          # Streamlit web application
│   ├── train_model.py                  # ML model training script
│   ├── requirements.txt                # Python dependencies
│   ├── text.txt                        # Possibly temp notes or logs
│   └── README.md                       # Project documentation
```


---

## 🛠️ How to Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/shabreenavincent/salary_prediction_streamlit.git
   cd salary_prediction_streamlit
   ```
## Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Launch the Streamlit app:

   ```bash
streamlit run app.py
   ```

## Acknowledgements

Project by Shabreena Vincent, AIDS department, Saveetha Engineering College

Special thanks to mentors and online communities supporting ML education

## Contact

For questions or collaborations, feel free to connect:

📧 Email: shabs1162@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/shabreena-vincent-a748052b2/
   
