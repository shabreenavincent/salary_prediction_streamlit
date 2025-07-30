# app.py (Shortform mapping fixed, enhanced About/Help, all previous fixes/suggestions retained)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import os
from streamlit_lottie import st_lottie
import matplotlib.ticker as mticker # For formatting axis ticks

st.set_page_config(page_title="Salary Predictor App", page_icon="💰", layout="wide")

# --- Constants for Paths ---
MODELS_DIR = "models"
ANIMATION_PATH = "assets/animation.json"
JOB_TITLES_PATH = "assets/job_titles.json"

# --- Global Constants for Bias Correction and Currency Conversion ---
# Adjust this factor based on your precise observed average overprediction
# from train_model.py (e.g., if overall_avg_percentage_error is 18.55%, factor is 1.1855)
# This value should ideally be derived from train_model.py output and matched here.
BIAS_CORRECTION_FACTOR = 1.1855 

# Suggestion: For a production app, fetch real-time INR conversion rate using an API.
INR_CONVERSION_RATE = 83.2

# --- Mappings for User-Friendly Selectboxes ---
EXPERIENCE_LEVEL_MAP = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior-level",
    "EX": "Executive-level"
}

EMPLOYMENT_TYPE_MAP = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance"
}

COMPANY_SIZE_MAP = {
    "S": "Small (less than 50 employees)",
    "M": "Medium (50 to 250 employees)",
    "L": "Large (more than 250 employees)"
}

# Create reverse mappings for convenience
REV_EXPERIENCE_LEVEL_MAP = {v: k for k, v in EXPERIENCE_LEVEL_MAP.items()}
REV_EMPLOYMENT_TYPE_MAP = {v: k for k, v in EMPLOYMENT_TYPE_MAP.items()}
REV_COMPANY_SIZE_MAP = {v: k for k, v in COMPANY_SIZE_MAP.items()}


# Load models & encoders
try:
    xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
    tree_model = joblib.load(os.path.join(MODELS_DIR, "tree_model.pkl"))
    meta_model = joblib.load(os.path.join(MODELS_DIR, "meta_model.pkl")) # Load the meta-model
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
except FileNotFoundError as e:
    st.error(f"Error loading model files. Please ensure the '{MODELS_DIR}' directory exists and contains all required .pkl files. Run train_model.py first. Details: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading models. Details: {e}")
    st.stop()


# Load animation
@st.cache_data # Cache the animation loading to avoid re-loading on every rerun
def load_animation(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Animation file not found at {filepath}. Please ensure 'assets/animation.json' exists.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file's content is valid JSON.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading animation: {e}")
        return None

animation = load_animation(ANIMATION_PATH)

# Load job titles
@st.cache_data # Cache the job titles loading
def load_job_titles(filepath):
    try:
        with open(filepath, "r") as f:
            # IMPORTANT: Ensure 'assets/job_titles.json' contains a comprehensive list of common job titles
            # tailored to your target audience for better user experience.
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Job titles file not found at {filepath}. Please ensure 'assets/job_titles.json' exists and contains a list of job titles.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from {filepath}. Please check the file's content is valid JSON.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading job titles: {e}")
        return []

job_titles = load_job_titles(JOB_TITLES_PATH)

# --- Helper function for preprocessing input data ---
def preprocess_input(df_input, encoders):
    """Applies feature engineering and label encoding to input DataFrame."""
    # Ensure columns for feature engineering exist
    if "company_size" in df_input.columns:
        df_input["company_size_num"] = df_input["company_size"].map({'S': 1, 'M': 2, 'L': 3})
    else:
        st.warning("Missing 'company_size' column for feature engineering.")
        df_input["company_size_num"] = 2 # Default to M if missing

    if "job_title" in df_input.columns:
        df_input["is_manager"] = df_input["job_title"].apply(lambda x: 1 if "manager" in str(x).lower() else 0)
    else:
        st.warning("Missing 'job_title' column for feature engineering.")
        df_input["is_manager"] = 0 # Default to not manager


    # Apply Label Encoding
    for col, le in encoders.items():
        if col in df_input.columns:
            # Handle unseen categories robustly:
            # Convert to list to use le.transform, then replace unknown with -1 (or other default)
            # This ensures that unseen values don't cause the app to crash.
            # It's important that the model understands what -1 means for these features.
            transformed_values = []
            for val in df_input[col]:
                if val in le.classes_:
                    transformed_values.append(le.transform([val])[0])
                else:
                    transformed_values.append(-1) # Assign -1 for unknown/unseen category
                    st.warning(f"Found unknown category '{val}' in column '{col}'. It has been encoded as -1. This might affect prediction accuracy.")
            df_input[col] = transformed_values
        else:
            st.warning(f"Column '{col}' expected by encoder not found in input data. Please check your data schema.")
            # If a critical column for the model is missing, you might want to stop or fill with a default.
            # For now, just a warning.
    return df_input


# Sidebar
st.sidebar.title("⚙️ Settings")
currency = st.sidebar.radio("Select currency", ("USD", "INR"))

# Updated tabs list
tabs = st.tabs(["🔮 Predict Salary", "📥 Batch Prediction", "ℹ️ About / Help"])

with tabs[0]:
    st.title("🔮 Predict Salary")
    col1, col2, col3 = st.columns(3)

    with col1:
        # User-friendly display names for experience level
        selected_experience_level = st.selectbox(
            "Experience Level",
            options=list(EXPERIENCE_LEVEL_MAP.values()),
            format_func=lambda x: x # Use the full string as is for display
        )
        # Map selected full form back to short form for model input
        experience_level_code = REV_EXPERIENCE_LEVEL_MAP.get(selected_experience_level)

        # User-friendly display names for employment type
        selected_employment_type = st.selectbox(
            "Employment Type",
            options=list(EMPLOYMENT_TYPE_MAP.values()),
            format_func=lambda x: x
        )
        employment_type_code = REV_EMPLOYMENT_TYPE_MAP.get(selected_employment_type)

        # User-friendly display names for company size
        selected_company_size = st.selectbox(
            "Company Size",
            options=list(COMPANY_SIZE_MAP.values()),
            format_func=lambda x: x
        )
        company_size_code = REV_COMPANY_SIZE_MAP.get(selected_company_size)


    with col2:
        job_title = st.selectbox("Job Title", job_titles)
        remote_ratio = st.slider("Remote Ratio (%)", 0, 100, 100)

    with col3:
        employee_residence = st.text_input("Employee Residence (e.g., US, IN)", "US")
        company_location = st.text_input("Company Location (e.g., US, IN)", "US")

    if st.button("✅ Predict Salary"):
        # Create a DataFrame for a single prediction using the mapped codes
        input_data = {
            "experience_level": [experience_level_code],
            "employment_type": [employment_type_code],
            "job_title": [job_title],
            "employee_residence": [employee_residence],
            "remote_ratio": [remote_ratio],
            "company_location": [company_location],
            "company_size": [company_size_code]
        }
        input_df = pd.DataFrame(input_data)

        # Preprocess the input DataFrame using the helper function
        processed_input_df = preprocess_input(input_df.copy(), encoders)

        # Predict with base models to create meta-features
        try:
            pred_log_xgb = xgb_model.predict(processed_input_df)[0]
            pred_log_lr = lr_model.predict(processed_input_df)[0]
            pred_log_tree = tree_model.predict(processed_input_df)[0]
        except Exception as e:
            st.error(f"Error during base model prediction. Check if input features match training features. Details: {e}")
            st.stop()

        # --- CRITICAL CORRECTION: Use the meta_model for stacking ensemble prediction ---
        # Create meta-features from base model predictions for the meta-model
        input_meta = pd.DataFrame({
            'xgb': [pred_log_xgb],
            'lr': [pred_log_lr],
            'tree': [pred_log_tree]
        })
        # Use the trained meta-model for final prediction
        try:
            final_pred_log = meta_model.predict(input_meta)[0]
        except Exception as e:
            st.error(f"Error during meta-model prediction. Details: {e}")
            st.stop()
        # --- END CRITICAL CORRECTION ---

        pred_usd = np.expm1(final_pred_log)

        # *** Bias correction for consistent overprediction ***
        pred_usd_corrected = pred_usd / BIAS_CORRECTION_FACTOR
        # *** END: Bias correction ***

        salary = int(round(pred_usd_corrected * INR_CONVERSION_RATE)) if currency == "INR" else int(round(pred_usd_corrected))

        st.success(f"💰 Predicted Salary: {salary:,} {currency}")

        # --- START: Improved Plotting for Single Prediction ---
        fig, ax = plt.subplots(figsize=(7, 4)) # Adjusted figure size for shorter plot
        bar_color = "#28a745" # A pleasant green color

        # Create the bar
        ax.bar(["Predicted Salary"], [salary], color=bar_color, width=0.5, edgecolor='black', linewidth=0.8)

        # Add the exact value on top of the bar
        ax.text("Predicted Salary", salary, f'{salary:,} {currency}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

        # Set title and labels with improved font sizes
        ax.set_title(f"Predicted Salary: {currency}", fontsize=16, fontweight='bold')
        ax.set_ylabel(f"Salary ({currency})", fontsize=12)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Format Y-axis to prevent scientific notation and use comma separators
        # Handle currency symbol for formatting
        if currency == "INR":
             ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('₹{x:,.0f}'))
        else: # Default to USD
             ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}')) 
        
        # Add a subtle grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Remove spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        plt.tight_layout() # Adjust layout to prevent labels overlapping
        st.pyplot(fig)
        # --- END: Improved Plotting for Single Prediction ---

with tabs[1]:
    st.title("📥 Batch Prediction")
    # Provide a template CSV for users
    template_data = {
        "experience_level": ["MI", "SE"],
        "employment_type": ["FT", "FT"],
        "job_title": ["Data Scientist", "Lead Data Scientist"],
        "employee_residence": ["US", "GB"],
        "remote_ratio": [100, 0],
        "company_location": ["US", "GB"],
        "company_size": ["M", "L"]
    }
    template_df = pd.DataFrame(template_data)
    csv_template = template_df.to_csv(index=False).encode()
    st.download_button("⬇ Download template CSV", csv_template, "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload your batch CSV", type=["csv"])
    if uploaded and st.button("📍 Predict salaries"):
        try:
            df_batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading uploaded CSV file. Please ensure it's a valid CSV. Details: {e}")
            st.stop()

        # Preprocess the batch DataFrame
        processed_df_batch = preprocess_input(df_batch.copy(), encoders)

        # Predict with base models
        try:
            pred_xgb_batch = xgb_model.predict(processed_df_batch)
            pred_lr_batch = lr_model.predict(processed_df_batch)
            pred_tree_batch = tree_model.predict(processed_df_batch)
        except Exception as e:
            st.error(f"Error during batch base model prediction. Check if input features match training features. Details: {e}")
            st.stop()
        
        # --- CRITICAL CORRECTION: Use the meta_model for stacking ensemble prediction ---
        # Create meta-features from base model predictions for the meta-model
        batch_meta = pd.DataFrame({
            'xgb': pred_xgb_batch,
            'lr': pred_lr_batch,
            'tree': pred_tree_batch
        })
        # Use the trained meta-model for final prediction
        try:
            final_pred_log_batch = meta_model.predict(batch_meta)
        except Exception as e:
            st.error(f"Error during batch meta-model prediction. Details: {e}")
            st.stop()
        # --- END CRITICAL CORRECTION ---

        preds_usd_raw = np.expm1(final_pred_log_batch)

        # *** Bias correction for consistent overprediction in batch ***
        preds_usd_corrected_batch = preds_usd_raw / BIAS_CORRECTION_FACTOR
        # *** END: Bias correction ***

        # Add predicted salaries to the original DataFrame
        df_batch["Predicted Salary"] = np.round(preds_usd_corrected_batch * INR_CONVERSION_RATE).astype(int) if currency == "INR" else np.round(preds_usd_corrected_batch).astype(int)
        
        st.subheader("Batch Predictions")
        st.dataframe(df_batch)

        # --- START: Plotting Options for Batch Prediction ---
        if not df_batch.empty:
            plot_type = st.radio("Select Plot Type", ("Histogram", "Box Plot"), key="batch_plot_type")

            if plot_type == "Histogram":
                fig_batch_chart, ax_batch_chart = plt.subplots(figsize=(10, 6)) # Adjusted figure size for shorter plot
                ax_batch_chart.hist(df_batch["Predicted Salary"], bins=25, color="#007bff", edgecolor='black', alpha=0.8) # Blue color

                mean_salary = df_batch["Predicted Salary"].mean()
                ax_batch_chart.axvline(mean_salary, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_salary:,.0f} {currency}')

                ax_batch_chart.set_title(f"Distribution of Predicted Salaries ({currency})", fontsize=18, fontweight='bold')
                ax_batch_chart.set_xlabel(f"Predicted Salary ({currency})", fontsize=14)
                ax_batch_chart.set_ylabel("Number of Employees", fontsize=14)

                if currency == "INR":
                    ax_batch_chart.xaxis.set_major_formatter(mticker.StrMethodFormatter('₹{x:,.0f}'))
                else: # Default to USD
                    ax_batch_chart.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
                ax_batch_chart.tick_params(axis='x', labelsize=12)
                ax_batch_chart.tick_params(axis='y', labelsize=12)
                
                ax_batch_chart.grid(axis='y', linestyle='--', alpha=0.7)
                ax_batch_chart.grid(axis='x', linestyle=':', alpha=0.5)
                ax_batch_chart.legend(fontsize=12)
                ax_batch_chart.spines['top'].set_visible(False)
                ax_batch_chart.spines['right'].set_visible(False)
                ax_batch_chart.spines['left'].set_linewidth(0.5)
                ax_batch_chart.spines['bottom'].set_linewidth(0.5)

            elif plot_type == "Box Plot":
                fig_batch_chart, ax_batch_chart = plt.subplots(figsize=(9, 5)) # Adjusted figure size for shorter plot
                box_color = "darkgreen"

                ax_batch_chart.boxplot(df_batch["Predicted Salary"], vert=False, patch_artist=True,
                                    boxprops=dict(facecolor=box_color, edgecolor='black'),
                                    medianprops=dict(color='red', linewidth=2),
                                    whiskerprops=dict(color='black'),
                                    capprops=dict(color='black'),
                                    flierprops=dict(marker='o', markersize=5, markerfacecolor='red', markeredgecolor='black'))

                ax_batch_chart.set_title(f"Box Plot of Predicted Salaries ({currency})", fontsize=18, fontweight='bold')
                ax_batch_chart.set_xlabel(f"Predicted Salary ({currency})", fontsize=14)
                ax_batch_chart.set_yticks([]) # Hide y-axis ticks as it's a single box plot

                if currency == "INR":
                    ax_batch_chart.xaxis.set_major_formatter(mticker.StrMethodFormatter('₹{x:,.0f}'))
                else:
                    ax_batch_chart.xaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))

                ax_batch_chart.tick_params(axis='x', labelsize=12)
                ax_batch_chart.grid(axis='x', linestyle='--', alpha=0.7)
                ax_batch_chart.spines['top'].set_visible(False)
                ax_batch_chart.spines['right'].set_visible(False)
                ax_batch_chart.spines['left'].set_visible(False) # Hide left spine for box plot
                
            plt.tight_layout()
            st.pyplot(fig_batch_chart)

            st.markdown("""
            **Note on other plot types (KDE, Violin):**
            Kernel Density Estimate (KDE) and Violin plots provide more detailed insights into salary distributions.
            They are often best created using the `seaborn` library (e.g., `import seaborn as sns`).
            If you install `seaborn` (`pip install seaborn`), you could add options for these plots.
            """)
        else:
            st.info("No data to display in batch prediction chart. Please upload a CSV with valid data.")
        # --- END: Plotting Options for Batch Prediction ---


        # Provide download button for predictions
        csv_predictions = df_batch.to_csv(index=False).encode()
        st.download_button("⬇ Download predictions", csv_predictions, "batch_predictions.csv", "text/csv")
    else:
        st.info("Upload your CSV file above and click 'Predict salaries' to get batch predictions.")

with tabs[2]: # This is the original About/Help tab, now moved to index 2
    st.title("ℹ️ About / Help")
    if animation: # Only display if animation loaded successfully
        st_lottie(animation, speed=1, height=300)
    st.markdown(f"""
    ---
    ### 👋 Welcome to the Salary Predictor App!
    This application helps you predict employee salaries based on various professional attributes. It uses a sophisticated machine learning model to provide accurate salary estimations for individuals or entire datasets.

    ---
    ### 🚀 How to Use

    #### 🔮 Predict Salary (Single Prediction)
    1.  **Select Currency:** Choose between USD (United States Dollar) or INR (Indian Rupee) from the sidebar.
    2.  **Input Details:** Fill in the requested information for the employee, including:
        * **Experience Level:** Select from options like "Entry-level," "Mid-level," "Senior-level," or "Executive-level."
        * **Employment Type:** Choose "Full-time," "Part-time," "Contract," or "Freelance."
        * **Company Size:** Indicate if the company is "Small," "Medium," or "Large."
        * **Job Title:** Select the relevant job title from the dropdown. *(Note: If you feel important job titles are missing, please update the `assets/job_titles.json` file in your project directory.)*
        * **Remote Ratio:** Use the slider to specify the percentage of remote work (0% to 100%).
        * **Employee Residence & Company Location:** Enter the 2-letter country code (e.g., "US" for United States, "IN" for India).
    3.  **Click "Predict Salary":** The app will display the predicted salary and a clear bar chart visualization.

    #### 📥 Batch Prediction
    1.  **Download Template CSV:** Click the "⬇ Download template CSV" button to get a sample CSV file. This file shows the required column headers and expected data format.
    2.  **Prepare Your Data:** Populate the template CSV with your employee data. Ensure column names match the template exactly.
    3.  **Upload Your CSV:** Click "Browse files" to upload your prepared CSV file.
    4.  **Click "Predict salaries":** The app will process your file, display the predictions in a table, and show a plot of the salary distribution. You can choose between a Histogram or a Box Plot for visualization to understand the salary spread.
    5.  **Download Predictions:** Click "⬇ Download predictions" to save the results, including the predicted salaries, to a new CSV file.

    ---
    ### 🧠 Model Details
    This application utilizes an advanced machine learning approach: a **stacking ensemble model**.
    * **How it Works:** It combines the strengths of three individual (base) models: XGBoost, Linear Regression, and Decision Tree. A separate "meta-model" (Linear Regression) then learns how to best weigh and combine the predictions from these base models to produce a more robust and accurate final prediction.
    * **Input Features:** The model considers: Experience Level, Employment Type, Job Title, Employee Residence, Remote Ratio, Company Location, and Company Size.
    * **Output:** The predicted salary is provided in your chosen currency (USD or INR).

    **Important Note on Overprediction Correction:**
    During the model training process, we observed a consistent tendency for the model to slightly overpredict salaries. To counteract this, a post-prediction bias correction is applied. A specific correction factor of `{BIAS_CORRECTION_FACTOR}` is used to scale down the final predictions, bringing them closer to real-world values. It's important to update this factor if you retrain the model and observe a different average percentage error using `train_model.py`.

    ---
    ### 💡 Troubleshooting & Tips
    * **"Error loading model files":** This usually means you haven't run `train_model.py` yet, or the generated `models` folder is not in the same directory as your `app.py`. Ensure `train_model.py` executes successfully to create all `.pkl` model files.
    * **"Unknown category" warnings:** If your input data (either from single prediction or batch upload) contains values for categorical features (like Job Title, Country Codes) that were *not* present in the dataset used for model training, the app will issue a warning and encode these as `-1`. This might affect prediction accuracy. For best results, try to use categories present in your original `DataScience_salaries_2025.csv` file.
    * **Inaccurate INR Conversion:** The INR conversion rate is currently a fixed value. For the most up-to-date and precise predictions in INR, consider integrating a real-time currency exchange API in a production version of the app.
    * **Job Titles are Missing/Unfamiliar:** The list of job titles comes from `assets/job_titles.json`. If you need more common or specific titles, you can manually edit this JSON file to include them.

    ---
    *Made with ❤️ by your team.*
    """)
