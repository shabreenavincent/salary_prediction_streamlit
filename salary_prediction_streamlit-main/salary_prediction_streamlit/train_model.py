# train_model.py (with stacking ensemble, error analysis, duplicate aggregation, and suggested improvements)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# --- Constants for Model Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
XGB_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 5,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# --- Helper Function for Evaluation ---
def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates and prints performance metrics (MAE, RMSE, R2) for a given model.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance on Test Set:")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   R2 Score: {r2:.4f}")
    return mae, rmse, r2

def train_and_save_all_models(file_path):
    print("✅ Loading data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the CSV is in the correct directory or provide the full path.")
        return
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    if 'salary_in_usd' not in df.columns:
        print("Error: 'salary_in_usd' column not found in the dataset.")
        return

    df = df.drop(['salary', 'salary_currency', 'work_year'], axis=1, errors='ignore')

    # --- START: Aggregate duplicates by averaging salary ---
    initial_rows = len(df)
    
    # Identify columns to group by (all columns except 'salary_in_usd')
    grouping_cols = [col for col in df.columns if col != 'salary_in_usd']
    
    # Aggregate duplicates by averaging 'salary_in_usd'
    df = df.groupby(grouping_cols, as_index=False)['salary_in_usd'].mean()
    
    rows_after_aggregation = len(df)
    if initial_rows > rows_after_aggregation:
        print(f"📊 Aggregated {initial_rows - rows_after_aggregation} duplicate rows. New dataset size: {rows_after_aggregation}")
    else:
        print("No duplicate rows found for aggregation.")
    # --- END: Aggregate duplicates ---

    df["company_size_num"] = df["company_size"].map({'S': 1, 'M': 2, 'L': 3})
    df["is_manager"] = df["job_title"].apply(lambda x: 1 if "manager" in str(x).lower() else 0)

    y = np.log1p(df["salary_in_usd"]) # Log transform the target
    X = df.drop("salary_in_usd", axis=1)

    categorical_cols = X.select_dtypes(include="object").columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("🚀 Training base models...")
    xgb = XGBRegressor(**XGB_PARAMS)
    lr = LinearRegression()
    tree = DecisionTreeRegressor(random_state=RANDOM_STATE)

    xgb.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    tree.fit(X_train, y_train)

    print("🔧 Creating meta-features for stacking...")
    train_meta = pd.DataFrame({
        'xgb': xgb.predict(X_train),
        'lr': lr.predict(X_train),
        'tree': tree.predict(X_train)
    })

    test_meta = pd.DataFrame({
        'xgb': xgb.predict(X_test),
        'lr': lr.predict(X_test),
        'tree': tree.predict(X_test)
    })

    print("🚀 Training meta-model...")
    meta_model = LinearRegression()
    meta_model.fit(train_meta, y_train)

    final_pred_log = meta_model.predict(test_meta)
    final_pred = np.expm1(final_pred_log)
    y_test_real = np.expm1(y_test)

    # Evaluate the stacking ensemble
    evaluate_model(y_test_real, final_pred, "Stacking Ensemble (Meta-model)")

    # --- START: Detailed Error Analysis for Stacking Ensemble ---
    errors = final_pred - y_test_real
    percentage_errors = (errors / y_test_real) * 100

    error_df = pd.DataFrame({
        'Actual_Salary': y_test_real,
        'Predicted_Salary': final_pred,
        'Absolute_Error': np.abs(errors),
        'Signed_Error': errors,
        'Percentage_Error': percentage_errors
    }, index=X_test.index)

    overall_avg_percentage_error = percentage_errors.mean()
    print(f"\nOverall average percentage error (positive = overprediction, negative = underprediction): {overall_avg_percentage_error:.2f}%")

    overpredictions_threshold = 10
    overpredictions_df = error_df[error_df['Percentage_Error'] > overpredictions_threshold]
    print(f"\n🚨 Overpredictions ( > {overpredictions_threshold}% of actual salary):")
    print(f"   Found {len(overpredictions_df)} instances out of {len(y_test_real)} test samples.")
    if not overpredictions_df.empty:
        print(f"   Average percentage overprediction for these errors: {overpredictions_df['Percentage_Error'].mean():.2f}%")
        print("\nTop 10 largest *overpredictions* (actual vs. predicted):")
        print(overpredictions_df.sort_values(by='Signed_Error', ascending=False).head(10))


    underpredictions_threshold = -10
    underpredictions_df = error_df[error_df['Percentage_Error'] < underpredictions_threshold]
    print(f"\n🔻 Underpredictions ( < {abs(underpredictions_threshold)}% of actual salary):")
    print(f"   Found {len(underpredictions_df)} instances out of {len(y_test_real)} test samples.")
    if not underpredictions_df.empty:
        print(f"   Average percentage underprediction for these errors: {underpredictions_df['Percentage_Error'].mean():.2f}%")
        print("\nTop 10 largest *underpredictions* (actual vs. predicted):")
        print(underpredictions_df.sort_values(by='Signed_Error', ascending=True).head(10))
    # --- END: Detailed Error Analysis ---

    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, os.path.join("models", "xgb_model.pkl"))
    joblib.dump(lr, os.path.join("models", "lr_model.pkl"))
    joblib.dump(tree, os.path.join("models", "tree_model.pkl"))
    joblib.dump(meta_model, os.path.join("models", "meta_model.pkl"))
    joblib.dump(encoders, os.path.join("models", "encoders.pkl"))

    print("\n✅ Models (base and meta) & encoders saved in models/")

if __name__ == "__main__":
    # Ensure this path matches the location of your uploaded file
    # This path was updated based on your previous input.
    train_and_save_all_models(r"C:\salary_prediction_streamlit\salary_prediction_streamlit\data\DataScience_salaries_2025.csv")
