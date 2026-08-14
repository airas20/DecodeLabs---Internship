
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

# -----------------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------------
CSV_PATH = "Dataset_for_Data_Analytics_-_Sheet1.csv"
TARGET_COLUMN = "OrderStatus"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_K_TO_TEST = 30
OUTPUT_DIR = "outputs"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_inspect_data(path: str) -> pd.DataFrame:
    """INPUT stage: load the raw CSV and print a quick data inventory."""
    df = pd.read_csv(path)

    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nTarget class balance ({TARGET_COLUMN}):")
    print(df[TARGET_COLUMN].value_counts())
    print()
    return df


def engineer_features(df: pd.DataFrame):
    
    data = df.copy()

    data["HasCoupon"] = data["CouponCode"].notna().astype(int)

    data["OrderMonth"] = pd.to_datetime(data["Date"]).dt.month

    drop_cols = [
        "OrderID", "Date", "CustomerID", "ShippingAddress",
        "TrackingNumber", "CouponCode", TARGET_COLUMN,
    ]
    X = data.drop(columns=drop_cols)

    categorical_cols = ["Product", "PaymentMethod", "ReferralSource"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data[TARGET_COLUMN])

   
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")
    print(f"Target classes: {list(label_encoder.classes_)}")
    print()

    return X, y, label_encoder


def split_and_scale(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")
    print()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def find_optimal_k(X_train, y_train, X_test, y_test, max_k=MAX_K_TO_TEST):
   
    error_rates = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        error_rates.append(np.mean(preds != y_test))

    best_k = k_range[int(np.argmin(error_rates))]

    plt.figure(figsize=(9, 5))
    plt.plot(k_range, error_rates, marker="o", markersize=4,
              linestyle="--", color="#1f4e79")
    plt.axvline(best_k, color="orange", linestyle=":", label=f"Optimal K = {best_k}")
    plt.title("Tuning the Engine: Choosing K (Elbow Method)")
    plt.xlabel("K Value")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/elbow_curve.png", dpi=150)
    plt.close()

    
    print(f"Best K found: {best_k}  (lowest test error = {min(error_rates):.4f})")
    print(f"Elbow curve saved to {OUTPUT_DIR}/elbow_curve.png")
    print()

    return best_k


def train_and_evaluate(X_train, y_train, X_test, y_test, k, label_encoder):
   
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    
    print(f"Model: KNeighborsClassifier(n_neighbors={k})")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

   
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix — KNN Order Status Classifier")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {OUTPUT_DIR}/confusion_matrix.png")

    return model, acc, f1_macro


def main():
    df = load_and_inspect_data(CSV_PATH)
    X, y, label_encoder = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    best_k = find_optimal_k(X_train, y_train, X_test, y_test)
    model, acc, f1_macro = train_and_evaluate(
        X_train, y_train, X_test, y_test, best_k, label_encoder
    )

    print("=" * 70)
    print("PROJECT 2 COMPLETE")
    print("=" * 70)
    print(f"Final Accuracy : {acc:.2%}")
    print(f"Final F1 Score : {f1_macro:.4f}")
    print(f"Outputs saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
