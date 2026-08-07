import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def explain_model():
    model        = pickle.load(open("models/best_model.pkl", "rb"))
    feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))

    df = pd.read_csv("data/historical_data.csv")
    df = df.dropna(subset=feature_cols + ["target_aqi_72h"])
    X  = df[feature_cols]

    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    print("✅ SHAP plot saved to models/shap_summary.png")

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols
    ).sort_values(ascending=False)

    print("\nTop 5 Most Important Features:")
    print(importance.head(5))

if __name__ == "__main__":
    explain_model()