import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model, X_test, y_test):
    # 1. predict
    y_pred = model.predict(X_test)
    
    # 2. indicator
    r = pearsonr(y_test, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n=== 评估结果 ===\nCorrelation (R): {r:.3f}\nRMSE: {rmse:.3f}")
    
    # 3.scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10, c='blue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    plt.xlabel("Observation")
    plt.ylabel("Prediction")
    plt.title(f"Bias Correction (R={r:.2f})")
    plt.legend()
    plt.show()
    
    # 4. SHAP analysis
    print("[-] [分析] 计算 SHAP 特征重要性...")
    try:
        xgb_model = model.named_steps['xgbregressor']
        preprocessor = model.named_steps['standardscaler']
        
        X_scaled = preprocessor.transform(X_test)
        
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_scaled)
        
        plt.figure()
        shap.summary_plot(shap_values, X_scaled, feature_names=list(X_test.columns))
    except Exception as e:
        print(f"[Warning] SHAP analysis error: {e}")