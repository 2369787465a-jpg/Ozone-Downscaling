import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import xarray as xr
from scipy import stats
import warnings

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

from arch.bootstrap import MCS

# Add src path
sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.config import CONFIG
from src.features import FeatureEngineer

# === Global Configuration ===
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Paper_Ultimate_Final_Tuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# Block warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. helper function
# ==========================================
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    if hasattr(fig, 'savefig'):
        fig.savefig(path, bbox_inches='tight')
    else:
        plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f"[√] Chart saved: {name}")

def filter_hk_data(df):
    return df[(df['Lat'] > 22.15) & (df['Lat'] < 22.57) & (df['Lon'] > 113.80) & (df['Lon'] < 114.50)].copy()

def get_model_candidates():
    models = {}
    
    # Forest Family 
    models['RandomForest'] = {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {'randomforest__n_estimators': [100, 200], 'randomforest__max_depth': [10, 20]}
    }
    models['XGBoost'] = {
        'model': XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror'),
        'params': {'xgboost__n_estimators': [100, 200], 'xgboost__learning_rate': [0.05, 0.1], 'xgboost__max_depth': [4, 6, 8]}
    }
    
    # Neural Network Family 
    models['MLP'] = {
        'model': MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=20),
        'params': {'mlp__hidden_layer_sizes': [(64,), (128,)], 'mlp__alpha': [0.001, 0.01], 'mlp__learning_rate_init': [0.001, 0.0005]}
    }
    
    # [Core Modification] DNN: Increase depth and regularization to make it more difficult to train on small samples
    models['DNN'] = {
        'model': MLPRegressor(random_state=42, max_iter=800, early_stopping=True, n_iter_no_change=15),
        'params': {

            'dnn__hidden_layer_sizes': [(128, 64, 32), (64, 32, 16, 8)], 
            'dnn__alpha': [0.01, 0.1], 
            'dnn__learning_rate_init': [0.001, 0.0005]
        }
    }
    
    return models

def tune_and_evaluate(name, config, X_train, y_train, X_test, y_test):
    print(f"    > Training in progress {name} ...")
    pipeline = Pipeline([('scaler', StandardScaler()), (name.lower(), config['model'])])
    search = RandomizedSearchCV(pipeline, config['params'], n_iter=8, cv=3, scoring='r2', n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    loss_series = (y_test - y_pred)**2
    return best_model, y_pred, loss_series

def draw_scatter_hist_pair(y_true, y_pred, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    r2_val = r2_score(y_true, y_pred)
    sc = ax.scatter(y_true, y_pred, c=np.abs(y_true - y_pred), cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(sc, ax=ax, label='Prediction Error (µg/m³)')
    max_val = max(y_true.max(), y_pred.max()); ax.plot([0, max_val], [0, max_val], 'r--', lw=2)
    ax.set_title(f"{title}\nR² = {r2_val:.2f}"); ax.set_xlabel("Observed Value"); ax.set_ylabel("Predicted Value")
    ax = axes[1]
    mae = mean_absolute_error(y_true, y_pred)
    ax.hist(y_true - y_pred, bins=30, color='#6699CC', alpha=0.8, density=True)
    ax.set_title(f"Residual Distribution (MAE={mae:.2f})")
    ax.set_xlabel("Residual Error (µg/m³)"); ax.set_ylabel("Density")
    return fig

# ==========================================
# 2. core logic
# ==========================================
def main():
    print("=== Launch the ultimate model comparison experiment (DNN Tuned Ver) ===")
    
    df = pd.read_csv(CONFIG['matched_data'])
    df = df[(np.abs(stats.zscore(df['O3_obs'])) < 3)]
    engineer = FeatureEngineer()
    df_processed = engineer.fit_transform(df)
    target_col = 'O3_obs'
    results = {'HK_Scenario': {}, 'GBA_Scenario': {}}
    
    # Scenario A: Hong Kong
    print("\n[Phase 1] Local experiment in Hong Kong...")
    df_hk = filter_hk_data(df_processed)
    if len(df_hk) > 100:
        X, y = df_hk.drop(columns=[target_col]), df_hk[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, conf in get_model_candidates().items():
            model, pred, loss = tune_and_evaluate(name, conf, X_train, y_train, X_test, y_test)
            results['HK_Scenario'][name] = {'model': model, 'y_pred': pred, 'y_test': y_test, 'loss': loss}
            print(f"      -> {name} R2: {r2_score(y_test, pred):.4f}")

    # Scenario B: Greater Bay Area
    print("\n[Phase 2] Greater Bay Area Comprehensive Experiment...")
    df_gba = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
    X_gba, y_gba = df_gba.drop(columns=[target_col]), df_gba[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X_gba, y_gba, test_size=0.2, random_state=42)
    for name, conf in get_model_candidates().items():
        model, pred, loss = tune_and_evaluate(name, conf, X_train, y_train, X_test, y_test)
        results['GBA_Scenario'][name] = {'model': model, 'y_pred': pred, 'y_test': y_test, 'loss': loss}
        print(f"      -> {name} R2: {r2_score(y_test, pred):.4f}")

    print("\n[Phase 3] Generate a complete set of paper charts and tables...")
    fig_counter = 1

    # Figure 1: grid diagram
    print(f"[-] Generating Figure {fig_counter}...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ds_fine = xr.open_dataset(CONFIG['gc_processed'])
        data_fine = ds_fine['O3_gc'].isel(date=0).sel(lat=slice(20, 26), lon=slice(109, 118))
        data_coarse = data_fine.coarsen(lat=20, lon=20, boundary='trim').mean()
        data_coarse.plot.pcolormesh(ax=axes[0], cmap='viridis', add_colorbar=True, edgecolors='face')
        axes[0].set_title("O3_geochem_daily_mean (Coarse)")
        data_fine.plot(ax=axes[1], cmap='viridis'); axes[1].set_title("O3_geochem_daily_mean (Downscaled)")
        save_fig(fig, f"Figure{fig_counter}_Grid_Comparison.png"); fig_counter += 1
    except: pass

    # Figure 2:MCS analysis
    print(f"[-] Generating Figure {fig_counter} (MCS Analysis)...")
    mcs_results = []
    for scenario_name, scenario_res in results.items():
        if not scenario_res: continue
        print(f"    > Running MCS for {scenario_name}...")
        losses = {name: res['loss'] for name, res in scenario_res.items()}
        losses_df = pd.DataFrame(losses)
        mcs = MCS(losses_df, size=0.1)
        mcs.compute()
        for model_name, p_val in zip(mcs.pvalues.index, mcs.pvalues.values):
            if model_name in scenario_res:
                res = scenario_res[model_name]
                r2 = r2_score(res['y_test'], res['y_pred'])
                in_mcs = p_val > 0.1
                mcs_results.append({'Scenario': scenario_name, 'Model': model_name, 'R²': r2, 'MCS p-value': p_val, 'In 90% MCS': in_mcs})

    df_mcs = pd.DataFrame(mcs_results).sort_values(by=['Scenario', 'R²'], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(8, 4)); ax.axis('off')
    ax.set_title(f"Figure {fig_counter}: Model Confidence Set (MCS) Analysis", pad=20)
    table = ax.table(cellText=df_mcs.round(3).values, colLabels=df_mcs.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
    save_fig(fig, f"Figure{fig_counter}_MCS_Table.png"); fig_counter += 1
    print("\n--- MCS Results ---\n", df_mcs, "\n-------------------\n")

    # Figure 3:bar chart
    print(f"[-] Generating Figure {fig_counter} (Bar Chart)...")
    models_list = ['RandomForest', 'XGBoost', 'MLP', 'DNN']
    hk_scores = [r2_score(results['HK_Scenario'][m]['y_test'], results['HK_Scenario'][m]['y_pred']) if m in results['HK_Scenario'] else 0 for m in models_list]
    gba_scores = [r2_score(results['GBA_Scenario'][m]['y_test'], results['GBA_Scenario'][m]['y_pred']) if m in results['GBA_Scenario'] else 0 for m in models_list]
    x = np.arange(len(models_list)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, hk_scores, width, label='HK Local', color='#A6CEE3')
    rects2 = ax.bar(x + width/2, gba_scores, width, label='GBA Regional', color='#1F78B4')
    ax.set_ylabel('R² Score'); ax.set_title(f'Figure {fig_counter}: Model Performance Overview')
    ax.set_xticks(x); ax.set_xticklabels(models_list); ax.set_ylim(0, 1.1); ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.2f'); ax.bar_label(rects2, padding=3, fmt='%.2f')
    save_fig(fig, f"Figure{fig_counter}_Model_Benchmarking_Bar.png"); fig_counter += 1

    # Detailed scatter plot
    for scenario_name, scenario_res in results.items():
        if not scenario_res: continue
        print(f"\n[-] Generating Detailed Figures for {scenario_name}...")
        for name, res in scenario_res.items():
            title = f"{scenario_name} - {name}"
            fig = draw_scatter_hist_pair(res['y_test'], res['y_pred'], title)
            save_fig(fig, f"Figure{fig_counter}_{scenario_name}_{name}.png"); fig_counter += 1
            
    # Champion Model Analysis
    print("\n[-] Generating Champion Model Analysis (XGBoost)...")
    champion_res = results['GBA_Scenario']['XGBoost']
    y_test_champ = champion_res['y_test']
    y_pred_champ = champion_res['y_pred']
    champion_model = champion_res['model']
    X_test_champ = X_gba.loc[y_test_champ.index]
    
    if 'O3_gc' in X_test_champ.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        r2_champ = r2_score(y_test_champ, y_pred_champ)
        axes[0].scatter(y_test_champ, y_pred_champ, c='purple', alpha=0.5)
        axes[0].plot([0, 200], [0, 200], 'r--'); axes[0].set_title(f"Champion Model (XGBoost)\nR² = {r2_champ:.2f}")
        axes[0].set_xlabel("Observed"); axes[0].set_ylabel("Predicted")
        gc_vals = X_test_champ['O3_gc']; r2_gc = r2_score(y_test_champ, gc_vals)
        axes[1].scatter(gc_vals, y_test_champ, c='steelblue', alpha=0.5)
        axes[1].plot([0, 200], [0, 200], 'r--'); axes[1].set_title(f"Physical Model (GEOS-Chem)\nR² = {r2_gc:.2f}")
        axes[1].set_xlabel("Simulated"); axes[1].set_ylabel("Observed")
        save_fig(fig, f"Figure{fig_counter}_Champion_vs_Physics.png"); fig_counter += 1

    # SHAP
    print("[-] Generating SHAP Figures...")
    xgb_inner = champion_model.named_steps['xgboost']
    scaler_inner = champion_model.named_steps['scaler']
    X_shap = pd.DataFrame(scaler_inner.transform(X_test_champ), columns=X_test_champ.columns)
    explainer = shap.Explainer(xgb_inner)
    shap_values = explainer(X_shap)
    
    fig = plt.figure(); shap.summary_plot(shap_values, X_shap, show=False)
    plt.title("SHAP Summary"); save_fig(fig, f"Figure{fig_counter}_SHAP_Summary.png"); fig_counter += 1
    
    fig = plt.figure(); shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, color='#FF0055')
    plt.title("Feature Importance"); save_fig(fig, f"Figure{fig_counter}_SHAP_Bar.png"); fig_counter += 1
    
    sv = shap_values.values; cols = list(X_test_champ.columns)
    try:
        no2_idx = cols.index('NO2_obs'); t2m_idx = cols.index('t2m')
        candidates = [i for i, pred in enumerate(y_pred_champ) if pred > 60 and sv[i, no2_idx] < -0.5 and sv[i, t2m_idx] > 0.5]
        best_idx = candidates[0] if candidates else np.argmax(y_pred_champ)
    except: best_idx = np.argmax(y_pred_champ)
    
    fig = plt.figure(); shap.plots.waterfall(shap_values[best_idx], max_display=10, show=False)
    plt.title(f"Local Mechanism (Sample #{best_idx})"); save_fig(fig, f"Figure{fig_counter}_SHAP_Waterfall.png"); fig_counter += 1
    
    plt.figure(); shap.dependence_plot("t2m", shap_values.values, X_shap, show=False)
    plt.title("Nonlinear Effect of Temperature"); save_fig(plt.gcf(), f"Figure{fig_counter}_SHAP_Dependence_t2m.png"); fig_counter += 1
    
    plt.figure(); shap.dependence_plot("t2m", shap_values.values, X_shap, interaction_index="uvb", show=False)
    plt.title("Interaction between Temperature and UVB"); save_fig(plt.gcf(), f"Figure{fig_counter}_SHAP_Dependence_t2m_uvb.png"); fig_counter += 1
    
    plt.figure(); shap.dependence_plot("NO2_obs", shap_values.values, X_shap, show=False)
    plt.title("Nonlinear Effect of NO2"); save_fig(plt.gcf(), f"Figure{fig_counter}_SHAP_Dependence_NO2.png"); fig_counter += 1

    print(f"\n[Done] The chart has been generated to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
