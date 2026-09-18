import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

plt.rcParams['figure.dpi'] = 150
NAVY = "#16263B"
TEAL = "#3FA7A0"
CH = "charts"

with open("trained_models.pkl", "rb") as f:
    data = pickle.load(f)

models = data["models"]
X_train, X_test = data["X_train"], data["X_test"]
feature_names = data["feature_names"]
champion = models["Gradient Boosting"]

importances = champion.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
imp_df.to_csv("feature_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
top = imp_df.head(12).iloc[::-1]
ax.barh(top['feature'], top['importance'], color=TEAL)
ax.set_xlabel("Feature Importance")
ax.set_title("Feature Importance — Gradient Boosting (Champion Model)")
plt.tight_layout()
plt.savefig(f"{CH}/08_feature_importance.png")
plt.close()

explainer = shap.TreeExplainer(champion)
sample = X_test.sample(min(1000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(sample)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, sample, show=False, plot_size=(8,6))
plt.tight_layout()
plt.savefig(f"{CH}/09_shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
shap_imp_df.to_csv("shap_importance.csv", index=False)

top4 = imp_df.head(4)['feature'].tolist()
fig, ax = plt.subplots(figsize=(10, 7))
PartialDependenceDisplay.from_estimator(champion, X_train, top4, ax=ax, n_cols=2,
                                          line_kw={"color": NAVY, "linewidth": 2})
plt.tight_layout()
plt.savefig(f"{CH}/10_partial_dependence.png")
plt.close()

print("done - all charts saved including PDP")
