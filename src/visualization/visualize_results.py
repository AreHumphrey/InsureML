import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, feature_names):

    importance = model.get_feature_importance()
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Важность признаков")
    plt.xlabel("Значимость")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.show()
