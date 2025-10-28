import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 

class VisualizeDataset:
    def __init__(self, data):
        """
        Classe per visualizzare ed esplorare un dataset Pandas.
        """
        self.data = data
        sns.set(style="whitegrid", palette="muted", font_scale=1.1)

    def overview(self):
        """Mostra info generali sul dataset."""
        print("Shape:", self.data.shape)
        print("\nTipi di dato:\n", self.data.dtypes.value_counts())
        print("\nPrime righe:")
        self.data.head()

    def missing_values(self, top_n=20):
        """Grafico delle colonne con più valori mancanti."""
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
        if len(missing) == 0:
            print("Nessun valore mancante nel dataset.")
            return

        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing.values, y=missing.index, color='tomato')
        plt.title(f"Top {top_n} colonne con valori mancanti")
        plt.xlabel("Numero di NaN")
        plt.ylabel("Colonna")
        plt.show()

    def numeric_distribution(self, column):
        """Istogramma e boxplot per una variabile numerica."""
        if column not in self.data.columns:
            print(f"Colonna '{column}' non trovata.")
            return

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.data[column].dropna(), kde=True, bins=30)
        plt.title(f"Distribuzione di {column}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=self.data[column])
        plt.title(f"Boxplot di {column}")
        plt.show()

    def correlation_heatmap(self, top_n=20):
        """Mostra una heatmap delle correlazioni tra le variabili numeriche più importanti."""
        numeric_df = self.data.select_dtypes(include=['number'])
        if numeric_df.empty:
            print("Nessuna colonna numerica trovata.")
            return

        corr = numeric_df.corr()
        top_corr = corr.nlargest(top_n, 'SalePrice')['SalePrice'].index \
                    if 'SalePrice' in corr.columns else corr.columns[:top_n]
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr.loc[top_corr, top_corr], annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Heatmap di correlazione (top {top_n})")
        plt.show()

    def categorical_distribution(self, column):
        """Conta e visualizza la distribuzione di una variabile categorica."""
        if column not in self.data.columns:
            print(f"Colonna '{column}' non trovata.")
            return

        plt.figure(figsize=(10, 5))
        order = self.data[column].value_counts().index
        sns.countplot(data=self.data, x=column, order=order, palette="Set3")
        plt.title(f"Distribuzione di {column}")
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def categorical_vs_target(self, column, target):
        """Mostra come cambia la variabile target rispetto a una categorica."""
        if column not in self.data.columns or target not in self.data.columns:
            print(f"Colonna '{column}' o '{target}' non trovata.")
            return

        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.data, x=column, y=target, palette="Set2")
        plt.title(f"{target} in funzione di {column}")
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def qq_plot(self, column):
        """QQ-plot per verificare la normalità di una variabile (residui o target)."""
        if column not in self.data.columns:
            print(f"Colonna '{column}' non trovata.")
            return
        series = self.data[column].dropna()
        plt.figure(figsize=(6,6))
        stats.probplot(series, dist="norm", plot=plt)
        plt.title(f"Q-Q plot di {column}")
        plt.show()

    def jointplot_feature(self, feature, target="SalePrice", kind="reg"):
        """Jointplot (scatter + distribuzioni) tra feature e target."""
        if feature not in self.data.columns or target not in self.data.columns:
            print("Feature o target non trovati.")
            return
        sns.jointplot(data=self.data, x=feature, y=target, kind=kind, height=7, marginal_kws=dict(bins=30))
        plt.suptitle(f"Jointplot: {feature} vs {target}", y=1.02)
        plt.show()




# Esempio (usando il dataset House Prices)
df = pd.read_csv("../dataset/train_clean.csv")

viz = VisualizeDataset(df)
viz.overview()
viz.missing_values()
viz.numeric_distribution("SalePrice")
viz.correlation_heatmap(top_n=15)
viz.categorical_distribution("Neighborhood")
viz.categorical_vs_target("OverallQual", "SalePrice")

viz.qq_plot("SalePrice")
viz.jointplot_feature("GrLivArea", "SalePrice")