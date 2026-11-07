import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Carregamento dos dados
iris = load_iris(as_frame=True)
df = iris.frame

# Adiciona a coluna com os nomes das espécies para facilitar a interpretação
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

features = ["petal length (cm)", "petal width (cm)"]

# 2. Exibição das estatísticas descritivas básicas no console
print("=== Resumo Estatístico das Variáveis ===")
print(df[features].describe().T[["mean", "std", "min", "50%", "max"]])

# 3. Configuração dos Boxplots (1 linha, 2 colunas)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot 1: Comprimento da Pétala por Espécie
sns.boxplot(
    x="species",
    y="petal length (cm)",
    data=df,
    ax=axes[0],
    palette="Set2",
    hue="species",
    legend=False,
)
axes[0].set_title("Distribuição do Comprimento da Pétala")
axes[0].set_xlabel("Espécie")
axes[0].set_ylabel("Comprimento (cm)")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Boxplot 2: Largura da Pétala por Espécie
sns.boxplot(
    x="species",
    y="petal width (cm)",
    data=df,
    ax=axes[1],
    palette="Set2",
    hue="species",
    legend=False,
)
axes[1].set_title("Distribuição da Largura da Pétala")
axes[1].set_xlabel("Espécie")
axes[1].set_ylabel("Largura (cm)")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

