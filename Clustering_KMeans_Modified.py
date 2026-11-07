import os
# Define o número de threads do OpenMP para 1 antes de carregar o sklearn
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
#from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Carregar o arquivo de dados original
df = pd.read_csv('iris_dataset_original.csv')

# 2. Separar as features numéricas da coluna de espécie
# df.iloc[:, :-1] seleciona todas as linhas e todas as colunas, EXCETO a última
X = df.iloc[:, :-1]

# 3. Padronização das Variáveis
# Garante que as variáveis fiquem na mesma escala (Média 0, Desvio Padrão 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Avaliação de Desempenho para Múltiplos Valores de K
wcss = []
silhouette_coefficients = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled) # treina o modelo

    # Coleta das métricas
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_coefficients.append(score)

# 5. Plotagem das Métricas de Avaliação
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5)) #largura e altura são em polegadas.

# Gráfico 1: Método do Cotovelo (Inércia)
ax1.plot(k_range, wcss, marker="o", color="b")
ax1.set_title("Método do Cotovelo (Inércia / WCSS)")
ax1.set_xlabel("Número de Clusters (K)")
ax1.set_ylabel("Inércia")
ax1.grid(True)

# Gráfico 2: Coeficiente de Silhueta
ax2.plot(k_range, silhouette_coefficients, marker="s", color="green")
ax2.set_title("Coeficiente de Silhueta Médio")
ax2.set_xlabel("Número de Clusters (K)")
ax2.set_ylabel("Silhouette Score")
ax2.grid(True)

plt.tight_layout()
plt.show()

# 5. Treinamento do Modelo Final (K = 3) e Visualização dos Resultados
k_otimo = 3
kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
labels = kmeans_final.fit_predict(X_scaled) #treina o modelo e atribui cada observação a um cluster
centroides = kmeans_final.cluster_centers_

plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],     # todas as variáveis referente ao comprimento da pétala
    X_scaled[:, 1],     # todas as variáveis referente a largura da pétala
    c=labels,           # o parâmetro de cor recebe o rótulo (0, 1 ou 2)
    cmap="viridis",     # paleta de cores padrão do Matplotlib
    s=50,               # tamanho do ponto
    alpha=0.8,          # opacidade
    label="Amostras",   # título na legenda
)
plt.scatter(
    centroides[:, 0],
    centroides[:, 1],
    c="red",
    marker="X",
    s=200,
    label="Centroides",
)

plt.title(f"Segmentação com K-Means (K = {k_otimo})")
plt.xlabel("Petal Length (Padronizado)")
plt.ylabel("Petal Width (Padronizado)")
plt.legend()
plt.grid(True)
plt.show()

# Exibição do Silhouette Score Final
sil_score_final = silhouette_score(X_scaled, labels)
print(f"Silhouette Score para K={k_otimo}: {sil_score_final:.4f}")