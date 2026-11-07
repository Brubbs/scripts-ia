import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carrega o conjunto de dados iris
iris = load_iris()

# Seleciona as colunas correspondentes a petal length (cm) e petal width (cm)
# petal length é a terceira coluna (índice 2) e petal width é a quarta (índice 3)
petal_length = iris.data[:, 2]
petal_width = iris.data[:, 3]

# Cria um gráfico de dispersão (scatter plot)
# O parâmetro 'c' define a cor dos pontos. Usaremos 'b' para azul (blue).
plt.scatter(petal_length, petal_width, c='b')

# Define os rótulos dos eixos X e Y
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

# Define o título do gráfico
plt.title("Relação entre Comprimento e Largura da Pétala")

# Exibe o gráfico
plt.show()

