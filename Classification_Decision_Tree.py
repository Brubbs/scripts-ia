import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay 

# 1. CRIANDO UM DATASET SIMULADO (Cenário: Churn de Clientes)
np.random.seed(42)
n_samples = 500

# Variáveis Explicativas (Features)
tempo_contrato_meses = np.random.randint(1, 60, n_samples)
gasto_mensal = np.random.uniform(50, 500, n_samples)
chamados_suporte = np.random.randint(0, 10, n_samples)

# Lógica sintética para criar a resposta (Target):
# Maior suporte + menor tempo de contrato = maior chance de cancelamento
prob_churn = (chamados_suporte * 0.15) - (tempo_contrato_meses * 0.01) + 0.2
y = (prob_churn > np.random.uniform(0, 1, n_samples)).astype(int)

df = pd.DataFrame({
    'tempo_contrato_meses': tempo_contrato_meses,
    'gasto_mensal': gasto_mensal,
    'chamados_suporte': chamados_suporte,
    'cancelou': y
})

# 2. SEPARAÇÃO DE FEATURES (X) E TARGET (y)
X = df[['tempo_contrato_meses', 'gasto_mensal', 'chamados_suporte']]
y = df['cancelou']

# 3. DIVISÃO EM TREINO E TESTE (80% Treino / 20% Teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. INSTANCIAÇÃO E TREINAMENTO DO MODELO
# Limitamos max_depth=3 para evitar overfitting e facilitar a interpretação
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# 5. PREDIÇÃO NOS DADOS DE TESTE
y_pred = modelo.predict(X_test)

# 6. AVALIAÇÃO DO MODELO
print("=== Acurácia no Teste ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}\n")

print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred, target_names=['Permaneceu (0)', 'Cancelou (1)']))

# 7. INSPECIONANDO AS REGRAS APRENDIDAS PELA ÁRVORE
print("=== Regras Aprendidas pelo Modelo ===")
regras = export_text(modelo, feature_names=list(X.columns))
print(regras)