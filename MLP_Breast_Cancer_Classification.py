import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import make_scorer, recall_score
cancer_data = load_breast_cancer()

X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
#y = 1 - cancer_data.target
y = cancer_data.target

print(f"Dimensões da base: {X.shape[0]} pacientes e {X.shape[1]} características.")
print("-" * 50)

modelo_pipeline = make_pipeline(
    StandardScaler(),
    #MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=1000, random_state=42)
    #MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    #MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
    #MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=10000, random_state=42)
    MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=500, random_state=42)
    #MLPClassifier(hidden_layer_sizes=(30, 15), max_iter=100, random_state=42)
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- TAMANHO DAS DIVISÕES (TREINO / TESTE) ---")

for fold, (indices_treino, indices_teste) in enumerate(kf.split(X)):
    qtd_treino = len(indices_treino)
    qtd_teste = len(indices_teste)
    print(f"Fold {fold + 1}: Treino com {qtd_treino} instâncias | Teste com {qtd_teste} instâncias")

print("-" * 50)

#metricas = ['accuracy', 'precision', 'recall']
recall_maligno = make_scorer(recall_score, pos_label=0)

metricas = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall_maligno': recall_maligno
}

print("\nExecutando o treinamento e validação em 5 Folds...")
resultados = cross_validate(modelo_pipeline, X, y, cv=kf, scoring=metricas)

for i in range(5):
    print(f"Fold {i+1} | "
          f"Acurácia: {resultados['test_accuracy'][i]*100:.2f}% | "
          f"Precisão: {resultados['test_precision'][i]*100:.2f}% | "
          f"Recall: {resultados['test_recall_maligno'][i]*100:.2f}%")

print("-" * 50)
print("\n--- DESEMPENHO MÉDIO FINAL ---")

print(f"Acurácia Média: {resultados['test_accuracy'].mean() * 100:.2f}% "
      f"(± {resultados['test_accuracy'].std() * 100:.2f}%)")

print(f"Precisão Média: {resultados['test_precision'].mean() * 100:.2f}% "
      f"(± {resultados['test_precision'].std() * 100:.2f}%)")

print(f"Recall Médio:   {resultados['test_recall_maligno'].mean() * 100:.2f}% "
      f"(± {resultados['test_recall_maligno'].std() * 100:.2f}%)")
print("-" * 30)

#MATRIZ DE CONFUSAO
y_pred = cross_val_predict(modelo_pipeline, X, y, cv=kf)

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    #display_labels=["Benigno", "Maligno"]
    display_labels=["Maligno", "Benigno"]

)

disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusão - MLP")
plt.show()