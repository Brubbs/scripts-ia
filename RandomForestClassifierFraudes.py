import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

print("\n=== 1. Entendimento do negócio ===")
CUSTO_FALSO_NEGATIVO = 500
CUSTO_FALSO_POSITIVO = 5

print("\n=== 2. Entendimento dos dados ===")
rng = np.random.default_rng(42)
n = 50_000

fraude = rng.random(n) < 0.005

valor = np.where(fraude, rng.lognormal(4.3, 1.0, n), rng.lognormal(3.6, 0.9, n)).round(2)
hora = np.where(fraude, rng.normal(3, 5, n), rng.normal(14, 5, n)) % 24
distancia = np.where(fraude, rng.exponential(35, n), rng.exponential(10, n)).round(1)
compras = np.where(fraude, rng.poisson(3.2, n), rng.poisson(1.6, n))

tipos = ["online", "presencial", "posto", "restaurante"]
tipo = np.where(fraude, rng.choice(tipos, n, p=[.55, .20, .10, .15]),
                        rng.choice(tipos, n, p=[.25, .45, .10, .20]))

df = pd.DataFrame({"valor": valor, "hora": hora.round(1), "distancia_casa_km": distancia,
                   "compras_24h": compras, "tipo": tipo, "fraude": fraude.astype(int)})

df.loc[rng.random(n) < 0.05, "distancia_casa_km"] = np.nan

print("Primeiras linhas dos dados brutos:")
print(df.head())
print("Linhas e colunas:", df.shape)
print("\nValores faltantes por coluna:")
print(df.isna().sum())
print("\nProporção de fraudes: {:.2%}".format(df["fraude"].mean()))

acuracia_baseline = 1 - df["fraude"].mean()
print("Acurácia de um 'modelo' que SEMPRE diz 'não é fraude': {:.2%}".format(acuracia_baseline))

print("\n=== 3. Preparação dos dados ===")
X = pd.get_dummies(df.drop(columns="fraude"), columns=["tipo"])
X["distancia_casa_km"] = X["distancia_casa_km"].fillna(X["distancia_casa_km"].median())

y = df["fraude"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print("Fraudes no treino: {} | no teste: {}".format(y_treino.sum(), y_teste.sum()))

print("\n=== 4. Modelagem ===")
modelo = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)
modelo.fit(X_treino, y_treino)

prob = modelo.predict_proba(X_teste)[:, 1]

print("AUC: {:.3f}".format(roc_auc_score(y_teste, prob)))

print("\n=== 5. Avaliação com o limiar padrão (0,5) ===")
def avaliar(limiar):
    previsto = (prob >= limiar).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_teste, previsto).ravel()

    return dict(limiar=limiar, TP=tp, FN=fn, FP=fp, TN=tn,
                sensibilidade=tp / (tp + fn),
                especificidade=tn / (tn + fp),
                precisao=tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
                acuracia=(tp + tn) / len(y_teste),
                custo_total=fn * CUSTO_FALSO_NEGATIVO + fp * CUSTO_FALSO_POSITIVO)

r = avaliar(0.5)
print("Fraudes pegas (TP): {} | fraudes perdidas (FN): {}".format(r["TP"], r["FN"]))
print("Alarmes falsos (FP): {} | legítimas liberadas (TN): {}".format(r["FP"], r["TN"]))
print()
print("Sensibilidade : {:.1%}".format(r["sensibilidade"]))
print("Especificidade: {:.2%}".format(r["especificidade"]))
print("Precisão      : {:.1%}".format(r["precisao"]))
print("Acurácia      : {:.2%}".format(r["acuracia"]))

print("\n=== 6. Avaliação: escolher o limiar pelo custo ===")
tabela = pd.DataFrame([avaliar(l) for l in [0.5, 0.2, 0.1, 0.05, 0.02]])
tabela["alertas"] = tabela["TP"] + tabela["FP"]

custo_sem_modelo = y_teste.sum() * CUSTO_FALSO_NEGATIVO
print("Custo sem nenhum modelo (todas as fraudes passam): R$ {:,.0f}".format(custo_sem_modelo))

colunas = ["limiar", "alertas", "TP", "FN", "FP", "sensibilidade",
           "especificidade", "precisao", "custo_total"]
print(tabela[colunas].round(3).to_string(index=False))
