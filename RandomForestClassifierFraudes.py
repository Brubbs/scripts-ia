"""
Do dado bruto à métrica: detecção de fraude em cartão de crédito

Percorre um projeto de ciência de dados de ponta a ponta, seguindo o CRISP-DM:
  1. Entendimento do negócio   2. Entendimento dos dados   3. Preparação
  4. Modelagem                 5. Avaliação (limiar 0,5)   6. Avaliação pelo custo

AVISO: os dados são SIMULADOS (nenhuma transação real) e os custos dos erros
são hipotéticos. Servem apenas para ilustrar a interpretação das métricas.

Requisitos: Python 3.9+ e as bibliotecas numpy, pandas e scikit-learn.
    pip install numpy pandas scikit-learn

Como executar:
    python fraudes.py

Como interpretar a saída (resumo):
  - Seção 2: só ~0,56% das transações são fraude (classes desbalanceadas) e um
    "modelo" que nunca detecta fraude já teria ~99,4% de acurácia.
  - Seção 5: com limiar 0,5, a acurácia (~99,5%) e a especificidade (~99,99%)
    parecem ótimas, mas a sensibilidade é de ~4%: o modelo perdeu quase todas as fraudes.
  - Seção 6: baixar o limiar aumenta a sensibilidade e reduz a precisão. O limiar
    deve ser escolhido pelo custo dos erros, que é uma decisão de negócio.

Os resultados podem variar ligeiramente conforme a versão do scikit-learn.
"""
# Bibliotecas usadas na demonstração
import numpy as np                                     # cálculos numéricos e geração de números aleatórios
import pandas as pd                                    # tabelas de dados (DataFrames)
from sklearn.model_selection import train_test_split   # separa os dados em treino e teste
from sklearn.ensemble import RandomForestClassifier    # o modelo de classificação (floresta aleatória)
from sklearn.metrics import confusion_matrix, roc_auc_score  # métricas de avaliação

# ======================================================================
# 1. Entendimento do negócio
# ======================================================================
print("\n=== 1. Entendimento do negócio ===")
# Premissas de negócio (hipotéticas): quanto custa cada tipo de erro, em reais.
# Estes valores definem o que é um "bom" modelo para a empresa, e não o algoritmo.
CUSTO_FALSO_NEGATIVO = 500   # fraude que passou despercebida (a empresa perde dinheiro)
CUSTO_FALSO_POSITIVO = 5     # transação legítima enviada para revisão (custo de análise e incômodo)

# ======================================================================
# 2. Entendimento dos dados
# ======================================================================
print("\n=== 2. Entendimento dos dados ===")
# --- Simulação dos dados (em um projeto real, aqui entraria a leitura do banco de dados) ---
rng = np.random.default_rng(42)   # gerador aleatório com semente fixa: o resultado se repete a cada execução
n = 50_000                        # número de transações simuladas

# Sorteia quais transações são fraude: cada uma tem 0,5% de chance (por isso a classe é rara)
fraude = rng.random(n) < 0.005

# Cada variável é sorteada com um perfil diferente para fraude e para transação legítima.
# np.where(condição, valor_se_fraude, valor_se_legítima) escolhe o perfil de cada linha.
valor     = np.where(fraude, rng.lognormal(4.3, 1.0, n), rng.lognormal(3.6, 0.9, n)).round(2)  # fraudes têm valores maiores
hora      = np.where(fraude, rng.normal(3, 5, n), rng.normal(14, 5, n)) % 24                   # fraudes ocorrem mais de madrugada
distancia = np.where(fraude, rng.exponential(35, n), rng.exponential(10, n)).round(1)          # fraudes ocorrem mais longe de casa
compras   = np.where(fraude, rng.poisson(3.2, n), rng.poisson(1.6, n))                         # fraudes vêm em rajadas (mais compras em 24h)

# Tipo de estabelecimento: fraudes são mais frequentes em compras online
tipos = ["online", "presencial", "posto", "restaurante"]
tipo = np.where(fraude, rng.choice(tipos, n, p=[.55, .20, .10, .15]),
                        rng.choice(tipos, n, p=[.25, .45, .10, .20]))

# Monta a tabela final; "fraude" é a variável-alvo (1 = fraude, 0 = legítima)
df = pd.DataFrame({"valor": valor, "hora": hora.round(1), "distancia_casa_km": distancia,
                   "compras_24h": compras, "tipo": tipo, "fraude": fraude.astype(int)})

# Apaga 5% das distâncias para imitar dados faltantes, comuns em dados reais
df.loc[rng.random(n) < 0.05, "distancia_casa_km"] = np.nan

print("Primeiras linhas dos dados brutos:")
print(df.head())   # mostra as 5 primeiras linhas
# Primeiro olhar para os dados: tamanho, faltantes e proporção da classe-alvo
print("Linhas e colunas:", df.shape)
print("\nValores faltantes por coluna:")
print(df.isna().sum())                                      # conta os faltantes em cada coluna
print("\nProporção de fraudes: {:.2%}".format(df["fraude"].mean()))  # a média de uma coluna 0/1 é a proporção de 1s
# Interpretação: ~0,56% de fraudes e ~5% de distâncias faltantes.

# Ponto de comparação: um "modelo" que sempre responde "não é fraude".
# Ele acerta todas as transações legítimas e erra todas as fraudes.
acuracia_baseline = 1 - df["fraude"].mean()
print("Acurácia de um 'modelo' que SEMPRE diz 'não é fraude': {:.2%}".format(acuracia_baseline))
# Interpretação: um modelo que nunca detecta fraude já teria ~99,4% de acurácia.


# ======================================================================
# 3. Preparação dos dados
# ======================================================================
print("\n=== 3. Preparação dos dados ===")
# Transforma a coluna de texto "tipo" em colunas numéricas (uma por categoria, com 0 ou 1).
# O modelo só trabalha com números.
X = pd.get_dummies(df.drop(columns="fraude"), columns=["tipo"])   # X = variáveis de entrada

# Preenche as distâncias faltantes com a mediana (valor central, pouco sensível a extremos)
X["distancia_casa_km"] = X["distancia_casa_km"].fillna(X["distancia_casa_km"].median())

y = df["fraude"]   # y = o que queremos prever

# Separa 70% para treinar e 30% para testar o modelo em dados que ele nunca viu.
# stratify=y garante a mesma proporção de fraudes nas duas partes (sem isso, o teste poderia ficar sem fraudes).
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

print("Fraudes no treino: {} | no teste: {}".format(y_treino.sum(), y_teste.sum()))

# ======================================================================
# 4. Modelagem
# ======================================================================
print("\n=== 4. Modelagem ===")
# Cria o modelo: 200 árvores de decisão que "votam" juntas.
# min_samples_leaf=5 evita que cada árvore decore casos isolados (reduz o excesso de ajuste).
modelo = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)

# Treina usando apenas os dados de treino
modelo.fit(X_treino, y_treino)

# Para cada transação de teste, o modelo devolve a probabilidade de ser fraude (entre 0 e 1).
# [:, 1] seleciona a coluna da classe "fraude".
prob = modelo.predict_proba(X_teste)[:, 1]

# AUC: mede se o modelo dá probabilidades maiores às fraudes do que às legítimas (1 = perfeito, 0,5 = sorteio)
print("AUC: {:.3f}".format(roc_auc_score(y_teste, prob)))
# Interpretação: a AUC (~0,92) mostra que o modelo ordena bem, mas não define o limiar nem o ganho para o negócio.


# ======================================================================
# 5. Avaliação com o limiar padrão (0,5)
# ======================================================================
print("\n=== 5. Avaliação com o limiar padrão (0,5) ===")
def avaliar(limiar):
    # Aplica um limiar às probabilidades e calcula as métricas de desempenho
    # Regra de decisão: probabilidade >= limiar vira "revisar" (1); caso contrário, "liberar" (0)
    previsto = (prob >= limiar).astype(int)

    # Matriz de confusão: TN, FP, FN e TP (verdadeiros/falsos negativos/positivos)
    tn, fp, fn, tp = confusion_matrix(y_teste, previsto).ravel()

    return dict(limiar=limiar, TP=tp, FN=fn, FP=fp, TN=tn,
                sensibilidade=tp / (tp + fn),      # das fraudes reais, quantas foram achadas
                especificidade=tn / (tn + fp),     # das legítimas, quantas foram liberadas corretamente
                precisao=tp / (tp + fp) if (tp + fp) > 0 else float("nan"),  # dos alertas, quantos eram fraude
                acuracia=(tp + tn) / len(y_teste), # proporção total de acertos
                custo_total=fn * CUSTO_FALSO_NEGATIVO + fp * CUSTO_FALSO_POSITIVO)  # custo em reais

# Avalia com o limiar padrão de 0,5
r = avaliar(0.5)
print("Fraudes pegas (TP): {} | fraudes perdidas (FN): {}".format(r["TP"], r["FN"]))
print("Alarmes falsos (FP): {} | legítimas liberadas (TN): {}".format(r["FP"], r["TN"]))
print()
print("Sensibilidade : {:.1%}".format(r["sensibilidade"]))
print("Especificidade: {:.2%}".format(r["especificidade"]))
print("Precisão      : {:.1%}".format(r["precisao"]))
print("Acurácia      : {:.2%}".format(r["acuracia"]))
# Interpretação: acurácia e especificidade altas escondem uma sensibilidade de ~4%.


# ======================================================================
# 6. Avaliação: escolher o limiar pelo custo
# ======================================================================
print("\n=== 6. Avaliação: escolher o limiar pelo custo ===")
# Testa vários limiares e reúne os resultados em uma tabela (uma linha por limiar)
tabela = pd.DataFrame([avaliar(l) for l in [0.5, 0.2, 0.1, 0.05, 0.02]])
tabela["alertas"] = tabela["TP"] + tabela["FP"]   # total de transações enviadas para revisão

# Custo de não usar modelo nenhum: todas as fraudes do teste passam
custo_sem_modelo = y_teste.sum() * CUSTO_FALSO_NEGATIVO
print("Custo sem nenhum modelo (todas as fraudes passam): R$ {:,.0f}".format(custo_sem_modelo))

# Mostra as colunas mais importantes, arredondadas
colunas = ["limiar", "alertas", "TP", "FN", "FP", "sensibilidade",
           "especificidade", "precisao", "custo_total"]
print(tabela[colunas].round(3).to_string(index=False))
# Interpretação: ao baixar o limiar, a sensibilidade sobe, a precisão cai e o custo total diminui.
# Experimente mudar CUSTO_FALSO_POSITIVO para 50 e execute de novo: o limiar de menor custo muda.
