"""
BASE COMUM DA QUESTÃO 2
=======================

Tudo o que precisa ser IDÊNTICO para os 5 classificadores fica aqui:

1. DADOS: X (4601 e-mails x 57 atributos) e as duas versões da resposta
     - "v1": rótulo original (1 = spam, 0 = não-spam);
     - "v2": cluster de objetos com K* = 2 dado pelo double k-means da
             Questão 1 (double-kmeans-spambase/output/cl_star.csv).
   Em ambas a classe POSITIVA (1) é a classe MINORITÁRIA: spam na v1
   (1813 de 4601) e o cluster de 34 e-mails na v2. Precisão, cobertura e
   F-measure são sempre calculadas para essa classe.

2. SPLITS: a validação cruzada estratificada 30 x 10-folds é sorteada UMA
   vez por versão e salva em questao-02/splits/. Todos os classificadores
   leem o mesmo arquivo e, portanto, são avaliados exatamente nos mesmos
   300 pares (treino, teste). Isso é obrigatório para o teste de Friedman /
   Nemenyi (que compara os classificadores split a split) e para o voto
   majoritário (que combina as previsões dos 4 classificadores no MESMO
   fold de teste). A validação interna de 5 folds também é a mesma para
   todos os classificadores dentro de um mesmo split externo.

3. MÉTRICAS e INTERVALO DE CONFIANÇA (com a correção de Nadeau & Bengio).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

RAIZ = Path(__file__).resolve().parent.parent
ARQ_DADOS = RAIZ / "double-kmeans-spambase" / "data" / "spambase.data"
ARQ_CL_STAR = RAIZ / "double-kmeans-spambase" / "output" / "cl_star.csv"
PASTA_SPLITS = Path(__file__).resolve().parent / "splits"

VERSOES = ("v1", "v2")
N_REPETICOES = 30
N_FOLDS = 10
N_FOLDS_INTERNOS = 5
SEMENTE = 2026

METRICAS = ("taxa_erro", "precisao", "cobertura", "f_measure")
NOMES_METRICAS = {
    "taxa_erro": "Taxa de erro",
    "precisao": "Precisão",
    "cobertura": "Cobertura",
    "f_measure": "F-measure",
}


# ---------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------

def carregar_dados():
    """Devolve X (n x 57) e um dicionário {versão: y}, com y em {0, 1}."""
    dados = pd.read_csv(ARQ_DADOS, header=None)
    X = dados.iloc[:, :57].to_numpy(float)
    y_v1 = dados.iloc[:, 57].to_numpy(int)

    if not ARQ_CL_STAR.exists():
        raise FileNotFoundError(
            f"{ARQ_CL_STAR} não encontrado. Rode o script R da Questão 1 "
            "ou questao-02/reconstroi_cl_star.py."
        )
    cl_star = pd.read_csv(ARQ_CL_STAR)["cluster"].to_numpy()
    rotulos, contagens = np.unique(cl_star, return_counts=True)
    y_v2 = (cl_star == rotulos[np.argmin(contagens)]).astype(int)   # 1 = cluster minoritário

    return X, {"v1": y_v1, "v2": y_v2}


# ---------------------------------------------------------------------
# 2. Splits fixos (30 x 10-folds estratificado) e validação interna
# ---------------------------------------------------------------------

def _arquivo_splits(versao):
    return PASTA_SPLITS / f"folds_{versao}.csv"


def gerar_splits(y, versao):
    """Sorteia os 30 x 10 folds e salva o fold de cada objeto em cada repetição.

    O arquivo tem uma linha por e-mail e uma coluna por repetição
    (rep01..rep30); o valor é o fold (1..10) em que o e-mail é TESTE.
    """
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPETICOES, random_state=SEMENTE)
    folds = np.zeros((len(y), N_REPETICOES), dtype=int)
    for i, (_, idx_teste) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[idx_teste, i // N_FOLDS] = i % N_FOLDS + 1

    PASTA_SPLITS.mkdir(exist_ok=True)
    colunas = [f"rep{r + 1:02d}" for r in range(N_REPETICOES)]
    pd.DataFrame(folds, columns=colunas).to_csv(_arquivo_splits(versao), index=False)
    return folds


def carregar_splits(y, versao):
    """Lista de (repetição, fold, idx_treino, idx_teste), numerados a partir de 1.

    Lê o arquivo salvo (ou o gera, na primeira vez) e confere que cada fold
    de teste preserva a estratificação.
    """
    arquivo = _arquivo_splits(versao)
    folds = pd.read_csv(arquivo).to_numpy() if arquivo.exists() else gerar_splits(y, versao)
    if folds.shape != (len(y), N_REPETICOES):
        raise ValueError(f"{arquivo} não corresponde aos dados da versão {versao}")

    splits = []
    for r in range(N_REPETICOES):
        for f in range(1, N_FOLDS + 1):
            teste = folds[:, r] == f
            splits.append((r + 1, f, np.flatnonzero(~teste), np.flatnonzero(teste)))
    return splits


def cv_interna(repeticao, fold):
    """Validação interna de 5 folds (sobre os 9 folds de treino) do split externo dado.

    A semente depende só do split externo, então todos os classificadores
    ajustam seus hiper-parâmetros nas mesmas 5 partes.
    """
    return StratifiedKFold(n_splits=N_FOLDS_INTERNOS, shuffle=True,
                           random_state=SEMENTE + 100 * repeticao + fold)


# ---------------------------------------------------------------------
# 3. Métricas e intervalo de confiança
# ---------------------------------------------------------------------

def calcular_metricas(y_true, y_pred):
    """Taxa de erro, precisão, cobertura e F-measure da classe positiva (1).

    zero_division=0: se um classificador não prevê nenhum positivo num fold
    (possível na v2, com 3 ou 4 positivos por fold), a precisão vale 0.
    """
    return {
        "taxa_erro": 1 - accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "cobertura": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f_measure": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def intervalo_confianca(valores, nivel=0.95):
    """Estimativa pontual e IC com a correção de Nadeau & Bengio (2003).

    Os 300 valores da CV repetida NÃO são independentes (os conjuntos de
    treino se sobrepõem), então o erro-padrão ingênuo dp/sqrt(n) é pequeno
    demais. A correção soma a razão n_teste/n_treino (= 1/9 no 10-fold):
        EP = dp * sqrt(1/n + n_teste/n_treino)
    """
    valores = np.asarray(valores, dtype=float)
    n = len(valores)
    media = valores.mean()
    dp = valores.std(ddof=1)
    erro_padrao = dp * np.sqrt(1 / n + 1 / (N_FOLDS - 1))
    margem = stats.t.ppf((1 + nivel) / 2, df=n - 1) * erro_padrao
    return {
        "estimativa": media,
        "desvio_padrao": dp,
        "ic_inferior": max(0.0, media - margem),
        "ic_superior": min(1.0, media + margem),
    }
