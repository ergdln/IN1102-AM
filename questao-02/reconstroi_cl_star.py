"""
RECONSTRUÇÃO DOS RÓTULOS DA VERSÃO 2 (cl_star) EM PYTHON
=========================================================

A versão 2 da base na Questão 2 usa como variável resposta a partição de
objetos com K* clusters encontrada pelo double k-means na Questão 1.
O script R da Questão 1 grava esses rótulos em
    double-kmeans-spambase/output/cl_star.csv
Este script existe para quem não tem R à mão: ele porta o MESMO algoritmo
(double_kmeans_spambase.R, seção 2), roda as 100 execuções para (K*, H*) e
só grava cl_star.csv se o resultado coincidir com as saídas da Questão 1:
    - função objetivo f  (output/resultados_KH.csv)
    - matriz G           (output/G_prototipos.csv)
    - clusters de variáveis (output/clusters_de_variaveis.csv)
A numeração dos clusters é alinhada à do R (comparando com esses arquivos),
para que "cluster 1" signifique o mesmo grupo nas duas linguagens.

Uso (a partir da raiz do repositório):
    python questao-02/reconstroi_cl_star.py
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA_Q1 = RAIZ / "double-kmeans-spambase"
SAIDA = PASTA_Q1 / "output" / "cl_star.csv"


# ---------------------------------------------------------------------
# Double k-means (mesma notação e mesmos passos do script R)
# ---------------------------------------------------------------------

def particao_aleatoria(rng, m, k):
    # Sorteia rótulos até que nenhum cluster fique vazio.
    while True:
        rotulo = rng.integers(0, k, m)
        if len(np.unique(rotulo)) == k:
            return rotulo


def indicadora(rotulo, k):
    M = np.zeros((len(rotulo), k))
    M[np.arange(len(rotulo)), rotulo] = 1
    return M


def calcula_G(X, U, V):
    # G[k,h] = média de X no bloco (objetos do cluster k) x (variáveis do cluster h)
    return (U.T @ X @ V) / np.outer(U.sum(0), V.sum(0))


def funcao_objetivo(G, U, V):
    # f = ||U G V'||^2 = soma_{k,h} n_k p_h G[k,h]^2
    return float(np.sum(np.outer(U.sum(0), V.sum(0)) * G ** 2))


def atualiza_particao(P, G, tamanhos_outro, xx):
    """Passo de atualização de U (ou de V, com os papéis trocados).

    P:  produtos (n x H) = X V  para U;  (J x K) = X' U  para V
    G:  protótipos já orientados (K x H para U; H x K para V)
    tamanhos_outro: tamanhos dos clusters da OUTRA partição
    xx: ||x||^2 de cada elemento (só usado para tratar cluster vazio)
    """
    k = G.shape[0]
    D = -2 * P @ G.T + (G ** 2) @ tamanhos_outro   # distância sem o termo ||x||^2
    rotulo = np.argmin(D, axis=1)

    # Cluster vazio: recebe o elemento pior ajustado de um cluster com > 1 elemento.
    for vazio in sorted(set(range(k)) - set(rotulo)):
        dist_propria = D[np.arange(len(rotulo)), rotulo] + xx
        doador_ok = np.bincount(rotulo, minlength=k)[rotulo] > 1
        dist_propria[~doador_ok] = -np.inf
        rotulo[np.argmax(dist_propria)] = vazio
    return rotulo


def dkm_uma_execucao(rng, X, K, H, maxiter=200, tol=1e-6):
    xx_linhas, xx_colunas = (X ** 2).sum(1), (X ** 2).sum(0)
    U = indicadora(particao_aleatoria(rng, X.shape[0], K), K)
    V = indicadora(particao_aleatoria(rng, X.shape[1], H), H)
    G = calcula_G(X, U, V)
    f_iter = [funcao_objetivo(G, U, V)]

    for _ in range(maxiter):
        U = indicadora(atualiza_particao(X @ V, G, V.sum(0), xx_linhas), K)
        G = calcula_G(X, U, V)
        V = indicadora(atualiza_particao(X.T @ U, G.T, U.sum(0), xx_colunas), H)
        G = calcula_G(X, U, V)
        f_iter.append(funcao_objetivo(G, U, V))
        if f_iter[-1] - f_iter[-2] <= tol:
            break

    return {"U": U, "V": V, "G": G, "f": f_iter[-1]}


def dkm_melhor_de_n(X, K, H, n_execucoes=100, semente=2026):
    rng = np.random.default_rng(semente)
    execucoes = [dkm_uma_execucao(rng, X, K, H) for _ in range(n_execucoes)]
    return max(execucoes, key=lambda r: r["f"])


# ---------------------------------------------------------------------
# Execução e conferência com as saídas da Questão 1
# ---------------------------------------------------------------------

def main():
    dados = pd.read_csv(PASTA_Q1 / "data" / "spambase.data", header=None)
    X = dados.iloc[:, :57].to_numpy(float)
    # z-score igual ao scale() do R (desvio-padrão com n-1)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)

    # (K*, H*) e f de referência vêm das saídas da Questão 1
    resumo = pd.read_csv(PASTA_Q1 / "output" / "resultados_KH.csv")
    estrela = resumo.loc[resumo["Sil"].idxmax()]
    Kstar, Hstar, f_ref = int(estrela["K"]), int(estrela["H"]), float(estrela["f"])
    print(f"(K*, H*) = ({Kstar}, {Hstar}), f de referência = {f_ref:.4f}")

    melhor = dkm_melhor_de_n(Xs, Kstar, Hstar)
    print(f"f obtido em Python   = {melhor['f']:.4f}")
    assert abs(melhor["f"] - f_ref) < 1e-4 * f_ref, "f diferente do obtido na Questão 1"

    # Alinha a numeração dos clusters de VARIÁVEIS com a do R
    cl_vars_ref = pd.read_csv(PASTA_Q1 / "output" / "clusters_de_variaveis.csv")["cluster_var"].to_numpy() - 1
    cl_vars = melhor["V"].argmax(1)
    perm_v = {h: np.bincount(cl_vars_ref[cl_vars == h], minlength=Hstar).argmax() for h in range(Hstar)}
    cl_vars = np.array([perm_v[h] for h in cl_vars])
    assert np.array_equal(cl_vars, cl_vars_ref), "clusters de variáveis diferentes da Questão 1"

    # Alinha a numeração dos clusters de OBJETOS comparando G com G_prototipos.csv
    G_ref = pd.read_csv(PASTA_Q1 / "output" / "G_prototipos.csv", index_col=0).to_numpy()
    G = melhor["G"][:, [h for h, _ in sorted(perm_v.items(), key=lambda par: par[1])]]
    perm_u = {k: int(np.argmin(np.abs(G_ref - G[k]).sum(1))) for k in range(Kstar)}
    assert len(set(perm_u.values())) == Kstar, "não foi possível alinhar os clusters de objetos"
    G = G[[k for k, _ in sorted(perm_u.items(), key=lambda par: par[1])]]
    assert np.allclose(G, G_ref, atol=1e-3), "matriz G diferente da Questão 1"

    cl_star = np.array([perm_u[k] for k in melhor["U"].argmax(1)]) + 1   # rótulos 1..K* como no R
    print("Tamanho dos clusters:", dict(zip(*np.unique(cl_star, return_counts=True))))

    # Mesmo formato do write.csv do R (cabeçalho entre aspas): o arquivo sai idêntico ao da Questão 1
    pd.DataFrame({"cluster": cl_star}).to_csv(SAIDA, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Conferido com a Questão 1 e salvo em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
