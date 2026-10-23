"""
QUESTÃO 2(d): CURVAS DE APRENDIZAGEM (F-MEASURE)
================================================

Para cada versão da base e cada proporção de treinamento p = 5%, 10%, ...,
95% (teste = 1 - p), com divisão estratificada:
  - treina os 5 classificadores no conjunto de treinamento;
  - calcula a F-measure (classe positiva) no próprio treinamento e no teste.
Cada proporção é repetida N_REPETICOES vezes com sorteios diferentes; o
gráfico mostra a média e a faixa de +-1 desvio-padrão.

HIPER-PARÂMETROS: fixos em cada proporção, iguais à configuração mais
escolhida na validação cruzada 30 x 10 do item (a) (lida de
output/resultados_cv_<versao>.csv; rode executar_cv.py antes). Escolher por
CV em cada tamanho não é possível nos menores: com 5% de treino a classe
minoritária da versão 2 tem só 2 exemplos, e não há como formar 5 folds
estratificados.

Saídas em questao-02/output/: curva_<versao>.csv e curva_<versao>.png

Uso (a partir da raiz do repositório):
    python questao-02/curva_aprendizagem.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

import classificadores as clf
import comum

PASTA = Path(__file__).resolve().parent / "output"
PROPORCOES = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_REPETICOES = 5


def hiperparametros_mais_escolhidos(versao):
    res = pd.read_csv(PASTA / f"resultados_cv_{versao}.csv")
    return {nome: json.loads(res.loc[res["classificador"] == nome, "hiperparametros"].mode()[0])
            for nome in clf.CLASSIFICADORES}


def avaliar_proporcao(X, y, proporcao, repeticao, configs):
    divisao = StratifiedShuffleSplit(n_splits=1, train_size=proporcao,
                                     random_state=comum.SEMENTE + repeticao)
    idx_tr, idx_te = next(divisao.split(X, y))
    conjuntos = {"treino": idx_tr, "teste": idx_te}

    linhas, preds, probas = [], {c: [] for c in conjuntos}, {c: [] for c in conjuntos}
    for nome, cls in clf.CLASSIFICADORES.items():
        modelo = cls(**configs[nome]).fit(X[idx_tr], y[idx_tr])
        for conjunto, idx in conjuntos.items():
            preds[conjunto].append(modelo.predict(X[idx]))
            probas[conjunto].append(modelo.predict_proba(X[idx]))
            f1 = comum.calcular_metricas(y[idx], preds[conjunto][-1])["f_measure"]
            linhas.append({"classificador": nome, "conjunto": conjunto, "f_measure": f1})

    for conjunto, idx in conjuntos.items():
        voto = clf.voto_majoritario(preds[conjunto], probas[conjunto], modelo.classes_)
        linhas.append({"classificador": "voto", "conjunto": conjunto,
                       "f_measure": comum.calcular_metricas(y[idx], voto)["f_measure"]})

    for linha in linhas:
        linha.update(proporcao_treino=proporcao, repeticao=repeticao + 1,
                     n_treino=len(idx_tr), n_teste=len(idx_te))
    return linhas


def grafico(curva, versao, arquivo):
    resumo = curva.groupby(["classificador", "conjunto", "proporcao_treino"])["f_measure"].agg(["mean", "std"])
    fig, eixos = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    cores = {"treino": "tab:blue", "teste": "tab:orange"}

    for eixo, nome in zip(eixos.flat, clf.NOMES):
        for conjunto, cor in cores.items():
            r = resumo.loc[(nome, conjunto)]
            x = r.index * 100
            eixo.plot(x, r["mean"], marker="o", ms=3, color=cor, label=conjunto)
            eixo.fill_between(x, r["mean"] - r["std"], r["mean"] + r["std"], color=cor, alpha=0.2)
        eixo.set_title(clf.NOMES[nome], fontsize=10)
        eixo.grid(alpha=0.3)

    # Último painel: só as curvas de teste, para comparar os classificadores.
    eixo = eixos.flat[-1]
    for nome in clf.NOMES:
        r = resumo.loc[(nome, "teste")]
        eixo.plot(r.index * 100, r["mean"], marker="o", ms=3, label=clf.NOMES[nome])
    eixo.set_title("Teste: comparação", fontsize=10)
    eixo.grid(alpha=0.3)
    eixo.legend(fontsize=7)

    eixos.flat[0].legend(fontsize=8)
    for eixo in eixos[-1]:
        eixo.set_xlabel("Conjunto de treinamento (%)")
    for eixo in eixos[:, 0]:
        eixo.set_ylabel("F-measure")
    fig.suptitle(f"Curvas de aprendizagem — versão {versao} "
                 f"(média de {N_REPETICOES} sorteios, faixa = ±1 desvio-padrão)")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def main():
    X, ys = comum.carregar_dados()
    for versao in comum.VERSOES:
        configs = hiperparametros_mais_escolhidos(versao)
        print(f"[{versao}] hiper-parâmetros fixos: {configs}")

        tarefas = [(p, r) for p in PROPORCOES for r in range(N_REPETICOES)]
        saidas = Parallel(n_jobs=-1)(delayed(avaliar_proporcao)(X, ys[versao], p, r, configs)
                                     for p, r in tarefas)
        curva = pd.DataFrame([linha for linhas in saidas for linha in linhas])
        curva.to_csv(PASTA / f"curva_{versao}.csv", index=False)
        grafico(curva, versao, PASTA / f"curva_{versao}.png")

        resumo = curva.pivot_table(index="proporcao_treino", columns=["classificador", "conjunto"],
                                   values="f_measure").round(3)
        print(resumo.iloc[[0, 3, 9, 18]].to_string(), "\n")


if __name__ == "__main__":
    main()
