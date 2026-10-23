"""
QUESTÃO 2(b) e 2(c): INTERVALOS DE CONFIANÇA E TESTES DE FRIEDMAN / NEMENYI
==========================================================================

Lê as 300 avaliações de cada classificador (output/resultados_cv_<versao>.csv,
geradas por executar_cv.py) e produz, para cada versão da base:

  b) estimativa pontual (média) e IC de 95% de cada métrica, com a
     correção de Nadeau & Bengio (comum.intervalo_confianca);
  c) teste de Friedman por métrica e pós-teste de Nemenyi.

TESTE DE FRIEDMAN: em cada uma das 300 avaliações (blocos) os 5
classificadores recebem postos 1..5 (1 = melhor; menor taxa de erro ou
maior precisão/cobertura/F-measure; empates recebem o posto médio).
H0: todos os classificadores têm o mesmo posto médio.

PÓS-TESTE DE NEMENYI (Demšar, 2006): se o Friedman rejeita H0, dois
classificadores diferem quando seus postos médios distam mais que a
diferença crítica
    CD = q_alfa * sqrt(k (k + 1) / (6 N)),
com k = 5 classificadores, N = 300 blocos e q_alfa o quantil da
amplitude studentizada dividido por sqrt(2) (q_0.05 = 2.728 para k = 5).

Observação: as 300 avaliações não são independentes (os treinos se
sobrepõem), hipótese que os dois testes assumem; com N = 300 a CD fica
pequena e diferenças pequenas já aparecem como significativas.

Saídas em questao-02/output/:
  ic_<versao>.csv, friedman_<versao>.csv, nemenyi_<versao>.csv,
  hiperparametros_<versao>.csv, cd_<versao>.png e resultados.md (todas as tabelas)

Uso (a partir da raiz do repositório):
    python questao-02/analisar.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import comum
from classificadores import NOMES

PASTA = Path(__file__).resolve().parent / "output"
ALFA = 0.05
ORDEM = list(NOMES)   # gaussiano, knn, parzen, logistica, voto
NOMES_CURTOS = {"gaussiano": "Gaussiano", "knn": "kNN", "parzen": "Parzen",
                "logistica": "Logística", "voto": "Voto"}


def tabela_ic(res):
    linhas = []
    for clf in ORDEM:
        valores = res[res["classificador"] == clf]
        for metrica in comum.METRICAS:
            ic = comum.intervalo_confianca(valores[metrica])
            linhas.append({"classificador": clf, "metrica": metrica, **ic})
    return pd.DataFrame(linhas)


def matriz_por_split(res, metrica):
    """Tabela 300 x 5: valor da métrica de cada classificador em cada split."""
    return res.pivot_table(index=["repeticao", "fold"], columns="classificador",
                           values=metrica)[ORDEM]


def postos(matriz, metrica):
    # Posto 1 = melhor. Na taxa de erro, menor é melhor; nas demais, maior.
    sinal = 1 if metrica == "taxa_erro" else -1
    return (sinal * matriz).rank(axis=1, method="average")


def friedman_nemenyi(res):
    k = len(ORDEM)
    q_alfa = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
    friedman, nemenyi, postos_medios = [], [], {}

    for metrica in comum.METRICAS:
        matriz = matriz_por_split(res, metrica)
        n = len(matriz)
        estat, p = stats.friedmanchisquare(*[matriz[c] for c in ORDEM])
        medios = postos(matriz, metrica).mean()
        cd = q_alfa * np.sqrt(k * (k + 1) / (6 * n))
        postos_medios[metrica] = (medios, cd)
        friedman.append({"metrica": metrica, "N": n, "qui_quadrado": estat, "p_valor": p,
                         "rejeita_H0": p < ALFA, "CD_nemenyi": cd})

        for i, a in enumerate(ORDEM):
            for b in ORDEM[i + 1:]:
                dif = abs(medios[a] - medios[b])
                nemenyi.append({"metrica": metrica, "classificador_a": a, "classificador_b": b,
                                "posto_medio_a": medios[a], "posto_medio_b": medios[b],
                                "diferenca": dif, "CD": cd,
                                "difere": bool(p < ALFA and dif > cd)})
    return pd.DataFrame(friedman), pd.DataFrame(nemenyi), postos_medios


def grafico_cd(postos_medios, versao, arquivo):
    """Diagrama de diferença crítica: postos médios por métrica, com a CD."""
    fig, eixos = plt.subplots(len(comum.METRICAS), 1, figsize=(8, 1.9 * len(comum.METRICAS)))
    for eixo, metrica in zip(eixos, comum.METRICAS):
        medios, cd = postos_medios[metrica]
        ordem = medios.sort_values()
        eixo.hlines(0, 1, len(ORDEM), color="0.6", lw=1)
        eixo.scatter(ordem.values, np.zeros(len(ordem)), color="tab:blue", zorder=3)
        for j, (clf, valor) in enumerate(ordem.items()):
            eixo.annotate(f"{NOMES_CURTOS[clf]}\n{valor:.2f}", (valor, 0), xytext=(0, 12 if j % 2 else -26),
                          textcoords="offset points", ha="center", fontsize=7)
        # Régua com o comprimento da CD: dois classificadores mais distantes
        # que ela têm postos médios significativamente diferentes.
        eixo.hlines(1.05, 1, 1 + cd, color="tab:red", lw=3)
        eixo.text(1 + cd + 0.05, 1.05, f"CD = {cd:.2f}", va="center", fontsize=7, color="tab:red")
        eixo.set_xlim(0.8, len(ORDEM) + 0.2)
        eixo.set_ylim(-1, 1.3)
        eixo.set_yticks([])
        eixo.set_title(f"{comum.NOMES_METRICAS[metrica]} (posto médio, 1 = melhor)", fontsize=9)
    fig.suptitle(f"Friedman / Nemenyi — versão {versao}", fontsize=10)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def tabela_hiperparametros(res):
    """Quantas vezes (em 300) cada configuração foi escolhida na CV interna."""
    hp = res[res["classificador"].isin(["gaussiano", "knn", "parzen", "logistica"])]
    contagem = hp.groupby(["classificador", "hiperparametros"]).size().rename("vezes").reset_index()
    contagem["configuracao"] = contagem["hiperparametros"].map(
        lambda s: ", ".join(f"{k}={v}" for k, v in json.loads(s).items()))
    contagem["classificador"] = pd.Categorical(contagem["classificador"], ORDEM)
    return (contagem.sort_values(["classificador", "vezes"], ascending=[True, False])
            [["classificador", "configuracao", "vezes"]])


# ---------------------------------------------------------------------
# Relatório em Markdown
# ---------------------------------------------------------------------

def md_ic(ic):
    linhas = ["| Classificador | " + " | ".join(comum.NOMES_METRICAS[m] for m in comum.METRICAS) + " |",
              "|---|" + "---|" * len(comum.METRICAS)]
    for clf in ORDEM:
        celulas = []
        for m in comum.METRICAS:
            r = ic[(ic["classificador"] == clf) & (ic["metrica"] == m)].iloc[0]
            celulas.append(f"{r.estimativa:.4f} [{r.ic_inferior:.4f}, {r.ic_superior:.4f}]")
        linhas.append(f"| {NOMES[clf]} | " + " | ".join(celulas) + " |")
    return "\n".join(linhas)


def md_friedman(fr, nem, postos_medios):
    linhas = ["| Métrica | χ² | p-valor | Rejeita H0? | CD | Postos médios (1 = melhor) |",
              "|---|---|---|---|---|---|"]
    for _, r in fr.iterrows():
        medios = postos_medios[r.metrica][0].sort_values()
        txt = ", ".join(f"{NOMES[c]} {v:.2f}" for c, v in medios.items())
        linhas.append(f"| {comum.NOMES_METRICAS[r.metrica]} | {r.qui_quadrado:.1f} | {r.p_valor:.2e} | "
                      f"{'sim' if r.rejeita_H0 else 'não'} | {r.CD_nemenyi:.3f} | {txt} |")
    linhas += ["", "Pares **sem** diferença significativa (Nemenyi, α = 0,05):", ""]
    for m in comum.METRICAS:
        iguais = nem[(nem["metrica"] == m) & ~nem["difere"]]
        pares = "; ".join(f"{NOMES[a]} × {NOMES[b]}" for a, b in
                          zip(iguais["classificador_a"], iguais["classificador_b"])) or "nenhum (todos diferem)"
        linhas.append(f"- {comum.NOMES_METRICAS[m]}: {pares}")
    return "\n".join(linhas)


def md_hiperparametros(hp):
    linhas = ["| Classificador | Configuração | Vezes escolhida (de 300) |", "|---|---|---|"]
    for clf in ORDEM[:-1]:
        top = hp[hp["classificador"] == clf].head(4)
        for _, r in top.iterrows():
            linhas.append(f"| {NOMES[clf]} | {r.configuracao} | {r.vezes} |")
    return "\n".join(linhas)


def main():
    titulos = {"v1": "Versão 1 — rótulo original (spam / não-spam)",
               "v2": "Versão 2 — clusters do double k-means (K* = 2)"}
    md = ["# Resultados da Questão 2", "",
          "Gerado por `analisar.py` a partir de `resultados_cv_<versao>.csv` "
          "(validação cruzada estratificada 30 × 10-folds, 300 avaliações por classificador). "
          "Precisão, cobertura e F-measure são da classe positiva (minoritária): "
          "spam na v1, o cluster de 34 e-mails na v2.", ""]

    for versao in comum.VERSOES:
        res = pd.read_csv(PASTA / f"resultados_cv_{versao}.csv")
        ic = tabela_ic(res)
        fr, nem, postos_medios = friedman_nemenyi(res)
        hp = tabela_hiperparametros(res)

        ic.to_csv(PASTA / f"ic_{versao}.csv", index=False)
        fr.to_csv(PASTA / f"friedman_{versao}.csv", index=False)
        nem.to_csv(PASTA / f"nemenyi_{versao}.csv", index=False)
        hp.to_csv(PASTA / f"hiperparametros_{versao}.csv", index=False)
        grafico_cd(postos_medios, versao, PASTA / f"cd_{versao}.png")

        md += [f"## {titulos[versao]}", "",
               "### b) Estimativa pontual e IC 95% (correção de Nadeau & Bengio)", "",
               md_ic(ic), "",
               "### c) Friedman e Nemenyi", "",
               md_friedman(fr, nem, postos_medios), "",
               f"![Diagrama de diferença crítica](cd_{versao}.png)", "",
               "### Hiper-parâmetros escolhidos na CV interna (mais frequentes)", "",
               md_hiperparametros(hp), ""]
        print(f"[{versao}] tabelas e gráfico gerados")

    (PASTA / "resultados.md").write_text("\n".join(md), encoding="utf-8")
    print("Resumo em output/resultados.md")


if __name__ == "__main__":
    main()
