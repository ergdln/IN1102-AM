"""
QUESTÃO 2(a): VALIDAÇÃO CRUZADA ESTRATIFICADA 30 x 10-FOLDS (ANINHADA)
=====================================================================

Para cada versão da base e para cada um dos 300 splits externos (os mesmos
para todos os classificadores, lidos de questao-02/splits/):

  1. Para cada um dos 4 classificadores base:
       a. se houver hiper-parâmetros, validação cruzada estratificada de
          5 folds DENTRO dos 9 folds de treino; escolhe a configuração de
          maior F-measure média;
       b. re-treina com os 9 folds inteiros usando a configuração escolhida;
       c. prevê o fold de teste (que não participou de a. nem de b.).
  2. Voto majoritário das 4 previsões do passo 1 no mesmo fold de teste.
  3. Calcula taxa de erro, precisão, cobertura e F-measure.

Saídas (questao-02/output/):
  resultados_cv_<versao>.csv  uma linha por (split, classificador): métricas
                              e hiper-parâmetros escolhidos
  grade_cv_<versao>.csv       F-measure média da CV interna de CADA
                              configuração em cada split (detalhe da escolha
                              dos hiper-parâmetros, para o relatório)

O progresso é gravado a cada split: se a execução for interrompida, rodar
de novo continua de onde parou.

Uso (a partir da raiz do repositório):
    python questao-02/executar_cv.py                       # v1 e v2, 30 repetições
    python questao-02/executar_cv.py --versoes v1 --repeticoes 1   # teste rápido
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

import classificadores as clf
import comum

PASTA_SAIDA = Path(__file__).resolve().parent / "output"


def avaliar_split(X, y, repeticao, fold, idx_tr, idx_te):
    """Executa os passos 1-3 num split externo. Devolve (linhas de resultado, linhas da grade)."""
    X_tr, y_tr, X_te, y_te = X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]
    cv = comum.cv_interna(repeticao, fold)
    base = {"repeticao": repeticao, "fold": fold}

    resultados, grade, predicoes, probas = [], [], [], []
    for nome, cls in clf.CLASSIFICADORES.items():
        inicio = time.time()
        cfg, f1_grade = clf.escolher_hiperparametros(cls, X_tr, y_tr, cv)
        modelo = cls(**cfg).fit(X_tr, y_tr)
        predicoes.append(modelo.predict(X_te))
        probas.append(modelo.predict_proba(X_te))

        resultados.append({**base, "classificador": nome,
                           **comum.calcular_metricas(y_te, predicoes[-1]),
                           "hiperparametros": json.dumps(cfg),
                           "segundos": round(time.time() - inicio, 2)})
        if f1_grade is not None:
            grade += [{**base, "classificador": nome, "configuracao": json.dumps(c), "f1_interno": f}
                      for c, f in zip(cls.configuracoes(), f1_grade)]

    voto = clf.voto_majoritario(predicoes, probas, modelo.classes_)
    resultados.append({**base, "classificador": "voto", **comum.calcular_metricas(y_te, voto),
                       "hiperparametros": json.dumps({}), "segundos": 0.0})
    return resultados, grade


def executar(versao, n_repeticoes, n_jobs=-1):
    X, ys = comum.carregar_dados()
    y = ys[versao]
    splits = [s for s in comum.carregar_splits(y, versao) if s[0] <= n_repeticoes]

    PASTA_SAIDA.mkdir(exist_ok=True)
    arq_res = PASTA_SAIDA / f"resultados_cv_{versao}.csv"
    arq_grade = PASTA_SAIDA / f"grade_cv_{versao}.csv"

    feitos = set()
    if arq_res.exists():
        anteriores = pd.read_csv(arq_res)
        feitos = set(zip(anteriores["repeticao"], anteriores["fold"]))
    pendentes = [s for s in splits if (s[0], s[1]) not in feitos]
    print(f"[{versao}] {len(splits)} splits, {len(splits) - len(pendentes)} já feitos, "
          f"{len(pendentes)} a fazer")

    # Os splits de uma mesma repetição rodam em paralelo; o resultado é
    # gravado ao fim de cada repetição.
    inicio, n_feitos = time.time(), 0
    for repeticao in sorted({s[0] for s in pendentes}):
        lote = [s for s in pendentes if s[0] == repeticao]
        saidas = Parallel(n_jobs=n_jobs)(delayed(avaliar_split)(X, y, *s) for s in lote)
        # Grava a grade antes dos resultados: um split só conta como feito
        # quando as duas saídas estão no disco.
        for linhas, arq in ((sum((g for _, g in saidas), []), arq_grade),
                            (sum((r for r, _ in saidas), []), arq_res)):
            pd.DataFrame(linhas).to_csv(arq, mode="a", header=not arq.exists(), index=False)

        n_feitos += len(lote)
        decorrido = time.time() - inicio
        print(f"[{versao}] {n_feitos}/{len(pendentes)} splits | {decorrido / 60:.1f} min | "
              f"restam ~{decorrido / n_feitos * (len(pendentes) - n_feitos) / 60:.1f} min", flush=True)

    return pd.read_csv(arq_res)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--versoes", nargs="+", default=list(comum.VERSOES), choices=comum.VERSOES)
    parser.add_argument("--repeticoes", type=int, default=comum.N_REPETICOES,
                        help="usa só as primeiras N repetições (teste rápido)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="processos em paralelo (-1 = todos os núcleos)")
    args = parser.parse_args()

    for versao in args.versoes:
        res = executar(versao, args.repeticoes, args.n_jobs)
        print(f"\n[{versao}] Média das métricas por classificador:")
        print(res.groupby("classificador", sort=False)[list(comum.METRICAS)].mean().round(4))
        print()


if __name__ == "__main__":
    main()
