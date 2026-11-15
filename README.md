# Projeto AM 2026-2 — Spambase

Projeto da disciplina IN1102 — Aprendizagem de Máquina (CIn/UFPE), Grupo 1:
Ernesto Gonçalves, Fernanda Pascoal e Raissa Bispo.

Base de dados: [Spambase (UCI)](https://archive.ics.uci.edu/dataset/94/spambase), 4601 e-mails × 57 atributos,
rotulados como spam ou não-spam.

| Pasta | Conteúdo |
|---|---|
| [`double-kmeans-spambase/`](double-kmeans-spambase/) | **Questão 1**: double k-means com K ∈ {2, 3, 4} e H ≤ K, escolha de K\* pela silhueta, índice de Rand corrigido, matriz G, matriz de confusão e função objetivo por iteração |
| [`questao-02/`](questao-02/) | **Questão 2**: 5 classificadores (bayesiano gaussiano, k-vizinhos, Parzen, regressão logística e voto majoritário) nas duas versões da base, com validação cruzada 30 × 10-folds, intervalos de confiança, Friedman/Nemenyi e curvas de aprendizagem. Detalhes em [`questao-02/README.md`](questao-02/README.md) |
| [`classificadores-modelo-original/`](classificadores-modelo-original/) | Experimentos preliminares da Questão 2 (os resultados finais estão em `questao-02/`) |

## Como rodar

**Questão 1** (R, com os pacotes `cluster` e `ggplot2`; `drclust` é opcional e só confere o resultado):

```bash
cd double-kmeans-spambase
Rscript double_kmeans_spambase.R
```

As saídas vão para `double-kmeans-spambase/output/`, incluindo `cl_star.csv`, que é a variável resposta da
versão 2 da base na Questão 2.

**Questão 2** (Python, com `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` e `joblib`): pelo notebook
[`questao-02/questao2.ipynb`](questao-02/questao2.ipynb), que também abre no Colab, ou pelos scripts descritos em
[`questao-02/README.md`](questao-02/README.md).
