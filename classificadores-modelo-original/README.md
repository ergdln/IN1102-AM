# Experimentos preliminares da Questão 2

Esta pasta guarda os primeiros experimentos da Questão 2, feitos no Google Colab para a versão 1 da base
(rótulo original spam / não-spam):

- `bayesiano-gaussiano/`: classificador bayesiano gaussiano, resultados e curva de aprendizagem;
- `bayesiano-janela-parsen/`: classificador bayesiano por janela de Parzen, com validação aninhada;
- `relatórios_classificadores/`: relatórios parciais desses dois classificadores.

**Os resultados finais do projeto estão na pasta [`questao-02/`](../questao-02/)**, que avalia os 5
classificadores nas duas versões da base com a mesma validação cruzada 30 × 10-folds.

Os números desta pasta diferem dos finais por dois motivos:

1. Cada classificador foi avaliado com uma divisão própria dos folds; na versão final todos usam os mesmos 300
   splits, o que é necessário para o voto majoritário e para o teste de Friedman/Nemenyi.
2. O gaussiano desta pasta usa uma inversão da covariância numericamente instável (taxa de erro de 21,8%); na
   versão final a covariância é regularizada e a taxa de erro é 17,1%.
