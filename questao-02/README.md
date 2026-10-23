# Questão 2 — Classificadores na Spambase

Compara 5 classificadores em duas versões da base Spambase (4601 e-mails × 57 atributos):

| Versão | Variável resposta | Classe positiva (1) |
|---|---|---|
| **v1** | rótulo original: spam / não-spam | spam (1813 e-mails, 39,4%) |
| **v2** | cluster de objetos do double k-means da Questão 1 (K\* = 2) | cluster de 34 e-mails (0,74%) |

Os rótulos da v2 vêm de `double-kmeans-spambase/output/cl_star.csv`, gerado pelo script R da Questão 1.

**Resultados:** [`output/resultados.md`](output/resultados.md) (tabelas de IC, Friedman/Nemenyi e hiper-parâmetros), [`output/curva_v1.png`](output/curva_v1.png) e [`output/curva_v2.png`](output/curva_v2.png) (curvas de aprendizagem).

## Principais resultados

F-measure média (300 avaliações, IC 95% em `output/resultados.md`):

| | Gaussiano | k-vizinhos | Parzen | Logística | Voto |
|---|---|---|---|---|---|
| **v1** | 0,815 | 0,888 | 0,897 | 0,906 | **0,919** |
| **v2** | 0,744 | 0,943 | 0,977 | 0,963 | **0,986** |

- **v1.** O voto majoritário é o melhor em taxa de erro (6,4%) e F-measure; o Friedman rejeita H0 nas 4 métricas e,
  pelo Nemenyi, todos os pares diferem em erro e F-measure. O gaussiano é o pior: alta cobertura (0,95) e baixa
  precisão (0,71), porque as variáveis (frequências cheias de zeros e muito assimétricas) estão longe da normal.
- **v2.** Todos acertam quase tudo (erro < 0,4%), pois os rótulos são uma partição do próprio X. As diferenças entre
  k-vizinhos, Parzen, logística e voto são pequenas e várias não são significativas; o gaussiano é claramente pior,
  porque estima uma covariância 57 × 57 com ~30 exemplos da classe minoritária. Os IC são largos: cada fold de teste
  tem só 3 ou 4 positivos, e um erro muda a F-measure do fold em ~0,15–0,25.
- **Curvas (d).** Gaussiano na v1: treino e teste convergem para ~0,81, ou seja, erro de viés (modelo inadequado);
  mais dados não ajudam. k-vizinhos e Parzen: F-measure de treino ~1 (memorizam o treino; k = 1 e h pequeno) e de
  teste crescendo com o tamanho do treino, ou seja, variância alta que diminui com mais dados. Logística: a diferença
  entre treino e teste fecha a partir de ~50%. Na v2 as curvas de teste são ruidosas, porque o conjunto de teste tem
  poucos positivos (com 95% de treino, só 2).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `comum.py` | Dados, as duas versões da resposta, splits, métricas e intervalo de confiança |
| `classificadores.py` | Os 5 classificadores e a seleção de hiper-parâmetros |
| `executar_cv.py` | Item (a): validação cruzada aninhada 30 × 10-folds |
| `analisar.py` | Itens (b) e (c): IC 95%, Friedman e Nemenyi |
| `curva_aprendizagem.py` | Item (d): curvas de aprendizagem da F-measure |
| `reconstroi_cl_star.py` | Gera `cl_star.csv` sem R (double k-means em Python, conferido com as saídas da Questão 1) |
| `splits/folds_<versao>.csv` | Os 30 × 10 folds sorteados (fold de teste de cada e-mail em cada repetição) |
| `output/` | Resultados (ver abaixo) |

## Como rodar

A partir da raiz do repositório, com `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` e `joblib`:

```bash
python questao-02/executar_cv.py          # (a)  ~10 min com 10 núcleos; retoma de onde parou
python questao-02/analisar.py             # (b) e (c)  alguns segundos
python questao-02/curva_aprendizagem.py   # (d)  ~1 min
```

Teste rápido: `python questao-02/executar_cv.py --repeticoes 1`.

## Organização dos experimentos

**Validação cruzada aninhada (item a).** Para cada versão, os 30 × 10 folds estratificados são sorteados uma vez
(semente 2026) e salvos em `splits/`. Todos os classificadores são avaliados nos **mesmos 300 splits**, o que é
exigido pelo voto majoritário (combina as previsões no mesmo fold de teste) e pelo Friedman/Nemenyi (compara os
classificadores split a split). Em cada split externo:

1. hiper-parâmetros escolhidos por validação cruzada estratificada de 5 folds nos 9 folds de treino, pela maior
   F-measure média (os 5 folds internos são os mesmos para todos os classificadores);
2. re-treino com os 9 folds usando a configuração escolhida;
3. avaliação no fold de teste, que não participa de 1 nem de 2.

A padronização (z-score) é ajustada dentro de cada treino, nunca com dados de teste.

**Métricas.** Taxa de erro, e precisão, cobertura e F-measure da classe positiva (minoritária).

**IC (item b).** Média das 300 avaliações e IC 95% com a correção de Nadeau & Bengio
(EP = dp · √(1/300 + 1/9)), porque as avaliações da CV repetida não são independentes.

**Friedman e Nemenyi (item c).** Postos dos 5 classificadores em cada uma das 300 avaliações; Friedman para
H0 "postos médios iguais"; se rejeitada, Nemenyi com CD = q₀,₀₅ · √(k(k+1)/6N), q₀,₀₅ = 2,728 (k = 5).

**Curvas (item d).** Treino de 5% a 95% (passo 5%), divisão estratificada, 5 sorteios por proporção;
F-measure no treino e no teste; hiper-parâmetros fixos na configuração mais escolhida no item (a).

## Classificadores e hiper-parâmetros

| Classificador | Modelo | Hiper-parâmetros (grade) |
|---|---|---|
| Bayesiano gaussiano | normal multivariada por classe, média e covariância de MV | `reg` somado à diagonal da covariância: 10⁻⁶ a 1 |
| Bayesiano k-vizinhos | P(ωᵢ\|x) = kᵢ/k | k ∈ {1, 3, …, 31}; distância euclidiana, city-block ou Chebyshev |
| Bayesiano Parzen | kernel produto de gaussianas univariadas | janela h: 0,1 a 5 |
| Regressão logística | penalidade L2 | C: 0,001 a 1000 |
| Voto majoritário | classe mais votada entre os 4 | — |

## Decisões de implementação

- **Gaussiano — regularização.** A covariância é quase singular na v1 (número de condição ~10¹¹) e singular na v2
  (a classe de 34 e-mails tem menos exemplos que atributos). Soma-se `reg` à diagonal da covariância dos dados
  padronizados. Como o valor muda o resultado, ele é escolhido por CV; `reg` = 10⁻⁶ é praticamente a MV pura.
- **k-vizinhos — empates de distância.** A base tem 394 linhas duplicadas e a distância de Chebyshev gera muitos
  empates. Todos os exemplos empatados na k-ésima distância entram na vizinhança (k' ≥ k), e P(ωᵢ\|x) = kᵢ/k'.
- **Regressão logística — só L2.** A penalidade L1 foi testada e descartada: com C grande o otimizador não converge
  em alguns folds (um ajuste passou de 6 minutos sem convergir).
- **Empates na seleção de hiper-parâmetros.** Na v2 há só ~6 positivos por fold interno e várias configurações
  empatam na F-measure. As grades estão ordenadas do modelo mais simples ao mais complexo e, em empate, vence o mais
  simples (maior `reg`, maior k, maior h, menor C).
- **Voto — empate 2 × 2.** Decidido pela média das probabilidades a posteriori dos 4 classificadores.

## Saídas (`output/`)

| Arquivo | Conteúdo |
|---|---|
| `resultados_cv_<versao>.csv` | Métricas e hiper-parâmetros escolhidos, por split e classificador |
| `grade_cv_<versao>.csv` | F-measure interna de cada configuração em cada split |
| `ic_<versao>.csv` | Estimativa pontual e IC 95% |
| `friedman_<versao>.csv`, `nemenyi_<versao>.csv` | Testes estatísticos |
| `cd_<versao>.png` | Diagrama de diferença crítica |
| `hiperparametros_<versao>.csv` | Frequência de cada configuração escolhida |
| `curva_<versao>.csv`, `curva_<versao>.png` | Curvas de aprendizagem |
| `resultados.md` | Todas as tabelas acima, em um arquivo |
