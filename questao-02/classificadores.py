"""
OS 5 CLASSIFICADORES DA QUESTÃO 2
=================================

Todos seguem a mesma interface:
    modelo = Classe(**hiperparametros).fit(X_treino, y_treino)
    P = modelo.predict_proba(X_teste)     # (n x 2): P(classe 0 | x), P(classe 1 | x)
    y = modelo.predict(X_teste)

e declaram a GRADE de hiper-parâmetros avaliada na validação cruzada
interna de 5 folds (vazia para quem não tem hiper-parâmetros).

PADRONIZAÇÃO: os classificadores baseados em distância (k-vizinhos, Parzen)
e a regressão logística padronizam os atributos (z-score) DENTRO do fit,
com média e desvio-padrão do conjunto de treino. Assim a padronização é
reaprendida em cada fold e nunca usa informação do conjunto de teste.

As contas das densidades são feitas em LOGARITMO: com 57 atributos as
densidades ficam da ordem de 1e-100 ou menores, e multiplicá-las em escala
natural dá underflow (vira 0). P(w_i | x) é obtido com o truque log-sum-exp.
"""

from itertools import product

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


class _Base:
    """Partes comuns: classes, priors de MV e regra de decisão de Bayes."""

    GRADE = {}   # hiper-parâmetro -> valores candidatos

    def _priors(self, y):
        # Estimativa de máxima verossimilhança de P(w_i): n_i / n
        self.classes_, contagens = np.unique(y, return_counts=True)
        self.log_priors_ = np.log(contagens / len(y))

    def _posteriores(self, log_conjunta):
        # P(w_i | x) = p(x | w_i) P(w_i) / soma_r p(x | w_r) P(w_r), em log
        return np.exp(log_conjunta - logsumexp(log_conjunta, axis=1, keepdims=True))

    def predict(self, X):
        # Regra de Bayes: classe de maior probabilidade a posteriori
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @classmethod
    def configuracoes(cls):
        """Todas as combinações da grade, como lista de dicionários."""
        nomes = list(cls.GRADE)
        return [dict(zip(nomes, valores)) for valores in product(*cls.GRADE.values())]

    @classmethod
    def prever_grade(cls, X_tr, y_tr, X_val):
        """Previsões em X_val para cada configuração da grade.

        Versão genérica (treina uma vez por configuração). Parzen e
        k-vizinhos sobrescrevem para reaproveitar as distâncias, que não
        dependem do hiper-parâmetro.
        """
        return [cls(**cfg).fit(X_tr, y_tr).predict(X_val) for cfg in cls.configuracoes()]


# ---------------------------------------------------------------------
# i) Bayesiano gaussiano
# ---------------------------------------------------------------------

class BayesGaussiano(_Base):
    """p(x | w_i) = normal multivariada com média e covariância de MV.

    Covariância de máxima verossimilhança (divide por n_i, bias=True).
    Na Spambase a covariância de cada classe é quase singular (número de
    condição ~1e11) e, na versão 2, a classe de 34 e-mails tem menos
    exemplos que atributos, então a covariância é singular e não pode ser
    invertida. Somamos reg * I à covariância dos dados padronizados (toda
    variável com variância 1, então reg é a fração de variância somada a
    cada uma). O valor de reg muda o resultado, por isso é escolhido por
    validação cruzada; reg = 1e-6 é praticamente a MV pura.
    """

    GRADE = {"reg": [1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]}   # do mais regularizado ao menos

    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, X, y):
        self._priors(y)
        self.escala_ = StandardScaler().fit(X)
        Z = self.escala_.transform(X)
        d = Z.shape[1]

        self.medias_, self.fatores_, self.log_dets_ = [], [], []
        for c in self.classes_:
            Zc = Z[y == c]
            cov = np.cov(Zc, rowvar=False, bias=True) + self.reg * np.eye(d)
            fator = cho_factor(cov, lower=True)
            self.medias_.append(Zc.mean(axis=0))
            self.fatores_.append(fator)
            self.log_dets_.append(2 * np.sum(np.log(np.diag(fator[0]))))   # log |Sigma_i|
        return self

    def predict_proba(self, X):
        Z = self.escala_.transform(X)
        d = Z.shape[1]
        colunas = []
        for media, fator, log_det, log_prior in zip(self.medias_, self.fatores_,
                                                     self.log_dets_, self.log_priors_):
            dif = Z - media
            mahalanobis = np.sum(dif * cho_solve(fator, dif.T).T, axis=1)   # (x-mu)' Sigma^-1 (x-mu)
            log_dens = -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)
            colunas.append(log_dens + log_prior)
        return self._posteriores(np.column_stack(colunas))


# ---------------------------------------------------------------------
# ii) Bayesiano baseado em k-vizinhos
# ---------------------------------------------------------------------

class BayesKNN(_Base):
    """Estimativa de densidade por k-vizinhos.

    Seja V o volume da menor região em torno de x que contém k exemplos de
    treino, dos quais k_i são da classe w_i. Então
        p(x | w_i) ~ k_i / (n_i V),   P(w_i) = n_i / n
    e a regra de Bayes fica
        P(w_i | x) = k_i / k.
    A "região" depende da distância: euclidiana (bola), city-block
    (losango) ou Chebyshev (cubo). k e a distância são hiper-parâmetros.

    EMPATES DE DISTÂNCIA: a Spambase tem 394 linhas duplicadas e, com a
    distância de Chebyshev, muitos pontos ficam exatamente à mesma
    distância. Se houver empate na k-ésima distância, "os k mais próximos"
    seria ambíguo. Seguimos a definição: a região é a menor que contém k
    exemplos, então TODOS os exemplos empatados na borda estão dentro dela
    (k' >= k exemplos) e P(w_i | x) = k_i / k'. Se ainda assim k_0 = k_1,
    vence a classe de maior prior.
    """

    GRADE = {"metrica": ["euclidean", "cityblock", "chebyshev"],
             "k": [31, 21, 15, 11, 9, 7, 5, 3, 1]}   # do mais suave ao mais complexo
    LOTE = 512

    def __init__(self, k=5, metrica="euclidean"):
        self.k, self.metrica = k, metrica

    def fit(self, X, y):
        self._priors(y)
        self.escala_ = StandardScaler().fit(X)
        self.Z_ = self.escala_.transform(X)
        self.y_ = np.searchsorted(self.classes_, y)   # índices 0..c-1
        return self

    def _probas(self, X, metrica, ks):
        """P(w_i | x) para cada k em ks: array (len(ks), n, c)."""
        Z = self.escala_.transform(X)
        saida = np.empty((len(ks), len(Z), len(self.classes_)))
        for inicio in range(0, len(Z), self.LOTE):
            lote = slice(inicio, inicio + self.LOTE)
            D = cdist(Z[lote], self.Z_, metric=metrica)
            D_ordenada = np.sort(D, axis=1)
            for j, k in enumerate(ks):
                raio = D_ordenada[:, k - 1:k]                  # distância do k-ésimo vizinho
                dentro = D <= raio + 1e-9 * (1 + raio)          # inclui os empatados na borda
                k_i = np.column_stack([(dentro & (self.y_ == i)).sum(axis=1)
                                       for i in range(len(self.classes_))])
                saida[j, lote] = k_i / k_i.sum(axis=1, keepdims=True)
        return saida

    def _decide(self, proba):
        # Classe de maior k_i / k'; em empate, a de maior prior.
        return self.classes_[np.argmax(proba + 1e-9 * np.exp(self.log_priors_), axis=1)]

    def predict_proba(self, X):
        return self._probas(X, self.metrica, [self.k])[0]

    def predict(self, X):
        return self._decide(self.predict_proba(X))

    @classmethod
    def prever_grade(cls, X_tr, y_tr, X_val):
        # Para cada distância, calcula e ordena as distâncias UMA vez e avalia todos os k.
        modelo = cls().fit(X_tr, y_tr)
        previsoes = {}
        for metrica in cls.GRADE["metrica"]:
            for k, proba in zip(cls.GRADE["k"], modelo._probas(X_val, metrica, cls.GRADE["k"])):
                previsoes[(metrica, k)] = modelo._decide(proba)
        return [previsoes[(cfg["metrica"], cfg["k"])] for cfg in cls.configuracoes()]


# ---------------------------------------------------------------------
# iii) Bayesiano baseado em janela de Parzen
# ---------------------------------------------------------------------

class BayesParzen(_Base):
    """Janela de Parzen com kernel multivariado produto de gaussianas univariadas.

        p(x | w_i) = 1/n_i  soma_{x_j em w_i}  prod_{l=1..d} (1/h) K((x_l - x_jl) / h),
        K(u) = exp(-u^2 / 2) / sqrt(2 pi)

    Com a mesma janela h em todos os atributos (padronizados), o produto
    dos kernels é  (2 pi)^(-d/2) h^(-d) exp(-||x - x_j||^2 / (2 h^2)).
    h é escolhido por validação cruzada.
    """

    GRADE = {"h": [5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}   # do mais suave ao mais complexo
    LOTE = 512   # linhas de X processadas por vez (limita a memória)

    def __init__(self, h=0.5):
        self.h = h

    def fit(self, X, y):
        self._priors(y)
        self.escala_ = StandardScaler().fit(X)
        Z = self.escala_.transform(X)
        self.dados_ = [Z[y == c] for c in self.classes_]
        return self

    def _log_conjunta(self, X, hs):
        """log[p(x | w_i) P(w_i)] para cada h em hs: array (len(hs), n, c)."""
        Z = self.escala_.transform(X)
        d = Z.shape[1]
        saida = np.empty((len(hs), len(Z), len(self.classes_)))
        for inicio in range(0, len(Z), self.LOTE):
            lote = slice(inicio, inicio + self.LOTE)
            for i, (Zc, log_prior) in enumerate(zip(self.dados_, self.log_priors_)):
                dist2 = cdist(Z[lote], Zc, metric="sqeuclidean")   # não depende de h
                for j, h in enumerate(hs):
                    log_dens = (logsumexp(-dist2 / (2 * h ** 2), axis=1) - np.log(len(Zc))
                                - d * np.log(h) - 0.5 * d * np.log(2 * np.pi))
                    saida[j, lote, i] = log_dens + log_prior
        return saida

    def predict_proba(self, X):
        return self._posteriores(self._log_conjunta(X, [self.h])[0])

    @classmethod
    def prever_grade(cls, X_tr, y_tr, X_val):
        # As distâncias são calculadas uma vez e reaproveitadas para todos os h.
        modelo = cls().fit(X_tr, y_tr)
        log_conj = modelo._log_conjunta(X_val, cls.GRADE["h"])
        return [modelo.classes_[np.argmax(lc, axis=1)] for lc in log_conj]


# ---------------------------------------------------------------------
# iv) Regressão logística
# ---------------------------------------------------------------------

class RegressaoLogistica(_Base):
    """P(w_1 | x) = 1 / (1 + exp(-(w'x + b))), ajustada por máxima verossimilhança
    com penalidade L2 (ridge). Hiper-parâmetro: C, o inverso da força da
    regularização (C pequeno = coeficientes mais encolhidos, modelo mais
    simples; C grande = quase sem regularização).

    A penalidade L1 foi testada e descartada: com C grande o otimizador
    (liblinear) não converge em alguns folds desta base (um único ajuste
    passou de 6 minutos sem convergir) e os coeficientes obtidos mudam muito
    com a tolerância. Com L2 o problema é estritamente convexo e o lbfgs
    converge sempre, em centésimos de segundo.
    """

    GRADE = {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]}   # do mais regularizado ao menos

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self._priors(y)
        self.escala_ = StandardScaler().fit(X)
        self.modelo_ = LogisticRegression(C=self.C, solver="lbfgs", max_iter=5000)
        self.modelo_.fit(self.escala_.transform(X), y)
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(self.escala_.transform(X))


# ---------------------------------------------------------------------
# v) Voto majoritário
# ---------------------------------------------------------------------

def voto_majoritario(predicoes, probas, classes):
    """Combina os 4 classificadores pela regra do voto majoritário.

    predicoes: lista com a classe prevista por cada classificador (seu voto);
    probas:    lista com a matriz (n x c) de P(w_i | x) de cada classificador.
    Vence a classe mais votada. Com 4 votantes e 2 classes pode haver
    empate (2 x 2): nesse caso decide a MÉDIA das probabilidades a
    posteriori dos 4.
    """
    classes = np.asarray(classes)
    votos = np.zeros_like(probas[0])
    for pred in predicoes:
        votos[np.arange(len(pred)), np.searchsorted(classes, pred)] += 1
    media = np.mean(probas, axis=0)
    # Ordena primeiro pelos votos e, entre classes empatadas, pela média.
    criterio = votos + media / (len(probas) + 1)
    return classes[np.argmax(criterio, axis=1)]


# ---------------------------------------------------------------------
# Seleção de hiper-parâmetros por validação cruzada interna
# ---------------------------------------------------------------------

def escolher_hiperparametros(cls, X, y, cv):
    """Escolhe a configuração de maior F-measure média na CV interna.

    X, y: os 9 folds de treino do split externo; cv: StratifiedKFold de 5.
    Devolve (melhor configuração, F-measure média de cada configuração).

    EMPATES: as grades estão ordenadas do modelo mais simples (mais suave,
    mais regularizado) para o mais complexo, e em caso de empate fica a
    PRIMEIRA configuração, ou seja, a mais simples. Na versão 2 isso é
    frequente: com ~6 positivos por fold interno a F-measure assume poucos
    valores e várias configurações empatam no máximo.
    """
    configuracoes = cls.configuracoes()
    if len(configuracoes) == 1:          # sem hiper-parâmetros
        return configuracoes[0], None

    f1 = np.zeros((cv.get_n_splits(), len(configuracoes)))
    for i, (idx_tr, idx_val) in enumerate(cv.split(X, y)):
        previsoes = cls.prever_grade(X[idx_tr], y[idx_tr], X[idx_val])
        f1[i] = [f1_score(y[idx_val], p, pos_label=1, zero_division=0) for p in previsoes]

    f1_medio = f1.mean(axis=0)
    return configuracoes[int(np.argmax(f1_medio))], f1_medio


CLASSIFICADORES = {
    "gaussiano": BayesGaussiano,
    "knn": BayesKNN,
    "parzen": BayesParzen,
    "logistica": RegressaoLogistica,
}
NOMES = {
    "gaussiano": "Bayesiano gaussiano",
    "knn": "Bayesiano k-vizinhos",
    "parzen": "Bayesiano Parzen",
    "logistica": "Regressão logística",
    "voto": "Voto majoritário",
}
