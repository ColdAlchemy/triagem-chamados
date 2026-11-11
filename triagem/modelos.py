"""Modelos de classificação de texto escritos do zero (só biblioteca padrão).

Todos seguem a mesma interface: fit(textos, etiquetas) e proba(texto) -> {classe: probabilidade}.
"""
import math
import random
from collections import Counter

from .texto import tokens


def _softmax(pontos: dict[str, float]) -> dict[str, float]:
    m = max(pontos.values())
    exps = {c: math.exp(p - m) for c, p in pontos.items()}
    z = sum(exps.values())
    return {c: e / z for c, e in exps.items()}


class Modelo:
    nome = "modelo"
    classes: list[str]

    def fit(self, textos: list[str], ys: list[str]) -> "Modelo":
        raise NotImplementedError

    def proba(self, texto: str) -> dict[str, float]:
        raise NotImplementedError

    def prever(self, texto: str) -> str:
        p = self.proba(texto)
        return max(self.classes, key=lambda c: (p[c], -self.classes.index(c)))


class Majoritario(Modelo):
    """Linha de base: sempre a classe mais frequente no treino."""

    nome = "majoritario"

    def fit(self, textos, ys):
        self.classes = sorted(set(ys))
        contagem = Counter(ys)
        self.freq = {c: contagem[c] / len(ys) for c in self.classes}
        return self

    def proba(self, texto):
        return dict(self.freq)


REGRAS = {
    "acesso": ["senha", "login", "conta", "acessar", "entrar", "codigo", "usuario", "desconectada"],
    "cobranca": ["cobranca", "cobrado", "cobrando", "fatura", "boleto", "estorno", "pagamento", "cartao", "extrato", "desconto"],
    "entrega": ["pedido", "entrega", "rastreio", "chegou", "entregador", "endereco", "prazo"],
    "defeito": ["erro", "trava", "liga", "tela", "bateria", "fecha", "esquenta", "som", "falha"],
    "sugestao": ["sugiro", "gostaria", "seria", "poderia", "adorei", "parabens", "queria", "podiam", "otimo"],
}


class Regras(Modelo):
    """Linha de base de um sistema por palavras-chave escritas à mão: não aprende nada dos dados."""

    nome = "regras"

    def fit(self, textos, ys):
        self.classes = sorted(set(ys))
        return self

    def proba(self, texto):
        palavras = set(tokens(texto, bigramas=False))
        acertos = {c: sum(1 for k in REGRAS.get(c, []) if k in palavras) for c in self.classes}
        total = sum(acertos.values())
        if total == 0:
            return {c: 1 / len(self.classes) for c in self.classes}
        return {c: a / total for c, a in acertos.items()}


class NaiveBayes(Modelo):
    """Naive Bayes multinomial com suavização de Laplace (alfa)."""

    nome = "naive-bayes"

    def __init__(self, alfa: float = 0.5):
        self.alfa = alfa

    def fit(self, textos, ys):
        self.classes = sorted(set(ys))
        n = len(ys)
        contagem_classe = Counter(ys)
        self.log_prior = {c: math.log(contagem_classe[c] / n) for c in self.classes}
        self.cont = {c: Counter() for c in self.classes}
        for t, y in zip(textos, ys):
            self.cont[y].update(tokens(t))
        self.vocab = set().union(*(set(c) for c in self.cont.values()))
        self.total = {c: sum(self.cont[c].values()) for c in self.classes}
        return self

    def proba(self, texto):
        v = len(self.vocab)
        toks = [t for t in tokens(texto) if t in self.vocab]  # palavra nunca vista não informa nada
        pontos = {}
        for c in self.classes:
            den = self.total[c] + self.alfa * v
            pontos[c] = self.log_prior[c] + sum(math.log((self.cont[c][t] + self.alfa) / den) for t in toks)
        return _softmax(pontos)


class Logistica(Modelo):
    """Regressão logística multiclasse (softmax) sobre TF-IDF, treinada com gradiente descendente estocástico."""

    nome = "logistica"

    def __init__(self, epocas: int = 25, taxa: float = 0.6, l2: float = 1e-4, semente: int = 3):
        self.epocas, self.taxa, self.l2, self.semente = epocas, taxa, l2, semente

    def _vetor(self, texto: str) -> dict[str, float]:
        contagem = Counter(t for t in tokens(texto) if t in self.idf)
        v = {t: (1 + math.log(f)) * self.idf[t] for t, f in contagem.items()}
        norma = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norma for t, x in v.items()}

    def fit(self, textos, ys):
        self.classes = sorted(set(ys))
        n = len(textos)
        df: Counter = Counter()
        for t in textos:
            df.update(set(tokens(t)))
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.w = {c: {} for c in self.classes}
        self.b = {c: 0.0 for c in self.classes}
        X = [self._vetor(t) for t in textos]
        ordem = list(range(n))
        rng = random.Random(self.semente)
        for ep in range(self.epocas):
            rng.shuffle(ordem)
            lr = self.taxa / (1 + 0.1 * ep)
            for i in ordem:
                x, y = X[i], ys[i]
                p = _softmax({c: self.b[c] + sum(self.w[c].get(t, 0.0) * v for t, v in x.items()) for c in self.classes})
                for c in self.classes:
                    g = p[c] - (1.0 if c == y else 0.0)
                    self.b[c] -= lr * g
                    wc = self.w[c]
                    for t, v in x.items():
                        wc[t] = wc.get(t, 0.0) - lr * (g * v + self.l2 * wc.get(t, 0.0))
        return self

    def proba(self, texto):
        x = self._vetor(texto)
        return _softmax({c: self.b[c] + sum(self.w[c].get(t, 0.0) * v for t, v in x.items()) for c in self.classes})

    def palavras_mais_fortes(self, classe: str, n: int = 8) -> list[tuple[str, float]]:
        """Para explicar o modelo: as palavras com maior peso positivo para a classe."""
        return sorted(self.w[classe].items(), key=lambda kv: -kv[1])[:n]


def catalogo() -> list[Modelo]:
    return [Majoritario(), Regras(), NaiveBayes(), Logistica()]
