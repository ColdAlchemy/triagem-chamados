"""Métricas de classificação, validação cruzada e a curva de 'encaminhar para humano'."""
import math
import random
from typing import Callable

from .modelos import Modelo


def matriz_confusao(reais: list[str], previstos: list[str], classes: list[str]) -> dict[str, dict[str, int]]:
    m = {r: {p: 0 for p in classes} for r in classes}
    for r, p in zip(reais, previstos):
        m[r][p] += 1
    return m


def relatorio(reais: list[str], previstos: list[str], classes: list[str]) -> dict:
    """Acurácia, precisão/revocação/F1 por classe e F1 macro (média simples entre as classes)."""
    acertos = sum(1 for r, p in zip(reais, previstos) if r == p)
    por_classe = {}
    for c in classes:
        tp = sum(1 for r, p in zip(reais, previstos) if r == c and p == c)
        fp = sum(1 for r, p in zip(reais, previstos) if r != c and p == c)
        fn = sum(1 for r, p in zip(reais, previstos) if r == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rev = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rev / (prec + rev) if prec + rev else 0.0
        por_classe[c] = {"precisao": prec, "revocacao": rev, "f1": f1, "suporte": tp + fn}
    return {
        "acuracia": acertos / len(reais),
        "f1_macro": sum(v["f1"] for v in por_classe.values()) / len(classes),
        "por_classe": por_classe,
    }


def avaliar(modelo: Modelo, teste: list[tuple[str, str]]) -> dict:
    reais = [y for _, y in teste]
    previstos = [modelo.prever(t) for t, _ in teste]
    r = relatorio(reais, previstos, modelo.classes)
    r["confusao"] = matriz_confusao(reais, previstos, modelo.classes)
    return r


def dobras(n: int, k: int, semente: int = 11) -> list[list[int]]:
    """Índices de k dobras disjuntas que juntas cobrem os n exemplos."""
    idx = list(range(n))
    random.Random(semente).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def validacao_cruzada(fabrica: Callable[[], Modelo], dados: list[tuple[str, str]], k: int = 5) -> dict:
    """Treina e testa k vezes; devolve média e desvio padrão do F1 macro e da acurácia."""
    f1s, acs = [], []
    for teste_idx in dobras(len(dados), k):
        fora = set(teste_idx)
        treino = [d for i, d in enumerate(dados) if i not in fora]
        teste = [dados[i] for i in teste_idx]
        m = fabrica().fit([t for t, _ in treino], [y for _, y in treino])
        r = avaliar(m, teste)
        f1s.append(r["f1_macro"])
        acs.append(r["acuracia"])
    def ms(xs):
        media = sum(xs) / len(xs)
        return media, math.sqrt(sum((x - media) ** 2 for x in xs) / len(xs))
    return {"f1_macro": ms(f1s), "acuracia": ms(acs), "dobras": k}


def curva_abstencao(modelo: Modelo, teste: list[tuple[str, str]], limiares: list[float]) -> list[dict]:
    """Para cada limiar de confiança: quanto o modelo responde sozinho (cobertura) e quanto acerta nisso."""
    previsoes = []
    for texto, real in teste:
        p = modelo.proba(texto)
        melhor = max(modelo.classes, key=lambda c: (p[c], -modelo.classes.index(c)))
        previsoes.append((p[melhor], melhor == real))
    linhas = []
    for lim in limiares:
        aceitas = [ok for conf, ok in previsoes if conf >= lim]
        linhas.append(
            {
                "limiar": lim,
                "cobertura": len(aceitas) / len(previsoes),
                "acuracia_nas_aceitas": (sum(aceitas) / len(aceitas)) if aceitas else None,
                "para_humano": len(previsoes) - len(aceitas),
            }
        )
    return linhas


def erros_mais_confiantes(modelo: Modelo, teste: list[tuple[str, str]], n: int = 5) -> list[dict]:
    """Os erros em que o modelo estava mais seguro: o que mais merece revisão."""
    erros = []
    for texto, real in teste:
        p = modelo.proba(texto)
        prev = max(modelo.classes, key=lambda c: (p[c], -modelo.classes.index(c)))
        if prev != real:
            erros.append({"texto": texto, "real": real, "previsto": prev, "confianca": p[prev]})
    return sorted(erros, key=lambda e: -e["confianca"])[:n]
