"""Tokenização em português: minúsculas, sem acento, palavras e pares de palavras (bigramas)."""
import re
import unicodedata


def normalizar(texto: str) -> str:
    sem = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return sem


def tokens(texto: str, bigramas: bool = True) -> list[str]:
    palavras = re.findall(r"[a-z0-9]+", normalizar(texto))
    saida = list(palavras)
    if bigramas:
        saida.extend(f"{a}_{b}" for a, b in zip(palavras, palavras[1:]))
    return saida
