"""Linha de comando: dados, avaliar, confusao, curva, erros, pesos e prever."""
import argparse
import json
import sys
from pathlib import Path

from .avaliacao import avaliar, curva_abstencao, erros_mais_confiantes, validacao_cruzada
from .dados import CLASSES, dividir, gerar
from .modelos import Logistica, Modelo, catalogo

RAIZ = Path(__file__).resolve().parent.parent
LIMIARES = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _preparar(n: int):
    dados = gerar(n)
    treino, teste = dividir(dados)
    return dados, treino, teste


def _treinados(treino) -> list[Modelo]:
    return [m.fit([t for t, _ in treino], [y for _, y in treino]) for m in catalogo()]


def _escolher(modelos: list[Modelo], nome: str) -> Modelo:
    for m in modelos:
        if m.nome == nome:
            return m
    raise SystemExit("Modelo desconhecido: " + nome + ". Opções: " + ", ".join(m.nome for m in modelos))


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(encoding="utf-8")
    pai = argparse.ArgumentParser(add_help=False)
    pai.add_argument("--n", type=int, default=750, help="tamanho do conjunto sintético (padrão 750)")
    ap = argparse.ArgumentParser(prog="triagem", description="Triagem de chamados com modelos escritos do zero.")
    sub = ap.add_subparsers(dest="comando", required=True)
    d = sub.add_parser("dados", parents=[pai], help="gera e grava o conjunto sintético em JSONL")
    d.add_argument("--saida", default=str(RAIZ / "data" / "chamados.jsonl"))
    sub.add_parser("avaliar", parents=[pai], help="compara os modelos no teste e na validação cruzada")
    for nome in ("confusao", "curva", "erros"):
        s = sub.add_parser(nome, parents=[pai])
        s.add_argument("--modelo", default="logistica")
    sub.add_parser("pesos", parents=[pai], help="palavras mais fortes de cada classe na regressão logística")
    p = sub.add_parser("prever", parents=[pai], help="classifica uma mensagem; abaixo do limiar, encaminha para humano")
    p.add_argument("mensagem")
    p.add_argument("--limiar", type=float, default=0.6)
    args = ap.parse_args(argv)

    dados, treino, teste = _preparar(args.n)

    if args.comando == "dados":
        destino = Path(args.saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8", newline="\n") as f:
            for texto, classe in dados:
                f.write(json.dumps({"texto": texto, "classe": classe}, ensure_ascii=False) + "\n")
        print(f"{len(dados)} exemplos gravados em {destino} (treino {len(treino)}, teste {len(teste)}).")
        return 0

    modelos = _treinados(treino)
    if args.comando == "avaliar":
        print(f"treino {len(treino)} · teste {len(teste)} · classes: {', '.join(CLASSES)}\n")
        print(f"{'modelo':14} {'acuracia':>9} {'f1_macro':>9}   validação cruzada (5 dobras, F1 macro)")
        for m in modelos:
            r = avaliar(m, teste)
            cv = validacao_cruzada(type(m), dados, 5)
            media, dp = cv["f1_macro"]
            print(f"{m.nome:14} {r['acuracia']:9.3f} {r['f1_macro']:9.3f}   {media:.3f} ± {dp:.3f}")
    elif args.comando == "confusao":
        m = _escolher(modelos, args.modelo)
        c = avaliar(m, teste)["confusao"]
        print("linhas = classe real, colunas = classe prevista")
        print(" " * 10 + "".join(x[:9].rjust(10) for x in m.classes))
        for real in m.classes:
            print(real[:9].ljust(10) + "".join(str(c[real][p]).rjust(10) for p in m.classes))
    elif args.comando == "curva":
        m = _escolher(modelos, args.modelo)
        print(f"{'limiar':>7} {'cobertura':>10} {'acerto nas aceitas':>19} {'p/ humano':>10}")
        for linha in curva_abstencao(m, teste, LIMIARES):
            ac = "-" if linha["acuracia_nas_aceitas"] is None else f"{linha['acuracia_nas_aceitas']:.3f}"
            print(f"{linha['limiar']:7.2f} {linha['cobertura']:10.3f} {ac:>19} {linha['para_humano']:10d}")
    elif args.comando == "erros":
        m = _escolher(modelos, args.modelo)
        for e in erros_mais_confiantes(m, teste, 8):
            print(f"[{e['confianca']:.2f}] real={e['real']} previsto={e['previsto']}\n    {e['texto']}")
    elif args.comando == "pesos":
        m = _escolher(modelos, "logistica")
        assert isinstance(m, Logistica)
        for c in m.classes:
            print(c.ljust(9), ", ".join(f"{t} ({w:.1f})" for t, w in m.palavras_mais_fortes(c)))
    elif args.comando == "prever":
        m = _escolher(modelos, "logistica")
        proba = m.proba(args.mensagem)
        melhor = max(m.classes, key=lambda c: proba[c])
        for c in sorted(proba, key=lambda c: -proba[c]):
            print(f"{c:9} {proba[c]:.3f}")
        if proba[melhor] >= args.limiar:
            print(f"Decisão: classificar como '{melhor}' (confiança {proba[melhor]:.2f} ≥ {args.limiar}).")
        else:
            print(f"Decisão: encaminhar para um humano (confiança {proba[melhor]:.2f} < {args.limiar}).")
    return 0
