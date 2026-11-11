"""Gera um conjunto SINTÉTICO de mensagens de suporte em português, de forma determinística.

Os dados são fictícios e montados por modelos de frase. Para não ficar trivial, as frases de classes
diferentes compartilham vocabulário, algumas mensagens trazem uma "pergunta extra" de outra classe
e parte delas tem erro de digitação. A etiqueta é sempre a da frase principal.
"""
import random
from collections import defaultdict

CLASSES = ["acesso", "cobranca", "entrega", "defeito", "sugestao"]

SLOTS = {
    "app": ["app", "aplicativo", "site", "painel"],
    "produto": ["fone", "celular", "notebook", "relógio", "carregador", "tablet"],
    "valor": ["R$ 49,90", "R$ 120", "R$ 15,00", "R$ 300", "R$ 89,90"],
    "dias": ["3", "5", "7", "10", "15"],
    "cidade": ["Imperatriz", "Recife", "Curitiba", "Goiânia", "Belém"],
}

MODELOS = {
    "acesso": [
        "não consigo entrar na minha conta do {app}",
        "esqueci a senha e o e-mail de recuperação não chega",
        "meu login foi bloqueado depois de várias tentativas",
        "como troco o e-mail cadastrado na conta",
        "o código de verificação não chegou no meu celular",
        "preciso redefinir a senha do {app}",
        "minha conta foi desconectada e não consigo acessar de novo",
        "o {app} diz que meu usuário não existe",
    ],
    "cobranca": [
        "fui cobrado duas vezes {valor} no cartão",
        "a fatura veio com um valor diferente do combinado",
        "quero o estorno da cobrança de {valor}",
        "não reconheço essa cobrança no meu extrato",
        "como emito a segunda via do boleto",
        "cancelei a assinatura mas continuam cobrando {valor}",
        "o desconto prometido não apareceu na fatura",
        "paguei o boleto e o pagamento não foi baixado",
    ],
    "entrega": [
        "meu pedido de {produto} não chegou e já faz {dias} dias",
        "o rastreio está parado há {dias} dias",
        "a entrega foi feita no endereço errado",
        "o pedido consta como entregue mas não recebi",
        "quero alterar o endereço de entrega do {produto}",
        "o entregador não encontrou minha casa em {cidade}",
        "o prazo de entrega para {cidade} passou e nada de chegar",
        "chegou só uma parte do pedido",
    ],
    "defeito": [
        "o {produto} chegou com a tela rachada",
        "o {app} fecha sozinho quando abro a câmera",
        "o {produto} não liga mais depois da atualização",
        "aparece um erro ao salvar e perco tudo",
        "a bateria do {produto} descarrega em poucas horas",
        "o {app} trava toda vez que tento abrir",
        "o {produto} esquenta demais e desliga sozinho",
        "o som do {produto} falha de vez em quando",
    ],
    "sugestao": [
        "seria ótimo ter modo escuro no {app}",
        "gostaria de poder pagar com pix também",
        "podiam incluir um filtro por preço na busca",
        "adorei o atendimento, parabéns à equipe",
        "sugiro avisar por mensagem quando o pedido sair para entrega",
        "o {app} poderia exportar relatórios em planilha",
        "queria que o {app} lembrasse meu usuário no celular",
        "muito bom o {produto}, só faltou vir com capa",
    ],
}

ABERTURAS = ["Olá,", "Bom dia,", "Boa tarde,", "Oi, pessoal,", "Prezados,", "", ""]
FECHAMENTOS = ["Obrigado.", "Aguardo retorno.", "Podem me ajudar?", "Urgente!", "", "", ""]
EXTRAS = ["Aliás,", "Também", "Ah, e", "Outra coisa:"]


def _preencher(modelo: str, rng: random.Random) -> str:
    texto = modelo
    for chave, opcoes in SLOTS.items():
        marca = "{" + chave + "}"
        while marca in texto:
            texto = texto.replace(marca, rng.choice(opcoes), 1)
    return texto


def _erro_de_digitacao(texto: str, rng: random.Random) -> str:
    palavras = texto.split(" ")
    candidatas = [i for i, p in enumerate(palavras) if len(p) >= 5 and p.isalpha()]
    if not candidatas:
        return texto
    i = rng.choice(candidatas)
    p = palavras[i]
    j = rng.randrange(1, len(p) - 2)
    palavras[i] = p[:j] + p[j + 1] + p[j] + p[j + 2 :]  # troca duas letras vizinhas
    return " ".join(palavras)


def gerar(n: int = 750, semente: int = 42) -> list[tuple[str, str]]:
    """Devolve n pares (texto, classe), todos com texto único, balanceados entre as classes."""
    rng = random.Random(semente)
    por_classe = n // len(CLASSES)
    vistos: set[str] = set()
    saida: list[tuple[str, str]] = []
    for classe in CLASSES:
        feitos, tentativas = 0, 0
        while feitos < por_classe:
            tentativas += 1
            if tentativas > por_classe * 200:
                raise RuntimeError("Não foi possível gerar textos únicos suficientes; reduza n.")
            partes = [rng.choice(ABERTURAS), _preencher(rng.choice(MODELOS[classe]), rng)]
            if rng.random() < 0.25:  # pergunta extra de outra classe: o rótulo continua o da frase principal
                outra = rng.choice([c for c in CLASSES if c != classe])
                partes.append(f"{rng.choice(EXTRAS)} {_preencher(rng.choice(MODELOS[outra]), rng)}.")
            partes.append(rng.choice(FECHAMENTOS))
            texto = " ".join(p for p in partes if p).strip()
            if rng.random() < 0.10:
                texto = _erro_de_digitacao(texto, rng)
            if texto in vistos:
                continue
            vistos.add(texto)
            saida.append((texto, classe))
            feitos += 1
    rng.shuffle(saida)
    return saida


def dividir(dados: list[tuple[str, str]], proporcao_teste: float = 0.3, semente: int = 7):
    """Divisão estratificada e determinística em (treino, teste)."""
    rng = random.Random(semente)
    por_classe: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for par in dados:
        por_classe[par[1]].append(par)
    treino: list[tuple[str, str]] = []
    teste: list[tuple[str, str]] = []
    for classe in CLASSES:
        itens = por_classe[classe][:]
        rng.shuffle(itens)
        corte = int(len(itens) * proporcao_teste)
        teste.extend(itens[:corte])
        treino.extend(itens[corte:])
    rng.shuffle(treino)
    rng.shuffle(teste)
    return treino, teste
