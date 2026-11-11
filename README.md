# triagem-chamados — classificar mensagens de suporte e saber quando pedir ajuda a uma pessoa

Classificador de mensagens de suporte (acesso, cobrança, entrega, defeito, sugestão) com **Naive Bayes** e **regressão logística escritos do zero** (só biblioteca padrão de Python), comparados a duas linhas de base, com validação cruzada, matriz de confusão e uma política de **"na dúvida, encaminhar para um humano"**.

> Eu defini o escopo, revisei e estudei o código; o guia `docs/ESTUDO.md` explica cada conceito. **Os dados são sintéticos** (gerados por modelos de frase), então os números abaixo servem para comparar métodos, não para prometer desempenho em dados reais.

## Para que serve (em termos simples)
Uma empresa recebe milhares de mensagens de clientes ("não consigo entrar na conta", "fui cobrado duas vezes"). Alguém precisa ler cada uma e mandar para o setor certo. Este projeto **faz essa leitura automaticamente** e, mais importante, **diz o quanto está seguro da resposta**: quando a confiança é baixa, em vez de chutar, ele manda a mensagem para uma pessoa. É assim que sistemas de IA úteis funcionam na prática: a máquina resolve o que é claro e o humano fica com o que é duvidoso.

## Resultado (750 mensagens: 525 para treinar, 225 para testar)
```
modelo          acuracia  f1_macro   validação cruzada (5 dobras, F1 macro)
majoritario        0.200     0.067   0.059 ± 0.005
regras             0.822     0.815   0.788 ± 0.005
naive-bayes        0.844     0.844   0.865 ± 0.028
logistica          0.902     0.902   0.890 ± 0.027
```
- **majoritário** (sempre a mesma classe) é a linha de base mínima: 0,20 com 5 classes equilibradas.
- **regras** são palavras-chave escritas à mão, o que muita empresa usa hoje. Os modelos que aprendem dos dados passam dele.
- A validação cruzada mostra que a diferença entre modelos é maior que a oscilação entre sorteios (±0,03).

### "Na dúvida, chama uma pessoa" (regressão logística)
```
 limiar  cobertura  acerto nas aceitas  p/ humano
   0.00      1.000               0.902          0
   0.60      0.911               0.932         20
   0.70      0.822               0.951         40
   0.80      0.760               0.971         54
   0.90      0.733               0.988         60
```
Com o limiar em 0,80, o sistema responde sozinho 76% das mensagens **acertando 97%**, e 24% vão para uma pessoa. Quem decide o limiar é o negócio, pesando o custo de errar contra o custo de uma pessoa ler a mensagem.

### Onde ele erra (`python -m triagem erros`)
Os erros mais confiantes são quase todos **mensagens com dois assuntos** ("adorei o atendimento, parabéns… aliás, o prazo de entrega passou"). O modelo escolhe uma classe só, mas a mensagem tem duas. Isso é um limite do desenho (uma etiqueta por mensagem), e não só do modelo. `python -m triagem prever "Oi, cobraram meu cartão duas vezes e o app trava"` cai em "encaminhar para um humano", que é o comportamento certo.

## Uso
```bash
python -m triagem avaliar              # tabela acima (leva alguns segundos)
python -m triagem curva                # cobertura × acerto por limiar
python -m triagem confusao             # matriz de confusão
python -m triagem erros                # erros em que o modelo estava mais seguro
python -m triagem pesos                # palavras mais fortes de cada classe
python -m triagem prever "esqueci minha senha" --limiar 0.6
python -m triagem dados                # regrava data/chamados.jsonl
python -m unittest                     # 24 testes, sem rede
```
**Nenhuma dependência.** Testado no Python 3.14 (a sintaxe usada exige 3.10 ou mais novo).

## O que tem dentro
| Módulo | Papel |
|---|---|
| `dados.py` | gerador **determinístico** de mensagens sintéticas: vocabulário compartilhado entre classes, "pergunta extra" de outra classe em 25% e erro de digitação em 10% |
| `texto.py` | normalização e tokens (palavras + bigramas) |
| `modelos.py` | `Majoritario`, `Regras`, `NaiveBayes` (Laplace) e `Logistica` (softmax + SGD sobre TF-IDF) |
| `avaliacao.py` | acurácia, precisão/revocação/F1, F1 macro, matriz de confusão, validação cruzada, curva de abstenção, erros mais confiantes |
| `cli.py` | os comandos acima |

## Ficha do modelo (resumo)
- **Uso pretendido:** demonstrar triagem de texto com abstenção. **Não** é para uso em produção.
- **Dados:** 750 mensagens sintéticas em português, 150 por classe, textos únicos, sem repetição entre treino e teste (há teste para isso).
- **Limitações conhecidas:** dados sintéticos são mais regulares que mensagens reais, então os números reais seriam menores; só uma etiqueta por mensagem; palavras fortes como "r" e "120" na classe cobrança são **artefatos** do gerador (vêm de "R$ 120"), o que mostra por que vale olhar os pesos do modelo.
- **Não foi feito:** teste com mensagens reais, calibração formal das probabilidades, comparação com um modelo de linguagem.

## Decisões de projeto
- **Dados gerados com semente fixa e arquivo versionado** (`data/chamados.jsonl`), e um teste garante que o arquivo é igual ao que o gerador produz.
- **Divisão sem vazamento:** nenhum texto aparece no treino e no teste.
- **Probabilidade importa tanto quanto o rótulo:** por isso a curva de abstenção faz parte da avaliação.
- **Tudo do zero:** o objetivo é entender o que uma biblioteca faria, não competir com uma.
