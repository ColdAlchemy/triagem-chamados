# Guia de estudo — triagem-chamados

## 1. Roteiro de leitura
1. `triagem/dados.py`: como os dados sintéticos são montados (e por que isso limita as conclusões).
2. `triagem/texto.py`: tokens e bigramas.
3. `triagem/modelos.py`: Naive Bayes, depois regressão logística.
4. `triagem/avaliacao.py`: métricas, validação cruzada e abstenção.
5. `tests/test_triagem.py`: veja as contas feitas à mão em `MetricasTest`.

## 2. Conceitos
**Classificação de texto.** Dada uma mensagem, escolher uma entre classes conhecidas.

**Linhas de base.** O "majoritário" (sempre a mesma classe) e as "regras" (palavras-chave). Um modelo só vale a pena se bater as duas.

**Naive Bayes.** Calcula, para cada classe, a probabilidade de aparecerem aquelas palavras, "ingenuamente" supondo que as palavras são independentes. Multiplica pela frequência da classe. A suavização de Laplace (alfa) evita probabilidade zero para palavra que nunca apareceu naquela classe.

**Regressão logística (softmax).** Cada classe tem um peso por palavra. A pontuação da classe é a soma dos pesos das palavras da mensagem; o softmax transforma as pontuações em probabilidades que somam 1. O treino ajusta os pesos para aumentar a probabilidade da classe certa.

**Gradiente descendente estocástico (SGD).** Passa pelos exemplos um a um, e a cada um empurra os pesos um pouco na direção que reduz o erro. A taxa de aprendizado controla o tamanho do passo; aqui ela diminui a cada época.

**TF-IDF.** Peso da palavra = frequência no texto × raridade no conjunto. Palavras comuns pesam pouco.

**Bigramas.** Pares de palavras seguidas ("não consigo") capturam ordem e negação que palavras soltas perdem.

**Precisão, revocação, F1.** Precisão: das mensagens que o modelo chamou de "cobrança", quantas eram. Revocação: das que eram de cobrança, quantas ele achou. F1: média harmônica das duas. **F1 macro** é a média simples entre as classes, e não deixa uma classe grande esconder uma ruim.

**Matriz de confusão.** Linhas = classe real, colunas = prevista. Mostra *quais* classes se confundem.

**Validação cruzada.** Divide em k partes, treina em k-1 e testa na restante, k vezes. Dá a média e o desvio: se a diferença entre dois modelos é menor que o desvio, não dá para afirmar que um é melhor.

**Abstenção (rejeição).** O modelo só responde quando a confiança passa de um limiar; o resto vai para uma pessoa. A curva cobertura × acerto mostra o que se ganha e o que se paga.

**Vazamento de dados.** Texto repetido em treino e teste infla a nota. Por isso os textos são únicos e há teste.

## 3. Perguntas prováveis (treine em voz alta)
- *Por que escrever do zero em vez de usar scikit-learn?* Para entender o que a biblioteca faz. Em produção eu usaria a biblioteca.
- *Os 0,90 provam que funciona?* Não: os dados são sintéticos e mais regulares que mensagens reais. Provam que o método de avaliar funciona e que o modelo aprende mais que as linhas de base.
- *Por que F1 macro e não só acurácia?* Porque classes desbalanceadas enganam a acurácia. Aqui as classes são equilibradas, mas o hábito de reportar F1 macro vale.
- *O que é a curva de abstenção?* É o gráfico (em tabela) de quanto o modelo responde sozinho versus quanto acerta nisso, conforme o limiar.
- *Por que as palavras "r" e "120" pesam na classe cobrança?* Vêm de "R$ 120" nos dados sintéticos; é um artefato, e olhar os pesos é como se descobre isso.
- *O que você faria com mensagens de dois assuntos?* Permitir várias etiquetas por mensagem (classificação multirrótulo).
- *Isso é IA generativa?* Não. É aprendizado de máquina clássico. Um modelo de linguagem poderia ser o "plano B" para os casos em que a confiança fica baixa.

## 4. Exercícios
1. Rode `python -m triagem avaliar` e mude `epocas` e `taxa` da `Logistica`. O que acontece com o F1?
2. Em `Logistica`, desligue os bigramas (`tokens(..., bigramas=False)`). O que piora?
3. Aumente a "pergunta extra" de 25% para 50% em `dados.py`. Compare a curva de abstenção.
4. Escreva 10 mensagens reais do seu dia a dia e veja o que o modelo faz com elas. Anote os erros.
5. Implemente classificação multirrótulo: uma regressão logística binária por classe.

## 5. O que dizer sobre a sua experiência (sem exagerar)
"Implementei do zero Naive Bayes e regressão logística para classificar mensagens de suporte, avaliei com F1 macro, validação cruzada e matriz de confusão, e acrescentei uma política de encaminhar para humano quando a confiança é baixa. Usei dados sintéticos e desenvolvi com o auxílio do Claude Code, estudando o código." Não diga que "treinou modelos em produção" nem que tem experiência com scikit-learn ou deep learning: não é o caso.
