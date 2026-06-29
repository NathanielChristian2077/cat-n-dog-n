# RELATÓRIO FINAL
## Inteligência Artificial

# Cat-n-Dog-n
### *Cat and Dog Acknowledgment Neural Network*

Instituto Federal de Educação, Ciência e Tecnologia de Santa Catarina — IFSC  
Bacharelado em Ciência da Computação

**Equipe**  
Carlos Eduardo Garcez Mattos  
Nathaniel Christian Silva Alves  
Vinícius Souza Corbellini

*Versão consolidada em 29 de junho de 2026*

---

# 1. Introdução

Este trabalho teve como objetivo aplicar conceitos de Redes Neurais Artificiais (RNAs) no problema de classificação binária de imagens de gatos e cães. A proposta foi organizada em três níveis de abstração: uma Rede Perceptron Multicamadas (MLP) implementada manualmente; uma Rede Neural Convolucional (CNN) construída camada a camada com PyTorch; e modelos convolucionais pré-treinados adaptados por *transfer learning*.

A primeira abordagem foi desenvolvida para expor os elementos matemáticos fundamentais de uma rede neural: propagação direta, funções de ativação, função de perda, retropropagação e atualização dos pesos. A segunda abordagem introduziu uma hipótese estrutural adequada a imagens, usando convoluções para preservar relações espaciais entre pixels. A terceira reutilizou representações visuais aprendidas previamente em ImageNet, permitindo avaliar como pesos pré-treinados afetam o desempenho quando há poucos exemplos rotulados disponíveis.

A comparação considerou acurácia, *loss*, métricas por classe, matrizes de confusão, custo computacional e estabilidade entre execuções. O ponto central não foi apenas identificar qual modelo obteve a maior acurácia, porque humanos já sofrem bastante com rankings simplistas, mas compreender o que cada grau de abstração entrega e esconde.

## 1.1 Especificações de hardware e software

| Item | Especificação |
|---|---|
| Processador | AMD Ryzen 7 5700X3D |
| Memória RAM | 48 GB DDR4 |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| Sistema operacional | Fedora 44 |
| Linguagens | Go e Python |
| Bibliotecas e ferramentas | PyTorch, TorchVision, NumPy, Pandas, Matplotlib, Scikit-learn, Pillow, JupyterLab, IPyKernel e Pytest |

---

# 2. Metodologia

## 2.1 Protocolo comum de dados e avaliação

Todas as abordagens utilizaram a divisão de dados fornecida pelo professor, sem reorganização aleatória dos exemplos. O conjunto contém 300 imagens para treinamento, 100 para validação e 100 para teste. Os conjuntos de validação e teste possuem 50 imagens de cada classe, tornando a avaliação balanceada.

A classe positiva foi definida explicitamente como **cães**. A seleção de arquiteturas e hiperparâmetros foi feita somente com base na validação. O teste final foi utilizado apenas depois de fixadas as escolhas experimentais, evitando que o conjunto de teste fosse usado como instrumento de ajuste.

Antes da interpretação definitiva dos resultados, foi prevista uma auditoria de duplicatas entre os *splits*. Duplicatas exatas entre treino, validação e teste caracterizam vazamento de dados e invalidam conclusões de generalização. A saída de `runs/dataset_audit/summary.json` deve ser anexada ou resumida antes da entrega final.

<!-- PENDÊNCIA: inserir aqui o resultado objetivo da auditoria de duplicatas entre splits. -->

---

## 2.2 Abordagem 1 — MLP manual em Go

### Descrição

A primeira abordagem consiste em uma MLP implementada manualmente em Go, sem utilização de bibliotecas específicas de construção ou treinamento de redes neurais. As imagens foram redimensionadas para `64 × 64`, convertidas para tons de cinza e vetorizadas, produzindo uma entrada de `4096` atributos por imagem.

A rede emprega camadas ocultas com função de ativação ReLU e uma saída binária com ativação sigmoidal, conforme os requisitos da atividade. O treinamento é baseado em *forward propagation*, cálculo de perda, *backpropagation* explícito e atualização dos pesos por gradiente descendente. A implementação manual permitiu analisar diretamente a relação entre vetor de entrada, matrizes de pesos, ativações e gradientes.

Foram exploradas topologias densas em que a configuração `32 → 64 → 512` apresentou os melhores resultados relativos nas comparações internas da Etapa 1. Para a entrega principal, deve ser mantida a versão que respeita a saída sigmoidal exigida no enunciado.

### Implementação

A implementação foi estruturada em torno das operações de multiplicação matriz-vetor, aplicação das funções de ativação, cálculo da perda e propagação dos gradientes. Em termos matemáticos, uma camada oculta segue a forma:

```text
z¹ = W¹x + b¹
a¹ = ReLU(z¹)
```

A saída binária é dada por:

```text
z² = W²a¹ + b²
ŷ = sigmoid(z²)
```

Para classificação binária, a função de perda recomendada é a *Binary Cross-Entropy*:

```text
L = −[y log(ŷ) + (1 − y) log(1 − ŷ)]
```

Com a combinação de saída sigmoidal e BCE, o erro na camada de saída simplifica para:

```text
δ² = ŷ − y
```

<!-- PENDÊNCIA: inserir um trecho curto do código Go real da MLP, preferencialmente a função de forward ou backward. Não inventar um trecho diferente do que foi entregue, porque o professor pode perguntar exatamente sobre ele. -->

### Procedimento de testes

- Pré-processamento: imagens em escala de cinza, `64 × 64` pixels e vetor de 4096 posições;
- Divisão de dados: 300 imagens de treino, 100 de validação e 100 de teste;
- Funções de ativação: ReLU nas camadas ocultas e sigmoide na saída;
- Função de perda: Binary Cross-Entropy;
- Critérios de avaliação: *loss*, acurácia, matriz de confusão e tempo de treinamento;
- Validação de corretude: testes unitários das funções de ativação, dimensões matriciais e verificação numérica dos gradientes.

<!-- PENDÊNCIA CRÍTICA: preencher arquitetura final, batch size, learning rate, número de épocas, seed, loss final, acurácia final, duração do treino e matriz de confusão da MLP. -->

---

## 2.3 Abordagem 2 — CNN construída com PyTorch

### Descrição

A segunda abordagem utilizou uma CNN construída manualmente com camadas do PyTorch, sem importar arquiteturas prontas ou pesos pré-treinados. Ao contrário da MLP, a CNN preserva a estrutura espacial das imagens e aprende filtros locais para bordas, texturas e composições visuais.

A entrada foi mantida em RGB, com dimensão `3 × 224 × 224`. A arquitetura contém seis convoluções organizadas em três blocos, três camadas de *max pooling*, normalização em lote, ReLU e um classificador denso configurável. O extrator convolucional foi mantido fixo nas comparações, permitindo que diferenças entre execuções fossem atribuídas ao *head* denso.

### Implementação

A arquitetura de extração visual segue o fluxo abaixo:

```text
RGB 3×224×224
  → Conv 3→32 → Conv 32→32 → MaxPool
  → Conv 32→64 → Conv 64→64 → MaxPool
  → Conv 64→128 → Conv 128→128 → MaxPool
  → AdaptiveAvgPool 1×1
  → classificador denso
```

Trecho representativo da implementação da CNN:

```python
self.features = nn.Sequential(
    ConvNormAct(3, 32),
    ConvNormAct(32, 32),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvNormAct(32, 64),
    ConvNormAct(64, 64),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvNormAct(64, 128),
    ConvNormAct(128, 128),
    nn.MaxPool2d(kernel_size=2, stride=2),
)
```

Cada bloco `ConvNormAct` combina convolução `3 × 3`, Batch Normalization e ReLU. A inicialização usa o método de Kaiming, adequado à presença de ReLU. O treinamento utiliza AdamW, *weight decay*, redução adaptativa da taxa de aprendizagem, *early stopping*, *mixed precision* quando disponível e limitação da norma dos gradientes.

### Procedimento de testes

- Pré-processamento: RGB, redimensionamento/corte para `224 × 224` e normalização por canal;
- *Data augmentation*: `RandomResizedCrop`, reflexão horizontal, `ColorJitter` e `RandomErasing`;
- Arquitetura final de referência: `cnn_baseline_128_sigmoid1`;
- Épocas máximas: 40;
- *Batch size*: 32;
- Taxa de aprendizagem inicial: `1e−3`;
- Otimizador: AdamW com `weight_decay = 1e−4`;
- Seleção: menor *loss* de validação;
- Avaliação final: checkpoint com melhor validação aplicado ao conjunto de teste.

---

## 2.4 Abordagem 3 — Transfer Learning com TorchVision

### Descrição

A terceira abordagem utilizou modelos convolucionais pré-treinados em ImageNet e disponibilizados pelo TorchVision. Foram avaliadas três arquiteturas: ResNet18, EfficientNet-B0 e ConvNeXt-Tiny. Em todos os casos, a camada classificadora original foi substituída por uma nova camada de duas saídas, adaptada para gatos e cães.

O treinamento foi dividido em duas fases. Na primeira, o *backbone* foi congelado e somente a nova cabeça classificadora foi treinada. Na segunda, o último estágio visual de cada arquitetura foi liberado para *fine-tuning* parcial, mantendo os estágios iniciais congelados. Essa escolha reduz o risco de reescrever representações úteis de ImageNet com apenas 300 imagens de treino.

### Implementação

A adaptação do classificador final foi realizada conforme a arquitetura:

```python
if identifier == "resnet18":
    model.fc = nn.Linear(model.fc.in_features, 2)
else:
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
```

A saída da rede contém dois *logits* crus e é otimizada com `CrossEntropyLoss`. A operação Softmax é aplicada apenas na conversão para probabilidades durante o cálculo das métricas, preservando estabilidade numérica.

### Procedimento de testes

- Pré-processamento: imagens RGB em `224 × 224`, com transformações compatíveis com os pesos ImageNet;
- *Data augmentation*: recorte/escala, reflexão horizontal, perturbações fotométricas e oclusão leve;
- Modelos: ResNet18, EfficientNet-B0 e ConvNeXt-Tiny;
- Seeds: 42, 73 e 101;
- Fase 1: cabeça classificadora, 12 épocas máximas, `learning rate = 1e−3`;
- Fase 2: *fine-tuning* parcial, 20 épocas máximas, `learning rate = 1e−4`;
- *Batch size*: 32;
- Otimizador: AdamW com `weight_decay = 1e−4`;
- Seleção: menor *loss* de validação, sem consultar o teste durante a escolha;
- Teste final: realizado somente após o congelamento das escolhas experimentais.

---

## 2.5 Divisão das atividades

> **PENDÊNCIA DE EQUIPE:** a tabela abaixo já contém os integrantes e a divisão técnica provável, mas os percentuais precisam ser confirmados pelos três membros antes da entrega. Não vale transformar memória coletiva em método científico de última hora.

| Integrante | Atividade desenvolvida | Contribuição estimada |
|---|---|---:|
| Nathaniel Christian Silva Alves | Implementação e validação da MLP manual em Go; apoio à análise experimental. | **Confirmar** |
| Carlos Eduardo Garcez Mattos | Implementação, testes ou documentação das Etapas 2 e 3. | **Confirmar** |
| Vinícius Souza Corbellini | Implementação, testes ou documentação das Etapas 2 e 3. | **Confirmar** |
| Equipe | Integração dos resultados, revisão do relatório e preparação da apresentação. | **Confirmar** |

---

# 3. Resultados

## 3.1 Métricas obtidas

| Abordagem / modelo | Loss de teste | Acurácia de teste | Precision | Recall | F1-score | Tempo de treinamento |
|---|---:|---:|---:|---:|---:|---:|
| MLP manual em Go | **Pendente** | **Pendente** | **Pendente** | **Pendente** | **Pendente** | **Pendente** |
| CNN do zero | 0,5817 | 66,0% | 63,8% | 74,0% | 68,5% | **Pendente de `run_summary.json`** |
| ResNet18 | 0,0308 ± 0,0041 | 98,33% ± 0,58 pp | 97,40% ± 1,08 pp | 99,33% ± 1,15 pp | 98,35% ± 0,57 pp | **Pendente** |
| EfficientNet-B0 | 0,0346 ± 0,0043 | 98,33% ± 0,58 pp | 96,78% ± 1,09 pp | 100,00% ± 0,00 pp | 98,36% ± 0,56 pp | **Pendente** |
| ConvNeXt-Tiny | 0,0575 ± 0,0193 | 98,00% ± 0,00 pp | 96,15% ± 0,00 pp | 100,00% ± 0,00 pp | 98,04% ± 0,00 pp | **Pendente** |

A CNN construída do zero obteve desempenho consideravelmente inferior aos modelos pré-treinados. Isso é esperado: a CNN precisou aprender filtros visuais a partir de apenas 300 imagens, enquanto os modelos de *transfer learning* partiram de representações já ajustadas em ImageNet.

Na validação, ConvNeXt-Tiny foi o modelo vencedor, com melhor *loss* média de `0,00188 ± 0,00056` e 100% de acurácia observada. No teste final, entretanto, ResNet18 apresentou a menor *loss* média, e EfficientNet-B0 apresentou o melhor equilíbrio global entre F1, *recall* e custo de memória.

### Recursos computacionais observados no fine-tuning

| Modelo | Parâmetros treináveis no fine-tuning | Pico de VRAM |
|---|---:|---:|
| ResNet18 | 8,39 milhões | 330 MiB |
| EfficientNet-B0 | 414,7 mil | 231 MiB |
| ConvNeXt-Tiny | 14,29 milhões | 634 MiB |

---

## 3.2 Matrizes de confusão

### Abordagem 1 — MLP manual em Go

<!-- PENDÊNCIA CRÍTICA: inserir a matriz de confusão real da MLP. -->

| Classe real \ Classe predita | Gato | Cão |
|---|---:|---:|
| Gato | **Pendente** | **Pendente** |
| Cão | **Pendente** | **Pendente** |

### Abordagem 2 — CNN do zero

Como o conjunto de teste contém 50 gatos e 50 cães, as métricas finais da CNN permitem reconstruir a matriz abaixo:

| Classe real \ Classe predita | Gato | Cão |
|---|---:|---:|
| Gato | 29 | 21 |
| Cão | 13 | 37 |

A CNN classificou corretamente 37 dos 50 cães, mas confundiu 21 gatos como cães. Esse padrão explica o *recall* de 74,0% para a classe positiva e a precisão menor, de 63,8%.

### Abordagem 3 — Transfer Learning

Para manter uma matriz representativa por abordagem, foi utilizada a ConvNeXt-Tiny, vencedora da seleção por validação. Em cada uma das três seeds, o modelo classificou corretamente todos os cães e confundiu dois gatos como cães:

| Classe real \ Classe predita | Gato | Cão |
|---|---:|---:|
| Gato | 48 | 2 |
| Cão | 0 | 50 |

As matrizes individuais de ResNet18 e EfficientNet-B0 devem ser mantidas como figuras complementares, pois exibem pequenas variações entre as três seeds.

---

## 3.3 Gráficos comparativos

Devem ser incluídos no documento final os artefatos gerados automaticamente pelos experimentos:

1. Curvas de *loss* e acurácia da CNN do zero;
2. Matriz de confusão da CNN do zero;
3. Curvas de treinamento das execuções de *transfer learning*;
4. Matrizes de confusão dos modelos selecionados;
5. Gráfico de barras comparando acurácia, F1-score, *loss* e duração de treinamento;
6. Gráfico de barras comparando consumo máximo de VRAM dos três modelos pré-treinados.

<!-- PENDÊNCIA: inserir as imagens reais de `runs/**/plots/` antes da exportação final. -->

---

# 4. Discussão dos resultados

## 4.1 Facilidade de implementação

A MLP manual foi a abordagem mais exigente conceitualmente, pois todas as operações fundamentais precisaram ser programadas e verificadas explicitamente. Isso inclui representar matrizes, implementar as funções de ativação, calcular gradientes e garantir que as dimensões de cada operação fossem compatíveis. O custo dessa abordagem foi maior, mas ela tornou visível a matemática normalmente escondida pelos frameworks.

A CNN construída com PyTorch reduziu a complexidade de implementação das operações internas, mas ainda exigiu decisões de arquitetura, definição das transformações, configuração do processo de treino e controle de *overfitting*. A presença de convoluções, normalização, *pooling* e *dropout* ofereceu maior capacidade de modelagem, ao custo de mais hiperparâmetros e maior demanda computacional.

O *transfer learning* foi a abordagem mais direta para obter alto desempenho. O principal trabalho foi adaptar corretamente as camadas finais, preservar as transformações compatíveis com ImageNet e definir uma estratégia de congelamento e *fine-tuning*. Portanto, ela foi mais simples no nível da arquitetura, embora ainda exija rigor metodológico para não escolher modelos olhando repetidamente para o teste, um hábito acadêmico que deveria vir com alarme sonoro.

## 4.2 Custo computacional e dificuldades de treinamento

A MLP trabalha com vetores de 4096 posições e não preserva a estrutura espacial da imagem. Seu custo por operação é relativamente simples, mas ela depende de matrizes densas grandes e tende a desperdiçar informação estrutural importante para visão computacional.

A CNN do zero exige mais processamento por utilizar imagens RGB de `224 × 224` e múltiplas convoluções. Entretanto, ela apresentou custo de memória controlado e treinamento estável com AdamW, Batch Normalization, *mixed precision*, limitação de gradientes e redução adaptativa da taxa de aprendizagem. O principal desafio não foi instabilidade numérica, mas a limitação estatística de aprender filtros robustos a partir de somente 300 imagens de treino.

Nos modelos pré-treinados, o custo variou de acordo com a arquitetura. EfficientNet-B0 apresentou o menor pico de VRAM, 231 MiB, enquanto ConvNeXt-Tiny exigiu 634 MiB e teve a maior quantidade de parâmetros treináveis na fase de *fine-tuning*. A ResNet18 ocupou uma posição intermediária. Os tempos exatos devem ser adicionados a partir dos resumos de execução, pois ainda não foram consolidados no repositório.

## 4.3 Desempenho

A CNN treinada do zero obteve 66,0% de acurácia e F1-score de 68,5%. Esse resultado funciona como um *baseline* honesto para o cenário de poucos dados: a rede precisou aprender representações visuais sem conhecimento prévio e apresentou dificuldade de generalização.

Os modelos pré-treinados obtiveram aproximadamente 98% de acurácia no teste. ConvNeXt-Tiny foi claramente superior no critério de seleção por validação, mas sua *loss* média no teste foi maior e mais variável. ResNet18 apresentou a menor *loss* média de teste, sugerindo probabilidades melhor calibradas nesse conjunto. EfficientNet-B0 alcançou o melhor compromisso prático: F1 médio ligeiramente maior, *recall* perfeito em todas as seeds e menor uso de VRAM.

Essas diferenças devem ser interpretadas com cautela. O conjunto de teste tem apenas 100 imagens, portanto uma classificação alterada equivale a um ponto percentual de acurácia. Além disso, as três seeds medem estabilidade de otimização no mesmo conjunto de teste; elas não substituem uma validação cruzada com novos exemplos.

## 4.4 Aplicabilidade prática

A MLP manual é adequada principalmente para fins didáticos, dados tabulares ou cenários em que a entrada já possui atributos compactos e semanticamente relevantes. Para imagens brutas, a vetorização elimina relações espaciais e limita a capacidade do modelo de reconhecer padrões locais.

A CNN construída do zero é mais indicada quando se dispõe de dados suficientes e quando há interesse em controlar completamente a arquitetura aprendida. Ela é útil para domínios cujas imagens diferem muito de ImageNet, mas demanda mais exemplos, tempo de treinamento e experimentação.

O *transfer learning* é a opção mais apropriada para conjuntos pequenos ou médios em tarefas visualmente próximas dos dados usados no pré-treinamento. Entre os modelos avaliados, EfficientNet-B0 é especialmente atraente para cenários com restrições de memória, enquanto ResNet18 oferece uma arquitetura mais simples e bem estabelecida. ConvNeXt-Tiny pode ser preferível quando a validação e a capacidade computacional justificarem seu maior custo.

## 4.5 Análise dos erros

A CNN do zero apresentou um viés em direção à classe positiva, cães. Dos 50 gatos do teste, 21 foram classificados como cães, enquanto 13 cães foram classificados como gatos. Esse comportamento pode estar relacionado ao número reduzido de imagens, à diversidade limitada do treino e à dificuldade de aprender características visuais robustas desde a inicialização aleatória.

Nos modelos pré-treinados, os principais erros também ocorreram na direção gato → cão. Nas execuções de 98% de acurácia com *recall* de 100%, dois gatos foram classificados como cães. Esse padrão sugere que os modelos foram muito sensíveis a evidências visuais associadas a cães, possivelmente por semelhança de textura, enquadramento, pelagem ou contexto de fundo.

A análise final deve incluir exemplos visuais desses falsos positivos e falsos negativos a partir de `test_predictions.csv`. Também é indispensável registrar o resultado da auditoria de duplicatas entre *splits*, pois desempenho elevado sem verificação de integridade de dados seria apenas um número muito bonito tentando fugir da responsabilidade.

---

# 5. Conclusão

O desenvolvimento deste trabalho permitiu comparar três formas de construir soluções neurais para classificação de imagens. A MLP manual apresentou os fundamentos matemáticos e computacionais do aprendizado supervisionado; a CNN construída do zero introduziu o viés espacial necessário para visão computacional; e o *transfer learning* demonstrou a vantagem prática de reutilizar representações já aprendidas.

A CNN treinada do zero alcançou 66,0% de acurácia, evidenciando a dificuldade de aprender representações visuais robustas a partir de somente 300 imagens. Em contraste, os modelos pré-treinados obtiveram aproximadamente 98% de acurácia. ConvNeXt-Tiny apresentou a melhor seleção por validação, ResNet18 atingiu a menor *loss* média no teste e EfficientNet-B0 ofereceu o melhor equilíbrio entre desempenho, estabilidade e custo computacional.

Como trabalhos futuros, podem ser investigados o aumento controlado da base de treinamento, validação cruzada, técnicas de regularização adicionais, ajuste de hiperparâmetros, avaliação em uma base externa e comparação do custo energético entre arquiteturas. Para a MLP manual, também é possível aprofundar a análise de esparsidade das ativações e do impacto de diferentes topologias densas, desde que isso seja tratado como experimento controlado e não como magia negra com gráficos.

---

# Referências

[1] INSTITUTO FEDERAL DE EDUCAÇÃO, CIÊNCIA E TECNOLOGIA DE SANTA CATARINA. *Trabalho de Implementação 2 — Redes Neurais Artificiais*. Lages: IFSC, 2026.

[2] PASZKE, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In: *Advances in Neural Information Processing Systems*, 2019.

[3] DENG, J. et al. ImageNet: A Large-Scale Hierarchical Image Database. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2009.

[4] HE, K. et al. Deep Residual Learning for Image Recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016.

[5] TAN, M.; LE, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In: *Proceedings of the 36th International Conference on Machine Learning*, 2019.

[6] LIU, Z. et al. A ConvNet for the 2020s. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022.

[7] GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. Cambridge: MIT Press, 2016.

---

# Apêndice A — Resultados complementares do GoMLP / proto-DSAP

Este apêndice registra a extensão experimental desenvolvida no repositório **GoMLP**, branch `proto-DSAP`. Ela não substitui a MLP densa exigida como núcleo da Etapa 1; trata-se de uma investigação adicional sobre como explorar a esparsidade naturalmente induzida pela ReLU. A ideia foi comparar a propagação densa convencional com uma forma de **Dynamic Sparse Activation Propagation (DSAP)** que evita multiplicações associadas a ativações nulas.

## A.1 Representação esparsa dinâmica

Após uma ReLU, toda ativação menor ou igual a zero é convertida em zero e não contribui para a camada seguinte. A variante DSAP representa apenas os neurônios ativos por meio de um vetor compacto, denominado `ActiveVector`, que preserva simultaneamente os índices originais e os valores positivos:

```text
idx = [j₁, j₂, ..., jₖ]
val = [aⱼ₁, aⱼ₂, ..., aⱼₖ]
```

A próxima camada é calculada como:

```text
zₒ = bₒ + Σⱼ ativo (aⱼ · Wⱼ,ₒ)
```

Assim, a arquitetura permanece fixa, mas cada imagem ativa uma sub-rede diferente. A versão `threshold = 0` remove somente ativações exatamente nulas e, portanto, deve reproduzir a mesma função da rede densa. Valores positivos de *threshold* removem ativações pequenas adicionais e passam a constituir uma aproximação experimental.

## A.2 Protocolo do protótipo

Os experimentos utilizaram o mesmo problema da Etapa 1: imagens em escala de cinza, redimensionadas para `64 × 64` e vetorizadas em 4096 entradas, com 300 imagens de treino, 100 de validação e 100 de teste. As MLPs utilizaram ReLU nas camadas ocultas, saída sigmoidal e Binary Cross-Entropy.

O benchmark de *forward propagation* foi executado em CPU, com `GOMAXPROCS = 1`, 500 repetições e 50 iterações de aquecimento. Essa decisão isola o custo do cálculo denso e esparso, sem atribuir a goroutines ou à GPU um ganho que ainda não foi medido. Os valores abaixo representam tempo médio por *forward* e não equivalem ao tempo total de treinamento, porque *backpropagation*, leitura dos dados e atualização dos pesos não fazem parte desse microbenchmark.

## A.3 Verificação de equivalência da DSAP exact

A versão `threshold = 0`, denominada **DSAP exact**, foi validada nas arquiteturas `4096 → 64 → 1`, `4096 → 256 → 64 → 1`, `4096 → 128 → 256 → 128 → 1`, `4096 → 256 → 256 → 128 → 1` e `4096 → 512 → 512 → 128 → 1`.

Em todos os casos avaliados, a comparação entre o *forward* denso e o *forward* DSAP exact resultou em:

```text
mismatch_count_from_dense = 0
max_abs_diff_from_dense = 0
```

Portanto, para ativações exatamente nulas produzidas pela ReLU, a representação compacta preservou a saída e as decisões do modelo denso. O experimento demonstra equivalência computacional, não uma melhoria automática de acurácia: a versão exact calcula a mesma função com uma rota de execução diferente.

## A.4 Resultados de desempenho do forward

| Camadas ocultas | Tempo denso (ns/forward) | Tempo DSAP exact (ns/forward) | Redução de tempo | Esparsidade média observada |
|---|---:|---:|---:|---:|
| `64` | 251,182 | 232,240 | 7,54% | 53,36% |
| `256 → 64` | 950,254 | 868,090 | 8,65% | 44,86% |
| `128 → 256 → 128` | 526,198 | 471,552 | 10,38% | 47,78% |
| `256 → 256 → 128` | 1.000.383 | 901.491 | 9,89% | — |
| `512 → 512 → 128` | 2.095.225 | 1.838.661 | 12,25% | — |

A redução de tempo cresceu de aproximadamente 7,5% para 12,2% conforme as redes ficaram mais profundas e largas. Isso é coerente com a hipótese da DSAP: quando uma camada ReLU produz muitos zeros, a camada seguinte pode deixar de executar produtos associados a essas posições. A arquitetura `128 → 256 → 128` apresentou o melhor equilíbrio observado entre economia de tempo e esparsidade, com redução de 10,38% e 47,78% de ativações inativas.

Mesmo assim, os dados não autorizam a frase preguiçosa “esparso é sempre mais rápido”. O ganho depende da taxa de esparsidade, do tamanho das camadas e do custo de compactar/acessar índices. Em hardware e bibliotecas otimizadas para operações densas, esse overhead pode reduzir ou eliminar o benefício observado.

## A.5 Qualidade preditiva e trade-offs de threshold

Na topologia `4096 → 256 → 64 → 1`, a DSAP exact preservou os resultados da versão densa: acurácia de `0,5900`, F1-score de `0,6435`, zero divergências de decisão e esparsidade média de `44,86%`. Em uma topologia mais profunda, `4096 → 128 → 256 → 128 → 1`, foram observados acurácia de `0,6000` e F1-score de `0,6550`, juntamente da redução de tempo de 10,38% apresentada na seção anterior.

Os testes de *threshold* positivo foram tratados como aproximações, e não como equivalência matemática. No modelo `4096 → 256 → 64 → 1`, os limiares entre `1e−6` e `0,001` não alteraram as métricas observadas. O limiar `0,05` manteve zero divergências de decisão naquele experimento e elevou a esparsidade para aproximadamente 47,44%. Já o limiar `0,10` produziu quatro divergências em relação à rede densa, *loss* de `0,735539`, acurácia de `0,6100`, F1-score de `0,6549` e esparsidade de `50,02%`. Em `threshold = 0,25`, a esparsidade subiu para `57,76%`, mas a acurácia e o F1 recuaram para `0,5800` e `0,6250`, respectivamente.

| Modo | Threshold | Acurácia | F1-score | Divergências da rede densa | Esparsidade |
|---|---:|---:|---:|---:|---:|
| DSAP exact | 0 | 0,5900 | 0,6435 | 0 | 44,86% |
| DSAP aproximada | 0,05 | sem alteração observada | sem alteração observada | 0 | 47,44% |
| DSAP aproximada | 0,10 | 0,6100 | 0,6549 | 4 | 50,02% |
| DSAP aproximada | 0,25 | 0,5800 | 0,6250 | — | 57,76% |

O aparente ganho de acurácia em `threshold = 0,10` não deve ser interpretado como superioridade da técnica. Como a aproximação muda as ativações e causa decisões diferentes, ela altera a função do classificador. Com apenas 100 exemplos de teste, quatro decisões já são suficientes para produzir uma variação visualmente atraente e estatisticamente bem pouco épica.

## A.6 Leitura crítica dos resultados

A baseline densa original `4096 → 128 → 1`, avaliada em seis seeds, apresentou aproximadamente 57,7% de acurácia média na validação, 49,7% de acurácia média no teste e 43,7% de F1 médio. A melhor acurácia de teste foi 54% e a pior 44%, sinalizando generalização próxima do acaso. Essa limitação é esperada para uma MLP que recebe pixels vetorizados: ela não preserva relações espaciais entre regiões vizinhas da imagem.

A DSAP não foi criada para corrigir essa limitação representacional. Sua contribuição é computacional e metodológica: demonstrar que ativações nulas de ReLU podem ser removidas de forma exata, preservando as saídas da MLP, e medir em quais arquiteturas essa redução de operações supera o overhead da estrutura esparsa.

Os resultados sugerem três conclusões principais:

1. A ReLU produziu esparsidade relevante, entre aproximadamente 44,9% e 53,4% nas arquiteturas medidas;
2. A DSAP exact preservou integralmente as decisões e os valores do *forward* denso nas arquiteturas testadas;
3. O ganho de desempenho foi real no microbenchmark de *forward*, variando de 7,54% a 12,25%, mas não deve ser confundido com aceleração equivalente do treinamento completo.

Como continuidade, uma avaliação mais rigorosa pode medir o custo do *backpropagation* esparso, a atualização de parâmetros, a execução com mini-batches maiores e o comportamento em processadores ou kernels especializados. Para o escopo deste trabalho, o protótipo já cumpre o objetivo de mostrar uma extensão experimental controlada da MLP manual, sem tentar vender um `if` em torno de zeros como se a humanidade tivesse acabado de descobrir uma nova física.

**Fonte dos dados:** experimentos locais do repositório GoMLP, branch `proto-DSAP`, com comparação `dense` versus `sparse_exact` e benchmarks de *forward propagation*.