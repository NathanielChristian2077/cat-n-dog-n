# Etapa 3: Transfer Learning

A Etapa 3 reaproveita o contrato experimental consolidado na Etapa 2: a mesma divisão `train/val/test`, métricas, gráficos, checkpoints e estrutura de artefatos. O que muda é a origem da representação visual: em vez de aprender filtros do zero, o projeto usa pesos ImageNet oficiais do TorchVision e adapta o classificador ao problema binário gatos versus cães.

## Modelos selecionados

Serão utilizados três backbones pré-treinados do TorchVision:

1. **ResNet18**: baseline residual clássico, de leitura simples e custo moderado.
2. **EfficientNet-B0**: principal candidato de eficiência, com baixo custo computacional relativo.
3. **ConvNeXt-Tiny**: candidato de maior capacidade para verificar se uma representação visual mais forte melhora a generalização no dataset pequeno.

Os três possuem pesos oficiais ImageNet no TorchVision. Para referência, a documentação atual lista aproximadamente 11,7 M parâmetros e 1,81 GFLOPs para ResNet18; 5,3 M e 0,39 GFLOPs para EfficientNet-B0; e 28,6 M e 4,46 GFLOPs para ConvNeXt-Tiny. Esses números servem para contextualizar custo relativo, não para prometer resultados no conjunto de gatos e cães.

## Questão experimental

A pergunta não é apenas “qual modelo acerta mais?”. Ela é:

> Como capacidade, custo computacional e estratégia de fine-tuning afetam a generalização em um conjunto pequeno, mantendo a mesma divisão usada na MLP manual e na CNN do zero?

A comparação final terá três níveis:

```text
A. MLP manual da Etapa 1
B. CNN do zero da Etapa 2
C. Transfer learning da Etapa 3
   ├── ResNet18
   ├── EfficientNet-B0
   └── ConvNeXt-Tiny
```

## Protocolo controlado

Para os três modelos, manter fixos:

- a divisão fornecida pelo professor;
- semente, sempre que tecnicamente possível;
- augmentations de treino, organizadas nos mesmos grupos exigidos;
- métricas, política de checkpoint e formato de artefatos;
- conjunto de validação como critério de escolha;
- conjunto de teste como avaliação final, não como bússola de ajuste.

Cada modelo terá duas fases.

### Fase 1: adaptação da cabeça

- carregar pesos ImageNet oficiais;
- congelar completamente o backbone;
- substituir o classificador final por uma cabeça binária de dois logits;
- treinar somente a nova cabeça com learning rate relativamente maior.

### Fase 2: fine-tuning parcial

- carregar o melhor checkpoint da Fase 1;
- descongelar apenas o último estágio convolucional e o classificador;
- reduzir o learning rate;
- escolher o checkpoint final pela loss de validação.

Fine-tuning total não será o ponto de partida: com apenas 300 imagens de treino, liberar dezenas de milhões de parâmetros cedo demais é uma receita para overfit com apresentação em PowerPoint.

## Preprocessamento

O código deve obter o preprocessamento de validação/teste diretamente de `weights.transforms()` de cada peso oficial. Isso respeita resize, crop, interpolação e normalização esperados por cada modelo. As augmentations de treino serão aplicadas antes da normalização específica do peso.

## Próxima implementação

1. Criar `src/cnn_cats_dogs/transfer_models.py` com uma fábrica para `resnet18`, `efficientnet_b0` e `convnext_tiny`.
2. Generalizar `engine.py` para receber uma fábrica de modelo e um adaptador de output.
3. Criar transforms de treino por peso pré-treinado.
4. Implementar checkpoints de duas fases, preservando modelo, pesos usados, fases congeladas e parâmetros treináveis.
5. Criar CLI `python -m cnn_cats_dogs.transfer` e comparador entre os três modelos.
6. Produzir notebook específico para a Etapa 3 e tabela final comparando as três abordagens.
