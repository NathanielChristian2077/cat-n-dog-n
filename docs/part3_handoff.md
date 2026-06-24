# Handoff para a Etapa 3: Transfer Learning

A Etapa 2 já isola tudo que deve ser reutilizado na Etapa 3: validação do dataset, loaders,
métricas, gráficos, checkpoints e estrutura de artefatos. A Etapa 3 deve trocar apenas o
módulo de modelo e o preprocessamento de entrada específico de cada peso pré-treinado.

## Próxima implementação

1. Criar `src/cnn_cats_dogs/transfer_models.py`.
2. Implementar pelo menos duas arquiteturas do TorchVision, recomendadas aqui:
   - `resnet18` como baseline robusto;
   - `mobilenet_v3_small` como contraste de menor custo computacional.
3. Carregar pesos oficiais do TorchVision e substituir as cabeças finais para uma saída binária.
4. Rodar fine-tuning em duas fases:
   - aquecimento da cabeça final com backbone congelado;
   - fine-tuning parcial ou total com learning rate menor.
5. Trocar os transforms neutros da Etapa 2 pelos transforms associados aos pesos de cada
   arquitetura, preservando RGB 3 x 224 x 224.
6. Manter a mesma divisão train/val/test, a mesma seed quando possível, e salvar resultados
   no mesmo formato de `runs/` para comparação justa.

## Comparação exigida no relatório

Para cada uma das três abordagens (MLP manual, CNN do zero, transfer learning), registrar:

- acurácia, loss, precision, recall e F1 no teste;
- tempo de treinamento;
- matriz de confusão;
- curva de aprendizado;
- tamanho do modelo e número de parâmetros treináveis;
- análise de erros e custo computacional.

## Decisão de desenho

O módulo `engine.py` continua reutilizável. A Etapa 3 precisa aceitar uma fábrica de modelos
em vez de instanciar `ScratchCNN` diretamente. Isso será feito quando os dois backbones forem
implementados, para não misturar transferência de aprendizado com a entrega obrigatória da
Etapa 2 e produzir uma lasanha de abstrações antes da hora.
