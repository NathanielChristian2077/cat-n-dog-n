# Etapa 3: Transfer Learning

A Etapa 3 está implementada nesta branch e utiliza pesos ImageNet oficiais do TorchVision para classificar gatos e cães. Ela reaproveita a divisão fornecida pelo professor, as métricas, os artefatos e o cuidado metodológico estabelecido na Etapa 2.

O resultado consolidado e a interpretação para o relatório estão em [`final_results.md`](final_results.md). O notebook correspondente é [`../notebooks/etapa3_transfer_learning.ipynb`](../notebooks/etapa3_transfer_learning.ipynb).

## Backbones implementados

| Identificador | Papel experimental | Parâmetros totais aproximados | VRAM observada no fine-tuning |
|---|---|---:|---:|
| `resnet18` | baseline residual | 11,18 M | 330 MiB |
| `efficientnet_b0` | eficiência | 4,01 M | 231 MiB |
| `convnext_tiny` | maior capacidade | 27,82 M | 634 MiB |

Cada modelo substitui a classificação ImageNet por uma camada de dois logits e usa `CrossEntropyLoss`. A Softmax não aparece dentro do modelo: ela é usada apenas para transformar logits em probabilidades durante a avaliação.

## Protocolo de treino

### Fase 1: adaptação da cabeça

- carrega os pesos oficiais;
- congela o backbone;
- treina somente o classificador binário novo;
- mantém BatchNorm congelada em modo de avaliação quando seus parâmetros permanecem congelados.

### Fase 2: fine-tuning parcial

- recarrega o melhor checkpoint da Fase 1;
- libera o último estágio visual e o classificador;
- reduz o learning rate de `1e-3` para `1e-4`;
- salva o melhor checkpoint de cada fase e o melhor checkpoint global pela loss de validação.

A execução não presume que a segunda fase seja automaticamente melhor. O checkpoint final pode ser o melhor da Fase 1 ou da Fase 2.

## Preprocessamento e augmentation

Validação e teste usam o `weights.transforms()` de cada conjunto de pesos oficial. O treino preserva quatro grupos de augmentation, aplicados antes da normalização ImageNet correspondente:

1. `RandomResizedCrop` para crop/escala espacial;
2. `RandomHorizontalFlip` para reflexão geométrica;
3. `ColorJitter` para perturbação fotométrica;
4. `RandomErasing` para oclusão/regularização.

Todas as imagens permanecem RGB `3×224×224`.

## Seleção e teste final

A tabela de comparação de validação é ordenada pela média da melhor loss de validação em três seeds: `42`, `73` e `101`. O script `compare_transfer.py` não consulta o teste.

Depois de congelar o protocolo, foram avaliados no teste os nove checkpoints já definidos pela combinação de três arquiteturas e três seeds. Essa avaliação mede estabilidade de treinamento e comparação final entre alternativas pré-especificadas. Nenhum hiperparâmetro, limiar, augmentation ou arquitetura deve ser ajustado a partir desses resultados.

## Auditoria de integridade do dataset

Antes de defender métricas próximas de 100%, execute:

```bash
python3 scripts/audit_dataset.py \
  --data-dir dataset \
  --output-dir runs/dataset_audit \
  --dhash-threshold 4
```

O script procura duplicatas byte-a-byte entre splits e candidatos visualmente similares por dHash. Uma duplicata exata entre treino, validação ou teste é sinal de possível vazamento e deve ser registrado, não removido silenciosamente.

## Comandos principais

Comparação por validação:

```bash
python3 scripts/compare_transfer.py \
  --data-dir dataset \
  --output-dir runs/transfer_comparison \
  --seeds 42 73 101 \
  --head-epochs 12 \
  --finetune-epochs 20 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

Avaliação final de um checkpoint:

```bash
python3 scripts/evaluate_transfer.py \
  --checkpoint runs/transfer_comparison/<modelo>/seed_<seed>/checkpoints/best_val_loss.pt \
  --data-dir dataset \
  --output-dir runs/transfer_final_test/<modelo>/seed_<seed> \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

## Artefatos de saída

Cada experimento armazena configuração, ambiente, transforms, histórico por época, summaries JSON, gráficos e predições. Consulte [`runs_to_keep.md`](runs_to_keep.md) para decidir o que versionar após remover checkpoints grandes e tentativas descartadas.
