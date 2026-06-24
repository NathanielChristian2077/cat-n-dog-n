# Etapa 3: Transfer Learning

A implementação da Etapa 3 está pronta nesta branch. Ela reaproveita o contrato experimental consolidado na Etapa 2: mesma divisão `train/val/test`, métricas, gráficos, checkpoints e estrutura de artefatos. A diferença é a origem da representação visual: pesos ImageNet oficiais do TorchVision, com o classificador adaptado para gatos versus cães.

## Modelos implementados

- `resnet18`: baseline residual clássico.
- `efficientnet_b0`: comparação de eficiência.
- `convnext_tiny`: comparação de maior capacidade.

Cada um recebe uma camada final nova com **dois logits** e é treinado com `CrossEntropyLoss`. A Softmax é aplicada apenas para gerar a probabilidade positiva nas métricas, não dentro do modelo.

## Pipeline implementado

```text
src/cnn_cats_dogs/
├── transfer_models.py    # factory dos pesos oficiais, cabeça binária e freeze/unfreeze
├── transfer_config.py    # hiperparâmetros e validação
├── transfer_data.py      # transforms por peso oficial e loaders do dataset do professor
├── transfer_engine.py    # treino em duas fases, métricas, checkpoints e avaliação final
├── dataset_audit.py      # auditoria de duplicatas exatas e candidatos perceptuais entre splits
├── transfer.py           # uma execução individual
├── transfer_compare.py   # comparação dos três modelos por validação e seeds
└── transfer_evaluate.py  # avaliação única do checkpoint vencedor no teste
```

## Estratégia de fine-tuning

### Fase 1: adaptação da cabeça

- carrega pesos ImageNet oficiais;
- congela todo o backbone;
- treina apenas o classificador binário novo;
- mantém BatchNorm congelada em modo de avaliação para não destruir estatísticas ImageNet com batches pequenos.

### Fase 2: fine-tuning parcial

- recarrega o melhor checkpoint da Fase 1;
- libera apenas o último estágio visual e o classificador;
- usa learning rate menor;
- seleciona o checkpoint global pela menor loss de validação.

O checkpoint escolhido pode pertencer à Fase 1 ou à Fase 2. Fine-tuning não recebe troféu de participação só porque veio depois.

## Preprocessamento

Validação e teste usam exatamente `weights.transforms()` de cada peso oficial. O treino mantém os mesmos grupos de augmentation exigidos pelo enunciado, antes da normalização ImageNet correspondente:

1. `RandomResizedCrop`: grupo espacial de crop/escala;
2. `RandomHorizontalFlip`: grupo geométrico;
3. `ColorJitter`: grupo fotométrico;
4. `RandomErasing`: grupo de oclusão/regularização.

As imagens permanecem RGB `3×224×224`.

## Protocolo correto de seleção

`transfer_compare.py` deliberadamente **não consulta o teste**. Ele executa os modelos com seeds diferentes e ordena `comparison_summary.csv` pela média da melhor loss de validação. Só depois de escolher arquitetura e seed pelo conjunto de validação deve-se executar `transfer_evaluate.py` uma única vez no teste.

## Auditoria de integridade do dataset

Resultados próximos de 100% em validação merecem ser verificados antes de virar frase de relatório. O script abaixo produz:

- `exact_cross_split_duplicates.csv`: repetição byte-a-byte entre splits, evidência objetiva de vazamento;
- `near_cross_split_candidates.csv`: pares visualmente similares por dHash, candidatos para inspeção visual, não prova automática;
- `image_manifest.csv` e `summary.json` para rastreabilidade.

```bash
python3 scripts/audit_dataset.py \
  --data-dir dataset \
  --output-dir runs/dataset_audit \
  --dhash-threshold 4
```

## Execução

Triagem dos três modelos em uma seed:

```bash
python3 scripts/compare_transfer.py \
  --data-dir dataset \
  --output-dir runs/transfer_smoke \
  --seeds 42 \
  --head-epochs 12 \
  --finetune-epochs 20 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

Comparação formal com três seeds:

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

Treino de um modelo isolado, sem abrir o teste:

```bash
python3 scripts/train_transfer.py \
  --data-dir dataset \
  --output-dir runs/efficientnet_b0_seed42 \
  --architecture efficientnet_b0 \
  --seed 42 \
  --head-epochs 12 \
  --finetune-epochs 20 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

Avaliação final de um checkpoint selecionado por validação:

```bash
python3 scripts/evaluate_transfer.py \
  --checkpoint runs/efficientnet_b0_seed42/checkpoints/best_val_loss.pt \
  --data-dir dataset \
  --output-dir runs/efficientnet_b0_final_test \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```
