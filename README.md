# cat-n-dog-n

Projeto de classificação binária de gatos e cães para as etapas de redes neurais do trabalho de IA. O repositório concentra a **CNN implementada do zero** da Etapa 2 e o **transfer learning com pesos ImageNet oficiais** da Etapa 3. A MLP manual da Etapa 1 é tratada como baseline conceitual e como origem dos presets de head comparados na CNN.

## Escopo e conformidade

| Etapa | Implementação | Regra central atendida |
|---|---|---|
| 1 | MLP manual, executada separadamente | sem biblioteca de rede neural pronta |
| 2 | `ScratchCNN` declarada camada a camada | sem arquitetura pronta e sem pesos pré-treinados |
| 3 | ResNet18, EfficientNet-B0 e ConvNeXt-Tiny | pesos oficiais TorchVision adaptados por fine-tuning |

A CNN da Etapa 2 recebe RGB `3×224×224`, contém seis convoluções, três `MaxPool2d`, normalização em lote, ReLU e um classificador denso configurável. O treinamento usa quatro famílias de augmentation: crop/escala, reflexão horizontal, perturbação fotométrica e oclusão leve.

A Etapa 3 mantém o mesmo dataset, as mesmas métricas e o mesmo protocolo de seleção, trocando somente a origem da representação visual por pesos ImageNet oficiais. Cada backbone usa dois logits e `CrossEntropyLoss`; a Softmax é aplicada apenas para converter logits em probabilidades durante as métricas.

## Estrutura do repositório

```text
src/cnn_cats_dogs/
├── data.py                 # dataset da Etapa 2 e validação rígida dos splits
├── model.py                # CNN construída do zero
├── engine.py               # treino e avaliação da Etapa 2
├── transfer_models.py      # factory ResNet18, EfficientNet-B0 e ConvNeXt-Tiny
├── transfer_data.py        # transforms específicos dos pesos ImageNet
├── transfer_engine.py      # head adaptation + fine-tuning parcial
├── dataset_audit.py        # auditoria de duplicatas entre splits
├── metrics.py              # accuracy, precision, recall, F1 e matriz de confusão
└── visualization.py        # curvas e matriz de confusão

scripts/
├── train.py                # compatibilidade: treino básico da CNN do zero
├── train_scratch.py        # treino configurável da Etapa 2
├── compare_phase1_heads.py # ablação dos heads derivados da Etapa 1
├── evaluate.py             # reavaliação de checkpoint da Etapa 2
├── train_transfer.py       # treino individual da Etapa 3
├── compare_transfer.py     # comparação dos três backbones por validação
├── evaluate_transfer.py    # teste final de checkpoint transfer learning
└── audit_dataset.py        # auditoria de integridade do dataset

docs/
├── phase1_cnn_comparison.md
├── part3_handoff.md
├── final_results.md
└── runs_to_keep.md

notebooks/
├── etapa2_cnn_pytorch.ipynb
└── etapa3_transfer_learning.ipynb
```

## Contrato do dataset

Use a divisão entregue pelo professor. O projeto **não cria nem reorganiza** split aleatório.

```text
dataset/
├── train/{cats,dogs}/
├── val/{cats,dogs}/
└── test/{cats,dogs}/
```

Também são aceitas variações em português, como `treino/validacao/teste` e `gatos/cachorros`, inclusive quando os três splits estiverem dentro de um diretório pai adicional. A classe positiva é explícita e normalmente definida como `dogs`.

Antes de interpretar resultados muito altos, audite o dataset:

```bash
python3 scripts/audit_dataset.py \
  --data-dir dataset \
  --output-dir runs/dataset_audit \
  --dhash-threshold 4
```

Duplicata exata entre splits é evidência de vazamento. Candidatos de similaridade perceptual por dHash precisam de inspeção visual, pois são apenas alertas.

## Instalação

Use Python 3.10 ou superior. Instale uma versão de `torch` e `torchvision` compatível com o driver NVIDIA e a CUDA do ambiente. Depois:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook,dev]'

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest
```

Os scripts em `scripts/` inserem `src/` no `sys.path`; portanto, também funcionam sem instalação editável. A instalação editável continua recomendada para notebooks, testes e uso dos módulos com `python -m`.

## Etapa 2: CNN do zero

O treino final de referência da CNN manual pode ser reproduzido com o preset baseline:

```bash
python3 scripts/train_scratch.py \
  --data-dir dataset \
  --output-dir runs/cnn_scratch_reproduction \
  --architecture cnn_baseline_128_sigmoid1 \
  --positive-class dogs \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --num-workers 8 \
  --device cuda
```

A ablação que transfere as topologias densas mais fortes da MLP para um backbone convolucional fixo está descrita em [`docs/phase1_cnn_comparison.md`](docs/phase1_cnn_comparison.md):

```bash
python3 scripts/compare_phase1_heads.py \
  --data-dir dataset \
  --output-dir runs/phase1_head_comparison \
  --seeds 42 73 101 \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --num-workers 8 \
  --device cuda
```

## Etapa 3: transfer learning

A comparação dos backbones é feita sem consultar o teste durante a seleção:

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

O protocolo possui duas fases:

1. **Adaptação da cabeça:** backbone congelado e treino apenas do classificador binário novo.
2. **Fine-tuning parcial:** recarrega o melhor checkpoint da fase anterior, libera o último estágio visual e reduz o learning rate.

A escolha é feita pela menor loss de validação. Só depois se executa o teste final do checkpoint escolhido:

```bash
python3 scripts/evaluate_transfer.py \
  --checkpoint runs/transfer_comparison/convnext_tiny/seed_101/checkpoints/best_val_loss.pt \
  --data-dir dataset \
  --output-dir runs/transfer_final_test/convnext_tiny/seed_101 \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

## Resultados consolidados

A CNN do zero alcançou `66,0%` de acurácia e `68,5%` de F1 no teste, evidenciando a limitação de aprender uma representação visual a partir de apenas 300 imagens de treino.

No transfer learning, a seleção por validação favoreceu ConvNeXt-Tiny. No teste final, porém, ResNet18 e EfficientNet-B0 apresentaram desempenho médio ligeiramente superior ou equivalente com menor custo computacional. O detalhamento, a interpretação correta das seeds e a tabela pronta para o relatório estão em [`docs/final_results.md`](docs/final_results.md).

## Artefatos produzidos

Cada execução salva, em seu diretório de `runs/`:

```text
artifacts/
├── experiment_config.json
├── history.csv
├── run_summary.json
└── test_predictions.csv
checkpoints/
├── best_val_loss.pt
└── last.pt
plots/
├── learning_curves.png
└── confusion_matrix_test.png
```

Checkpoints podem ser grandes; o guia [`docs/runs_to_keep.md`](docs/runs_to_keep.md) indica quais artefatos manter no repositório e quais podem ser descartados após a avaliação final.

## Notebooks de entrega

- [`notebooks/etapa2_cnn_pytorch.ipynb`](notebooks/etapa2_cnn_pytorch.ipynb): CNN manual, dados, arquitetura, treino e leitura dos artefatos finais da Etapa 2.
- [`notebooks/etapa3_transfer_learning.ipynb`](notebooks/etapa3_transfer_learning.ipynb): auditoria, modelos pré-treinados, protocolo em duas fases, comparação de validação, teste final e análise de trade-offs.

Os notebooks usam os mesmos módulos de produção. Eles existem para explicar e reproduzir o experimento, não para duplicar centenas de linhas de lógica de treino.
