# cat-n-dog-n

Implementação de uma CNN **criada do zero** com PyTorch e TorchVision para classificação
binária de imagens de gatos e cães. Este repositório é o fallback individual da Etapa 2: ele
não importa arquiteturas prontas nem pesos pré-treinados. Framework, sim; muleta arquitetural,
não. A humanidade sobreviveu.

## Conformidade com a Etapa 2

| Exigência | Implementação |
|---|---|
| CNN com PyTorch e TorchVision | `torch`, `torchvision` e `ImageFolder` |
| Nenhuma arquitetura pronta / nenhum modelo pré-treinado | `ScratchCNN`, declarada camada a camada em `src/cnn_cats_dogs/model.py` |
| Pelo menos duas convoluções | seis `Conv2d` |
| Pelo menos uma pooling | três `MaxPool2d` |
| Pelo menos uma camada totalmente conectada | `Linear(128, 128)` e `Linear(128, 1)` |
| RGB 3 × 224 × 224 | transforms de treino/validação/teste em `data.py` |
| Três técnicas de data augmentation em grupos distintos | crop/scale, flip, color jitter e random erasing |
| Métricas | loss, accuracy, precision, recall, F1, matriz de confusão, curvas e tempo |

## Estrutura esperada do dataset

Use **exatamente** a divisão entregue pelo professor. O código não faz split aleatório, porque
mudar a divisão no meio do trabalho seria uma forma bastante criativa de destruir a comparação.

```text
data/
├── train/
│   ├── cats/
│   └── dogs/
├── val/
│   ├── cats/
│   └── dogs/
└── test/
    ├── cats/
    └── dogs/
```

Os nomes podem ser `gatos/cachorros` ou outro par, desde que sejam iguais nos três splits. Use
`--positive-class` para definir qual pasta é a classe `1`.

## Instalação

Crie um ambiente virtual e instale as dependências. Em Google Colab, PyTorch já vem instalado.
Para GPU local, instale um par PyTorch/TorchVision compatível com seu driver/CUDA antes de
instalar o restante.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\Activate.ps1   # Windows PowerShell

pip install --upgrade pip
pip install -e '.[notebook,dev]'
```

## Treino

```bash
python scripts/train.py \
  --data-dir data \
  --output-dir runs/cnn_scratch \
  --positive-class dogs \
  --epochs 25 \
  --batch-size 32 \
  --num-workers 2 \
  --device auto
```

Em Windows puro, comece com `--num-workers 0`. Em Colab/Linux, `2` ou `4` normalmente é um
ponto de partida aceitável. Ajuste o `batch-size` de acordo com a memória da GPU, pois VRAM
continua sendo uma entidade maldosa que não negocia.

## Saídas salvas

Após o treino, `runs/cnn_scratch/` conterá:

```text
artifacts/
  experiment_config.json   # hiperparâmetros, ambiente e arquitetura
  history.csv              # métricas por época
  run_summary.json         # melhor época, tempo e resultado de teste
  test_predictions.csv     # probabilidades e predições do teste
checkpoints/
  best_val_loss.pt
  last.pt
plots/
  learning_curves.png
  confusion_matrix_test.png
```

O melhor checkpoint é escolhido exclusivamente por **loss de validação**. O conjunto de teste
é consultado uma vez no fim, como manda o bom senso estatístico, esse animal raramente visto.

## Reavaliação

```bash
python scripts/evaluate.py \
  --checkpoint runs/cnn_scratch/checkpoints/best_val_loss.pt \
  --data-dir data \
  --output-dir runs/re_evaluation
```

## Notebook de entrega

Abra `notebooks/etapa2_cnn_pytorch.ipynb` a partir da raiz do repositório. Ele usa os mesmos
módulos do código de produção, mostra as transformações, apresenta a arquitetura, dispara o
treino e carrega os gráficos/resultados para a entrega. O notebook não replica 400 linhas de
código por sadismo pedagógico.

## Testes rápidos

```bash
pytest
```

Os testes verificam o contrato RGB `3 x 224 x 224`, o formato de saída binário e a validação
estrita da estrutura `train/val/test`.

## Preparação para a Etapa 3

A infraestrutura foi organizada para reaproveitar loaders, métricas, gráficos e checkpoints no
transfer learning. O plano está em [`docs/part3_handoff.md`](docs/part3_handoff.md).
