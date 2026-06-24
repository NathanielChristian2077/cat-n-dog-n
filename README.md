# cat-n-dog-n

Implementação de uma CNN **criada do zero** com PyTorch e TorchVision para classificação binária de imagens de gatos e cães. Este repositório é o fallback individual da Etapa 2: ele não importa arquiteturas prontas nem pesos pré-treinados. Framework, sim; muleta arquitetural, não.

## Conformidade com a Etapa 2

| Exigência | Implementação |
|---|---|
| CNN com PyTorch e TorchVision | `torch`, `torchvision` e loaders TorchVision |
| Nenhuma arquitetura pronta / nenhum modelo pré-treinado | `ScratchCNN`, declarada camada a camada em `src/cnn_cats_dogs/model.py` |
| Pelo menos duas convoluções | seis `Conv2d` |
| Pelo menos uma pooling | três `MaxPool2d` |
| Pelo menos uma camada totalmente conectada | `Linear(128, 128)` e `Linear(128, 1)` |
| RGB 3 × 224 × 224 | transforms de treino/validação/teste em `data.py` |
| Três técnicas de data augmentation em grupos distintos | crop/scale, flip, color jitter e random erasing |
| Métricas | loss, accuracy, precision, recall, F1, matriz de confusão, curvas, tempo e pico de VRAM |

## Dataset: sem reorganização manual

Use **exatamente** a divisão entregue pelo professor. O projeto não cria split aleatório, porque mudar a divisão no meio do trabalho seria uma forma bastante criativa de destruir a comparação.

O loader aceita estas variações sem copiar, renomear ou criar links simbólicos:

```text
dataset/
├── train/ | training/ | treino/
├── val/   | validation/ | validacao/
└── test/  | testing/ | teste/
```

Cada split deve conter as duas pastas de classe. Os nomes podem ser `cats/dogs`, `gatos/cachorros` ou equivalentes, inclusive quando o conjunto do professor vem dentro de **um** diretório pai adicional:

```text
dataset/
└── cats_dogs_professor/
    ├── treino/
    │   ├── Gatos/
    │   └── Cachorros/
    ├── validacao/
    │   ├── Gatos/
    │   └── Cachorros/
    └── teste/
        ├── Gatos/
        └── Cachorros/
```

A classe positiva continua explícita. `--positive-class dogs` também reconhece `cachorros`; use o nome de uma pasta caso o dataset tenha rótulos fora desse par.

No início de cada treino, o programa mostra as contagens negativa/positiva de `train`, `val` e `test`. O treino aplica `pos_weight = negativos / positivos` na BCE para compensar uma divisão de treino desbalanceada, mas validação e teste continuam com BCE comum. Isso não muda os arquivos nem os splits do professor, só impede que a rede ganhe pontos fingindo que toda imagem é a classe majoritária.

## Setup local com CUDA

Use Python 3.10 ou superior. Primeiro instale um par `torch`/`torchvision` com CUDA compatível com o driver NVIDIA do sistema pelo seletor oficial do PyTorch. Depois instale o projeto no ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Exemplo CUDA 12.6. Troque pelo comando do seletor oficial se o driver pedir outra variante.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[notebook,dev]'
```

Verificação obrigatória, porque uma GPU ignorada é só um aquecedor caro:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem CUDA')"
```

## Treino rápido local

Coloque o dataset dentro de `dataset/` e rode da raiz do repositório:

```bash
python scripts/train.py \
  --data-dir dataset \
  --output-dir runs/cnn_scratch_balanced \
  --positive-class dogs \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --num-workers 8 \
  --device cuda
```

`batch-size 32` é o ponto de partida deliberado para datasets acadêmicos pequenos: ele dá mais atualizações por época. Uma RTX com memória sobrando não exige batch 128; isso só deixa a GPU entediada enquanto o otimizador recebe dois ou três gradientes e começa a inventar uma personalidade.

O perfil CUDA já usa AMP FP16, TF32 quando disponível, cuDNN autotuning, `channels_last`, AdamW fundido quando o build suporta, `pin_memory`, workers persistentes e prefetch. A execução padrão privilegia throughput; para uma repetição estritamente determinística, use a `TrainingConfig(deterministic=True)` no notebook ou no código Python.

Se houver erro de memória, reduza apenas o batch: `32 → 16`. Não mexa no tamanho 224×224, que é requisito da etapa.

## Saídas salvas

Após o treino, `runs/cnn_scratch_balanced/` conterá:

```text
artifacts/
  experiment_config.json   # hiperparâmetros, layout resolvido, distribuição de classes, ambiente e arquitetura
  history.csv              # métricas por época, tempo e pico de VRAM
  run_summary.json         # melhor época, tempo e resultado de teste
  test_predictions.csv     # probabilidades e predições do teste
checkpoints/
  best_val_loss.pt
  last.pt
plots/
  learning_curves.png
  confusion_matrix_test.png
```

O melhor checkpoint é escolhido exclusivamente por **loss de validação**. O conjunto de teste é consultado uma vez no fim, como manda o bom senso estatístico, esse animal raramente visto.

## Reavaliação

```bash
python scripts/evaluate.py \
  --checkpoint runs/cnn_scratch_balanced/checkpoints/best_val_loss.pt \
  --data-dir dataset \
  --output-dir runs/re_evaluation \
  --batch-size 32 \
  --num-workers 8 \
  --device cuda
```

## Notebook de entrega

Abra `notebooks/etapa2_cnn_pytorch.ipynb` a partir da raiz do repositório. Ele usa os mesmos módulos do código de produção, mostra as transformações, apresenta a arquitetura, dispara o treino e carrega os gráficos/resultados para a entrega. O notebook não replica 400 linhas de código por sadismo pedagógico.

## Testes rápidos

```bash
pytest
```

Os testes verificam o contrato RGB `3 × 224 × 224`, a saída binária e a descoberta/validação da estrutura de dataset.

## Preparação para a Etapa 3

A infraestrutura foi organizada para reaproveitar loaders, métricas, gráficos e checkpoints no transfer learning. O plano está em [`docs/part3_handoff.md`](docs/part3_handoff.md).
