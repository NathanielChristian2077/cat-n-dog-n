# Comparação Etapa 1 → Etapa 2

A Etapa 2 não copia uma MLP pixel-a-pixel para dentro de uma CNN. A entrada agora é RGB `3×224×224`, e a extração espacial é responsabilidade dos blocos convolucionais. A comparação preserva o que é transferível: a **topologia do classificador denso final**.

## O que está sendo comparado, e o que não está

Esta é uma **ablação de head**, não uma alegação de que uma MLP e uma CNN sejam a mesma arquitetura com outro corte de cabelo.

- Na Etapa 1, o head recebia atributos planos dos pixels.
- Na Etapa 2, o head recebe um vetor de 128 descritores espaciais aprendido pelo backbone convolucional.
- Portanto, preservamos a ordem e as larguras relativas das camadas densas vencedoras, mas não o tamanho da primeira matriz de pesos nem o viés espacial da CNN.

A pergunta experimental fica precisa: **qual topologia de decisão que funcionou bem na MLP também se adapta melhor a uma representação convolucional fixa?**

## Backbone fixo

Todos os experimentos usam exatamente o mesmo extrator:

```text
RGB 3×224×224
  → Conv 3→32 → Conv 32→32 → MaxPool
  → Conv 32→64 → Conv 64→64 → MaxPool
  → Conv 64→128 → Conv 128→128 → MaxPool
  → AdaptiveAvgPool 1×1
  → vetor de 128 características
```

Logo, qualquer diferença observada entre as execuções vem do head denso e do formato de saída, não de uma mudança escondida no extrator visual.

## Presets derivados dos melhores resultados da Etapa 1

| Rank Etapa 1 | Preset | Head CNN após as 128 features | Loss |
|---:|---|---|---|
| 1 | `phase1_rank1_32x64x512_softmax2` | `128 → 32 → 64 → 512 → 2` | `CrossEntropyLoss` |
| 2 | `phase1_rank2_32x64x512_sigmoid1` | `128 → 32 → 64 → 512 → 1` | `BCEWithLogitsLoss` |
| 3 | `phase1_rank3_64x32x512_sigmoid1` | `128 → 64 → 32 → 512 → 1` | `BCEWithLogitsLoss` |
| 4 | `phase1_rank4_128x32x512_sigmoid1` | `128 → 128 → 32 → 512 → 1` | `BCEWithLogitsLoss` |
| 5 | `phase1_rank5_64x32x512_softmax2` | `128 → 64 → 32 → 512 → 2` | `CrossEntropyLoss` |
| 6 | `phase1_rank6_128x32x512_softmax2` | `128 → 128 → 32 → 512 → 2` | `CrossEntropyLoss` |

As variantes `softmax2` retornam **dois logits crus**, não uma camada Softmax explícita no modelo. `CrossEntropyLoss` aplica a normalização de maneira numericamente estável. As variantes `sigmoid1` retornam **um logit cru**, e `BCEWithLogitsLoss` faz a mesma combinação estável com sigmoid.

## Protocolo de comparação

Mantenha fixos: dataset, splits, transforms, backbone convolucional, batch size, learning rate, weight decay, número máximo de épocas, scheduler, early stopping e dropout. Varie apenas:

1. a sequência de larguras do head;
2. a parametrização binária da saída: `sigmoid1` ou `softmax2`;
3. a seed, para separar arquitetura de sorte de inicialização.

A comparação completa usa três seeds: `42`, `73` e `101`, totalizando 18 execuções. Os arquivos `comparison_runs.csv` e `comparison_summary.csv` guardam resultados individuais e médias/desvios-padrão.

O `comparison_summary.csv` é ordenado por **média da melhor loss de validação**, com desvio-padrão como desempate. Métricas de teste permanecem registradas para o relatório final, mas não definem a arquitetura vencedora e não devem orientar novos ajustes.

## Execução

Após instalar o projeto em modo editável, execute todos os seis presets:

```bash
python -m cnn_cats_dogs.compare \
  --data-dir dataset \
  --output-dir runs/phase1_head_comparison \
  --positive-class dogs \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --num-workers 8 \
  --device cuda
```

Para uma triagem inicial com uma única seed:

```bash
python -m cnn_cats_dogs.compare \
  --data-dir dataset \
  --output-dir runs/phase1_head_smoke \
  --seeds 42 \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --num-workers 8 \
  --device cuda
```

Para executar um preset isolado:

```bash
python -m cnn_cats_dogs.cli \
  --data-dir dataset \
  --output-dir runs/rank1_softmax2_seed42 \
  --architecture phase1_rank1_32x64x512_softmax2 \
  --seed 42 \
  --epochs 40 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --num-workers 8 \
  --device cuda
```
