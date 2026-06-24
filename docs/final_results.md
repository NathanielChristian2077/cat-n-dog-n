# Resultados consolidados e interpretação

Este documento registra os resultados finais obtidos com o protocolo definido no projeto. A comparação mantém a divisão original do professor: 300 imagens em treino, 100 em validação e 100 em teste, com 50 imagens por classe em cada split de validação e teste.

## Regra de seleção

A arquitetura foi selecionada por **loss de validação**, não por teste. O conjunto de teste foi consultado somente depois que modelos, seeds, augmentations, fases de fine-tuning e hiperparâmetros já estavam fixados. As três seeds medem estabilidade do treinamento sobre o mesmo split de teste; elas não equivalem a 300 exemplos de teste independentes.

Antes de defender resultados próximos de 100%, inclua no material entregue o resultado de `runs/dataset_audit/summary.json`. Duplicatas byte-a-byte entre splits invalidam a interpretação de generalização. Candidatos perceptuais por dHash devem ser inspecionados visualmente antes de qualquer conclusão.

## Etapa 2: CNN construída do zero

A CNN manual utilizou seis convoluções, três camadas de pooling, BatchNorm, ReLU, augmentation em quatro grupos e classificador binário. A melhor execução disponível atingiu:

| Métrica de teste | Resultado |
|---|---:|
| Loss | 0,5817 |
| Acurácia | 66,0% |
| Precision | 63,8% |
| Recall | 74,0% |
| F1-score | 68,5% |

### Interpretação

A diferença entre o desempenho de treino e validação indicou limitação de generalização. Com apenas 300 imagens, aprender filtros visuais do zero é consideravelmente mais difícil do que adaptar descritores já aprendidos em ImageNet. Este resultado funciona como baseline honesto: ele mostra o custo de não possuir pré-treinamento, não um erro de implementação.

## Etapa 3: seleção por validação

Cada backbone foi treinado em duas fases: adaptação da cabeça com backbone congelado, seguida de fine-tuning parcial do último estágio visual. Os valores abaixo são a média e o desvio-padrão amostral da melhor loss de validação em três seeds.

| Modelo | Melhor loss de validação, média ± DP | Acurácia de validação observada | Parâmetros treináveis no fine-tuning | Pico de VRAM |
|---|---:|---:|---:|---:|
| ResNet18 | 0,04873 ± 0,00086 | 97%–100% | 8,39 M | 330 MiB |
| EfficientNet-B0 | 0,01649 ± 0,00206 | 99% | 414,7 mil | 231 MiB |
| ConvNeXt-Tiny | **0,00188 ± 0,00056** | **100%** | 14,29 M | 634 MiB |

### Seleção por validação

ConvNeXt-Tiny foi o vencedor inequívoco do protocolo de validação. Mesmo seu pior resultado de validação (`0,002513`) apresentou loss menor que o melhor resultado da EfficientNet-B0 (`0,014312`). O padrão apareceu em todas as três seeds, portanto não é explicado por uma inicialização favorável isolada.

Entretanto, loss e acurácia respondem a perguntas diferentes. Quando todas as imagens de validação já são classificadas corretamente, a redução adicional da loss mede principalmente **calibração e confiança**. Por isso, o comportamento no teste deve ser analisado separadamente.

## Etapa 3: resultados finais de teste

| Modelo | Loss de teste, média ± DP | Acurácia, média ± DP | Precision, média ± DP | Recall, média ± DP | F1, média ± DP |
|---|---:|---:|---:|---:|---:|
| ResNet18 | **0,0308 ± 0,0041** | 98,33% ± 0,58 pp | **97,40% ± 1,08 pp** | 99,33% ± 1,15 pp | 98,35% ± 0,57 pp |
| EfficientNet-B0 | 0,0346 ± 0,0043 | 98,33% ± 0,58 pp | 96,78% ± 1,09 pp | **100,00% ± 0,00 pp** | **98,36% ± 0,56 pp** |
| ConvNeXt-Tiny | 0,0575 ± 0,0193 | 98,00% ± 0,00 pp | 96,15% ± 0,00 pp | **100,00% ± 0,00 pp** | 98,04% ± 0,00 pp |

Os resultados individuais de teste foram:

| Modelo | Seed 42 | Seed 73 | Seed 101 |
|---|---:|---:|---:|
| ResNet18, loss | 0,0264 | 0,0344 | 0,0315 |
| ResNet18, acurácia | 99% | 98% | 98% |
| EfficientNet-B0, loss | 0,0329 | 0,0394 | 0,0314 |
| EfficientNet-B0, acurácia | 99% | 98% | 98% |
| ConvNeXt-Tiny, loss | 0,0644 | 0,0357 | 0,0723 |
| ConvNeXt-Tiny, acurácia | 98% | 98% | 98% |

### Leitura dos resultados

- **ConvNeXt-Tiny venceu a validação**, mas apresentou a maior perda média e a maior variabilidade de loss no teste. Ele continuou muito preciso em termos de acurácia, porém perdeu parte da calibração quando exposto ao split final.
- **ResNet18 apresentou a menor loss média de teste**, sugerindo probabilidades mais bem calibradas neste conjunto final.
- **EfficientNet-B0 ofereceu o melhor compromisso global**: F1 médio ligeiramente superior, acurácia empatada com ResNet18, recall perfeito nas três seeds e custo de memória inferior aos demais.
- A diferença entre 98% e 99% corresponde a uma imagem em um teste de 100 exemplos. Não é apropriado declarar superioridade estatística dramática entre ResNet18 e EfficientNet-B0 apenas com esse conjunto.

Todos os modelos mostraram um leve viés em direção à classe positiva, cães. Nas execuções com 98% de acurácia e recall 100%, os erros foram dois gatos classificados como cães. Esse padrão deve aparecer na análise de erros acompanhada das matrizes de confusão e das imagens correspondentes em `test_predictions.csv`.

## Conclusão pronta para o relatório

> A CNN treinada do zero obteve 66,0% de acurácia e F1-score de 68,5%, evidenciando a dificuldade de aprender representações visuais robustas a partir de apenas 300 imagens. Em contraste, os modelos pré-treinados em ImageNet atingiram aproximadamente 98% de acurácia no teste. ConvNeXt-Tiny apresentou a melhor seleção por validação, enquanto ResNet18 alcançou a menor loss média de teste e EfficientNet-B0 ofereceu o melhor compromisso entre desempenho, estabilidade e custo computacional. Os resultados reforçam que transfer learning é particularmente vantajoso em cenários de poucos dados, mas também mostram que a melhor validação não garante a melhor calibração no teste final.

## Limitações e cuidados metodológicos

1. O conjunto de teste contém apenas 100 imagens. Uma mudança de uma imagem equivale a um ponto percentual de acurácia.
2. As seeds repetem o treinamento, mas reutilizam a mesma validação e o mesmo teste. O desvio-padrão mede estabilidade de otimização, não substitui validação cruzada.
3. A auditoria de duplicatas entre splits deve ser mantida como evidência de integridade do dataset.
4. Depois da execução dos testes reportados aqui, não se deve ajustar limiar, augmentation, arquitetura ou hiperparâmetro usando esses mesmos resultados.
