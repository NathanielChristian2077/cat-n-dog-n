# Artefatos de execução a manter no repositório

O diretório `runs/` pode conter dezenas de checkpoints, cópias intermediárias e arquivos grandes. Para uma entrega acadêmica reproduzível, preserve evidência suficiente para ler resultados e reconstruir o protocolo, sem transformar o repositório em depósito de pesos binários.

## Regra geral

- **Nunca versionar o dataset**.
- **Não versionar checkpoints `.pt`**, exceto se o professor exigir explicitamente o modelo treinado.
- Manter configurações, históricos CSV, resumos JSON, gráficos e tabelas agregadas.
- Manter somente as execuções que sustentam números citados no relatório.

A configuração do repositório ignora `dataset/` e `runs/**/checkpoints/`. Arquivos importantes de `runs/` podem ser adicionados normalmente sem incluir pesos pesados.

## Auditoria do dataset

Manter:

```text
runs/dataset_audit/
├── summary.json
├── exact_cross_split_duplicates.csv
└── near_cross_split_candidates.csv
```

`image_manifest.csv` é opcional. Ele melhora rastreabilidade, mas pode ser removido se o repositório precisar ficar mais enxuto.

## Etapa 2: CNN do zero

Para a execução final usada no relatório, por exemplo `runs/cnn_scratch_v2/`, manter:

```text
artifacts/
├── experiment_config.json
├── history.csv
├── run_summary.json
└── test_predictions.csv
plots/
├── learning_curves.png
└── confusion_matrix_test.png
```

Pode remover:

```text
checkpoints/
last.pt
best_val_loss.pt
```

## Etapa 3: comparação por validação

Em `runs/transfer_comparison/`, manter no nível raiz:

```text
comparison_runs.csv
comparison_summary.csv
```

Para cada combinação de modelo e seed citada no relatório, manter:

```text
<modelo>/seed_<seed>/
├── artifacts/
│   ├── experiment_config.json
│   ├── history.csv
│   └── run_summary.json
└── plots/
    └── learning_curves.png
```

Não é necessário manter nove cópias de matrizes de confusão nessa etapa, porque a seleção não abriu o teste. Não manter `checkpoints/` depois de concluir as avaliações finais.

## Etapa 3: teste final

Em `runs/transfer_final_test/`, manter, para cada modelo e seed:

```text
<modelo>/seed_<seed>/
├── artifacts/
│   ├── evaluation_summary.json
│   └── test_predictions.csv
└── plots/
    └── confusion_matrix_test.png
```

Esses arquivos sustentam as tabelas de teste, a análise de erros e as figuras do relatório.

## Sequência prática antes do commit

1. Confirme os números em `docs/final_results.md` contra os JSONs e CSVs mantidos.
2. Remova checkpoints e diretórios de tentativas descartadas.
3. Execute `git status` e confira se não há dataset, cache de pesos ou ambiente virtual.
4. Adicione apenas os artefatos curados:

```bash
git add README.md docs notebooks src scripts tests pyproject.toml .gitignore
git add runs/dataset_audit/summary.json \
        runs/dataset_audit/exact_cross_split_duplicates.csv \
        runs/dataset_audit/near_cross_split_candidates.csv
# Adicione os diretórios curados de Etapa 2 e Etapa 3 explicitamente.
```

5. Revise o tamanho do commit antes do push:

```bash
git status --short
git diff --stat --cached
```
