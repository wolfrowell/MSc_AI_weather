# CLAUDE.md — MSc_AI_weather

Projeto de dissertação de mestrado (PUCRS/PPGCC) sobre fine-tuning do GraphCast para previsão regional do Brasil.

## Contexto geral

- **Modelo base**: GraphCast_operational (0.25°, 13 pressure levels, mesh_size=6, ~36M params)
- **Objetivo**: Fine-tuning para previsão regional do Brasil
- **Dataset atual**: ERA5 example dataset (2022-01-01) do GCS — ainda não são dados reais brasileiros
- **Orientador**: Lucas Kupssinskü, PhD

## Estrutura do projeto

```
MSc_AI_weather/
├── run_baseline.py              ← avalia GraphCast pretrained + persistence; salva predições em results/
├── run_metrics.py               ← recomputa métricas a partir de predições salvas (sem re-rodar o modelo)
├── run_training.py              ← entry point do fine-tuning
├── visualize_results.ipynb      ← gráficos RMSE, skill score, mapas
└── src/
    ├── patch.py                 ← GraphCastPatcher (adiciona freeze_encoder/freeze_processor ao graphcast.py)
    ├── data_loader.py           ← GCSDataLoader
    ├── model.py                 ← GraphCastModel (strategy-agnostic)
    ├── baselines.py             ← PersistenceBaseline, PretrainedBaseline
    ├── evaluator.py             ← Evaluator (RMSE, ACC, skill scores → CSV)
    ├── metrics.py               ← lat_weights, rmse, acc (Eq. 20, 29 do paper)
    ├── trainer.py               ← Trainer + TrainingConfig
    └── finetuning/
        ├── base.py              ← FineTuningStrategy (ABC) + FineTuningState
        └── frozen_modules.py    ← FrozenModuleStrategy (implementação atual)
```

## Arquitetura GraphCast

```
grid → [grid2mesh_gnn] → mesh → [mesh_gnn x16] → mesh → [mesh2grid_gnn] → grid
         (encoder ~3M)           (processor ~30M)          (decoder ~3M)
```

`GraphCastPatcher` (src/patch.py) insere `jax.lax.stop_gradient` após o encoder e/ou processor para economizar memória no backprop. Isso é feito via patches de string no arquivo instalado via pip, pois graphcast não tem API pública para isso.

## Tradeoff de memória por estratégia

| O que treinar | stop_gradient | Memória backprop |
|---|---|---|
| Decoder | após encoder + processor | Baixa |
| Processor | após encoder | Média |
| Encoder | nenhum | Alta |

## Fluxo de avaliação

```bash
# 1. Roda o modelo — salva predições em results/*.nc
python run_baseline.py --eval-steps 4 --dataset "source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc"

# 2. Recomputa métricas sem re-rodar o modelo
python run_metrics.py
```

Predições são salvas como NetCDF (`results/preds_*.nc`) para que bugs em `metrics.py` possam ser corrigidos sem re-rodar o forward pass.

## Métricas implementadas

Seguem exatamente o paper GraphCast (Lam et al. 2023), seção "Verification Methods":
- **RMSE** (Eq. 20): latitude-weighted, sqrt INSIDE a média sobre inicializações
- **ACC** (Eq. 29): anomaly correlation coefficient com climatologia ERA5 1993–2016
- Pesos de latitude e nível idênticos ao `graphcast/losses.py`

## Como adicionar um novo método de fine-tuning

1. Criar `src/finetuning/lora.py` (subclasse de `FineTuningStrategy`)
2. Implementar os 4 métodos: `prepare`, `build_predictor`, `merge_for_inference`, `print_summary`
3. Em `run_training.py`, trocar uma linha: `strategy = LoRAStrategy(...)`
4. `GraphCastModel` e `Trainer` não precisam mudar

## Dependências importantes

- `jax[cuda12]` — JAX com suporte CUDA 12 (A6000/Ampere)
- `numpy<2.0` — evita quebra na aritmética de datetime
- `xarray>=2024.9.0,<2025.0.0` — necessário para DataTree
- `pandas<3.0` — evita quebra na resolução de timedelta (ns vs us)
- `graphcast @ git+https://github.com/deepmind/graphcast.git`

## Referências

- Lam et al. (2023) — GraphCast paper (métricas: Eq. 20, 29)
- Nipen et al. (2024) arxiv 2409.02891 — stretched-grid GNN para região Nórdica (análogo Brasil)
