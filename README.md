# FreTransformer

> **Fourier Graph Convolution Transformer for Financial Multivariate Time Series Forecasting** — the official implementation (IEEE IJCNN 2024).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: IEEE IJCNN 2024](https://img.shields.io/badge/Paper-IEEE%20IJCNN%202024-blue.svg)](https://doi.org/10.1109/IJCNN60899.2024.10650090)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIJCNN60899.2024.10650090-orange.svg)](https://doi.org/10.1109/IJCNN60899.2024.10650090)

FreTransformer is a frequency-domain, complex-valued Transformer for **Financial Multivariate Time Series (Fin-MTS)** forecasting. Financial series are non-linear, volatile, and contain hidden periodicities (e.g. credit and monetary-policy cycles) that are hard to model in the time domain. FreTransformer maps the input to the frequency domain via the Fast Fourier Transform, exposing these hidden periodicities, and introduces a **Fourier Graph Convolution (FGCN)** that captures both intra-series (within a single series) and inter-series (cross-asset) dependencies in a *unified* way, by folding the time-node adjacency and the cross-series weight into a single learnable complex-valued matrix. A complex-valued Transformer encoder/decoder operates on the spectral coefficients, after which an inverse FFT maps the result losslessly back to the time domain.

## Links

- **Paper:** *Fourier Graph Convolution Transformer for Financial Multivariate Time Series Forecasting*, Junxian Zhou, Shoujin Wang, Yuming Ou — 2024 International Joint Conference on Neural Networks (IJCNN), IEEE.
- **IEEE Xplore:** [doi.org/10.1109/IJCNN60899.2024.10650090](https://doi.org/10.1109/IJCNN60899.2024.10650090) (document `10650090`)
- **Repository:** <https://github.com/AmsonntagChow/FreTransformer>
- This repository is the **official implementation** of the paper.

## Key Features

- **Unified intra- & inter-series modeling.** A single learnable complex-valued matrix (the FGCN kernel) encodes both the time-node adjacency (intra-series) and the cross-series weight (inter-series), instead of modeling them separately.
- **Frequency-domain forecasting.** Inputs are mapped to the frequency domain with `torch.fft.rfft` (amplitude + phase, carried as real/imaginary parts), disclosing hidden periodicities, then reconstructed losslessly with `torch.fft.irfft`.
- **Fully complex-valued Transformer.** Embeddings, multi-head attention, and the encoder/decoder feed-forward blocks are all reimplemented to operate on `torch.complex` tensors, with soft-shrinkage sparsity on the FGCN weights.
- **Fourier Multi-Head Attention.** Attention scores are softmaxed independently over their real and imaginary parts and recombined into a complex attention matrix.
- **Financial-grade evaluation.** Ships forecasting metrics (MAPE, MAE, RMSE, MSE) plus the correlation metrics used in quant research — Information Coefficient (IC) and Rank IC — and a backtesting utility (Sharpe, Sortino, win/loss runs, and an aggregate performance report).
- **End-to-end pipeline.** Data-fetch scripts (Yahoo Finance / Finnhub) → a PyTorch `Dataset` class with sliding lookback/horizon windows → a single `main.py` train/validate/test entrypoint.

## Architecture Overview

The forecasting path is a frequency-domain, sequence-to-sequence encoder–decoder. The top-level wrapper does the FFT/iFFT; the complex Transformer does the heavy lifting in between.

```
            time-domain input x  [B, L, N]
                       │
                       │  permute → rFFT over time axis (torch.fft.rfft, norm='ortho')
                       ▼
            complex spectrum     [B, L//2+1, N]
                       │
        ┌──────────────┴───────────────────────────────┐
        │              FTrans  (complex Transformer)    │
        │                                               │
        │   DataEmbedding  =  FGCN token embedding       │
        │                     + positional embedding     │
        │                                               │
        │   Encoder:  N × EncoderLayer                   │
        │     ( Fourier Multi-Head Attention →           │
        │       complex LayerNorm → FGCN FFN → LayerNorm)│
        │                                               │
        │   Decoder:  M × DecoderLayer                    │
        │     ( masked self-attention → cross-attention  │
        │       over encoder output → FGCN FFN )         │
        │                                               │
        │   forecast head → [B, pre_length//2+1, c_out]  │
        └──────────────┬───────────────────────────────┘
                       │  permute → irFFT (n=L, norm='ortho')
                       ▼
            time-domain output   [B, L, N]
```

Grounded in the actual code:

- **`FTransformer`** (`model/Fourier_Transformer.py`) — top-level wrapper. Permutes `x` `[B, L, N] → [B, N, L]`, applies `torch.fft.rfft` along the time axis, runs the complex Transformer (`self.ftrans`), then reconstructs with `torch.fft.irfft(n=L)`.
- **`FTrans`** (`layers/FTrans.py`) — the complex Transformer: encoder/decoder embeddings, `Encoder`, `Decoder`, and task-specific heads (`long_term_forecast`, `short_term_forecast`, `imputation`, `anomaly_detection`, `classification`). The forecast head returns `dec_out[:, -pre_length:, :]` where `pre_length = configs.pre_length//2+1`.
- **FGCN (Fourier Graph Convolution)** — implemented throughout as a complex linear map: two stacked real-weight matrices applied via `einsum` (the complex multiply `(a+bi)(c+di)`), followed by `F.softshrink` sparsity and `torch.view_as_complex`.
- **Embeddings** (`layers/FEmbed.py`) — `PositionalEmbedding`, complex `TokenEmbedding`, `DataEmbedding`, plus `FixedEmbedding`, `TemporalEmbedding`, `TimeFeatureEmbedding`, `DataEmbedding_wo_pos`, and `PatchEmbedding`.
- **Attention** (`layers/FSelfAttention_Family.py`) — `FullAttention` is the active complex attention (real/imag parts softmaxed separately); wrapped by a complex multi-head `AttentionLayer`. (`DSAttention`, `ProbAttention`, `ReformerLayer`, and `TwoStageAttentionLayer` are also defined but are not used by the default forecast path.)
- **Encoder/Decoder** (`layers/FTransformer_EncDec.py`) — `ConvLayer`, `EncoderLayer`, `Encoder`, `DecoderLayer`, `Decoder`, each with complex LayerNorm residuals and an FGCN feed-forward block.

> **Note:** the model hardcodes `cuda:0` in `FTransformer.__init__` (`self.to('cuda:0')`) and in the validation/test paths. Running on CPU-only or on a different GPU index requires editing these references.

## Repository Structure

```
FreTransformer/
├── main.py                         # Single entrypoint: argparse → build dataset → train/validate → test → loss_plot.png
├── README.md
├── LICENSE                         # MIT License
│
├── data/                           # Data-fetch scripts + the produced CSVs live here
│   ├── fetch_candles_yhf.py        # yfinance fetcher (NO API key) → data/^GSPC_candles_D.csv (S&P 500)
│   ├── fetch_candles.py            # Finnhub fetcher (needs API key) → data/WSPX.MI_candles_D.csv
│   ├── fetch_concat_candles.py     # Finnhub fetcher (needs API key) → data/NVDA_candles_15_concat.csv (multi-resolution)
│   ├── symbol_lookup.py            # Finnhub symbol lookup utility (needs API key)
│   └── data_loader.py              # PyTorch Dataset classes: Dataset_Dhfm, Dataset_ECG, Dataset_Fin, Dataset_Opt
│
├── model/
│   └── Fourier_Transformer.py      # FTransformer: rFFT → FTrans → irFFT wrapper
│
├── layers/
│   ├── __init__.py                 # (empty)
│   ├── FTrans.py                   # FTrans: complex Transformer (embeddings, Encoder, Decoder, task heads)
│   ├── FEmbed.py                   # Complex embeddings (positional, token, data, temporal, time-feature, patch)
│   ├── FSelfAttention_Family.py    # Attention family (FullAttention active; AttentionLayer multi-head wrapper)
│   └── FTransformer_EncDec.py      # ConvLayer, EncoderLayer, Encoder, DecoderLayer, Decoder
│
└── utils/
    ├── utils.py                    # save_model / load_model / evaluate + metrics (MAPE, MAE, RMSE, MSE, IC, Rank-IC)
    ├── model.py                    # Metrics/helper module (eval_*, save_results, dump_config) — helpers, not the model
    ├── masking.py                  # TriangularCausalMask, ProbMask
    └── backtest.py                 # Trading metrics (Sharpe, Sortino, consecutive runs) + performance_mean report
```

> The model class lives in `model/Fourier_Transformer.py`; `utils/model.py` is a *metrics/helper* module despite its name. `main.py` imports `save_model` / `load_model` / `evaluate` from `utils/utils.py` (these symbols are defined in `utils/utils.py`, not `utils/model.py`).
>
> Note on the `Dataset` classes: `data/data_loader.py` defines `Dataset_Dhfm`, `Dataset_ECG`, `Dataset_Fin`, and `Dataset_Opt`, but `main.py` imports and uses **only** `Dataset_Fin` (`from data.data_loader import Dataset_Fin`). The other three classes are present in the file but are not wired into the entrypoint.

## Installation

This is a Python / PyTorch project. (A CUDA-capable GPU is recommended; the validation and test code paths hardcode `cuda:0`.)

```bash
# 1. Clone
git clone https://github.com/AmsonntagChow/FreTransformer.git
cd FreTransformer

# 2. Create an environment (conda or venv)
conda create -n fretransformer python=3.10 -y
conda activate fretransformer
# or:  python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies (install the PyTorch build that matches your CUDA version)
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn einops PyYAML \
            yfinance finnhub-python reformer_pytorch
```

The third-party dependencies, gathered from the actual imports across the codebase:

| Purpose | Packages | Imported in |
| --- | --- | --- |
| Core / model | `torch`, `numpy`, `einops`, `reformer_pytorch` | `main.py`, layers, model |
| Data & preprocessing | `pandas`, `scikit-learn` (`sklearn`), `yfinance`, `finnhub-python` (`finnhub`) | `data/*` |
| Metrics & stats | `scipy` | `utils/utils.py`, `utils/model.py` |
| Plotting / config | `matplotlib`, `seaborn`, `PyYAML` (`yaml`) | `model/Fourier_Transformer.py`, `main.py`, `utils/model.py` |

Suggested `requirements.txt` (install the appropriate PyTorch build for your platform/CUDA):

```text
torch
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
einops
reformer_pytorch
yfinance
finnhub-python
PyYAML
```

## Data Preparation

All paths in the code are **relative**, so always run from the repository root, and make sure a `data/` directory exists. The default financial dataset is the S&P 500 daily index (`^GSPC`), fetched from Yahoo Finance.

### Default dataset (S&P 500, no API key)

```bash
mkdir -p data
python data/fetch_candles_yhf.py     # yfinance, no key required
```

This downloads `^GSPC` daily candles (`2000-01-01` → `2023-11-01`) and writes `data/^GSPC_candles_D.csv` with the standard Yahoo Finance columns: `Date, Open, High, Low, Close, Adj Close, Volume` — i.e. a date column plus six daily features (matching `--enc_in 6`). `Dataset_Fin` treats column 0 as the date and columns `1:` as the six features.

### Optional alternative fetchers (Finnhub — paid API key required)

The Finnhub scripts hardcode the placeholder `api_key="Your Key"`, which you **must** replace with a real Finnhub API key before running (no env-var loading is performed). The `stock_candles` endpoint is a premium endpoint and may return `403` on free tiers.

```bash
python data/fetch_candles.py         # → data/WSPX.MI_candles_D.csv  (WSPX.MI, daily, 2010-01-01 → 2023-08-01)
python data/fetch_concat_candles.py  # → data/NVDA_candles_15_concat.csv  (NVDA, 15-min, multi-resolution)
python data/symbol_lookup.py         # prints symbol_lookup('sp500') results to stdout
```

> Column layout differs by fetcher:
> - `data/fetch_candles_yhf.py` (yfinance) writes `Date, Open, High, Low, Close, Adj Close, Volume`.
> - `data/fetch_candles.py` (Finnhub) inserts a `date` column and drops the `s` (status) and `t` (timestamp) fields, leaving close-first columns: `date, c, h, l, o, v`.
> - `data/fetch_concat_candles.py` (Finnhub) drops only `s`, keeps the `t` field, adds **no** `date` column, and concatenates three deep copies of the frame side-by-side (`axis=1`): a 2-step down-sampled copy, the original, and a 4-step down-sampled copy — producing a wide multi-resolution frame with repeated `c, h, l, o, t, v` blocks.
>
> `Dataset_Fin` only assumes column 0 is the date and treats columns `1:` as features, so feature count/ordering must be consistent with whichever CSV you point it at.

### Dataset / windowing

`main.py` maps `--data` to a CSV via a small parser dict (only these three keys are valid). **Every key instantiates `Dataset_Fin`** — the dict only chooses the CSV path and scaler `type`:

| `--data` | CSV path | Dataset class used |
| --- | --- | --- |
| `Fin` (default) | `data/^GSPC_candles_D.csv` | `Dataset_Fin` |
| `ECG` | `data/ECG_data.csv` | `Dataset_Fin` |
| `COVID` | `data/covid.csv` | `Dataset_Fin` |

`Dataset_Fin` splits train/validation/backtest by **explicit, inclusive date ranges** passed on the command line (defaults below), and produces sliding windows: `seq_length` input steps → next `pre_length` target steps, stride 1.

> **Note:** `Dataset_Fin` fits a `MinMaxScaler` (only when its `type == '1'`, which is the case for all three keys) on the full file *before* the date masks are applied, so normalization statistics are shared across splits.

## Usage / Training

There is no run script or config file — the entrypoint is `python main.py [flags]`, run from the repository root. The train/validate/test loop is guarded by `if __name__ == '__main__':`.

### Quick start

```bash
python main.py
```

Defaults: `--data Fin`, `--train_epochs 100`, `--batch_size 32`, `--learning_rate 0.00013`. The training loop computes its own device as `cuda:0` when CUDA is available, else `cpu`; validation, test, and the model constructor hardcode `cuda:0`. Running it trains for 100 epochs, saves a whole-model checkpoint `output/Fin/train/<epoch>_dhfm.pt` after every epoch (see the checkpoint caveat below), validates each epoch, then `test()` reloads `output/Fin/train/99_dhfm.pt` and reports test metrics, and finally writes `loss_plot.png`.

### Explicit hyperparameters (matching defaults)

```bash
python main.py \
  --seq_length 20 --pre_length 20 \
  --learning_rate 0.00013 --decay_rate 0.5 --exponential_decay_step 5 \
  --n_heads 8 --e_layers 2 --d_model 16
```

### Key arguments

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--data` | str | `Fin` | Dataset key: `Fin`, `ECG`, or `COVID` (all load via `Dataset_Fin`) |
| `--embed_size` | int | `128` | Embedding (hidden) dimension |
| `--hidden_size` | int | `256` | Hidden dimension |
| `--train_epochs` | int | `100` | Number of training epochs |
| `--batch_size` | int | `32` | Batch size |
| `--learning_rate` | float | `0.00013` | RMSprop learning rate |
| `--exponential_decay_step` | int | `5` | Step the LR scheduler every N epochs |
| `--decay_rate` | float | `0.5` | `ExponentialLR` gamma |
| `--validate_freq` | int | `1` | Validate every N epochs |
| `--early_stop` | bool | `True` | Parsed but never used (no early stopping is implemented) |
| `--seq_length` | int | `20` | Input lookback length |
| `--pre_length` | int | `20` | Prediction horizon |
| `--label_len` | int | `48` | Decoder start-token length |
| `--enc_in` | int | `6` | Encoder input size (features) |
| `--dec_in` | int | `6` | Decoder input size |
| `--c_out` | int | `6` | Output size |
| `--d_model` | int | `16` | Model dimension |
| `--n_heads` | int | `8` | Number of attention heads |
| `--e_layers` | int | `2` | Number of encoder layers |
| `--d_layers` | int | `1` | Number of decoder layers |
| `--d_fgcn` | int | `32` | FGCN feed-forward dimension |
| `--dropout` | float | `0.1` | Dropout rate |
| `--activation` | str | `gelu` | Activation function |
| `--factor` | int | `1` | Attention factor |
| `--number_frequency` | int | `1` | Number of frequency components |
| `--task_name` | str | `long_term_forecast` | One of `long_term_forecast`, `short_term_forecast`, `imputation`, `classification`, `anomaly_detection` |
| `--embed` | str | `timeF` | Time embedding: `timeF`, `fixed`, `learned` |
| `--freq` | str | `h` | Frequency for time features (`s/t/h/d/b/w/m`, or e.g. `15min`, `3h`) |
| `--output_attention` | flag | off | `store_true` — output attention in the encoder |
| `--device` | str | `cuda:0` | Parsed but never referenced (see note below) |

Date-split flags (used by `Dataset_Fin`, defaults correspond to the paper's "Phase 12"):

| Flag | Default |
| --- | --- |
| `--start_train` / `--end_train` | `2020-11-18` / `2021-09-18` |
| `--start_vali` / `--end_vali` | `2021-09-19` / `2021-11-18` |
| `--start_backtest` / `--end_backtest` | `2021-11-19` / `2022-05-01` |

Training details: optimizer is `RMSprop` (`eps=1e-08`); LR schedule is `ExponentialLR` (gamma = `--decay_rate`); the loss is `MSELoss(reduction='mean')` (`SmoothL1Loss` and `L1Loss` are present but commented out). The random seed is fixed at `9999`.

> **Important caveats (current code behavior):**
> - **`test()` is hardcoded** to load `output/Fin/train/99_dhfm.pt`. Keep `--train_epochs >= 100` (the default produces epoch index 99) or `test()` will not find the checkpoint; `test()` also hardcodes the `output/Fin/train` directory, so it mismatches for non-`Fin` datasets.
> - **`--device` is never read.** It is parsed (`args.device`) but unused; the training loop computes `device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')`, while `validate()`/`test()` and `FTransformer.__init__` hardcode `cuda:0`. On a CPU-only machine the per-epoch validation will raise a CUDA error.
> - **`--early_stop` is parsed but never used** (no early stopping is implemented).
> - **Checkpoint naming.** `save_model` builds the filename as `str(epoch) + '_dhfm.pt'` only when `epoch` is truthy, so epoch `0` is saved as `_dhfm.pt` (no numeric prefix) and epochs `1..99` as `1_dhfm.pt`..`99_dhfm.pt`.
> - Checkpoints are saved as **whole model objects** (`torch.save(model)`, not `state_dict`), so loading requires the same class definitions/import paths to be importable.

## Evaluation & Backtesting

Forecasting metrics are computed by `utils/utils.py` via the unified `evaluate(y, y_hat)` entrypoint (`y` shaped `[count, time_step, node]`), which returns a tuple:

```
(MAPE, MAE, RMSE, MSE, eval_ic, eval_rank_ic)
```

- **MAPE / MAE / RMSE / MSE** — standard forecasting error metrics.
- **IC (Information Coefficient)** — average Pearson correlation between predictions and ground truth (`scipy.stats.pearsonr`, per-sample then averaged).
- **Rank IC** — Spearman rank correlation (`scipy.stats.spearmanr`, per-sample then averaged).

During training, `main.py` prints these per-epoch (`RAW : MAPE ...; MAE ...; RMSE ...; MSE ...; ic ...; rank_ic ...`).

`utils/backtest.py` provides trading/finance performance metrics for evaluating predicted trading signals:

- `sharpe_ratio(return_series, N, rf)` and `sortino_ratio(series, N, rf)` — annualized risk-adjusted return (total volatility vs downside-only deviation).
- `count_consecutive_pos_values` / `count_consecutive_neg_values` — longest consecutive win / loss runs.
- `performance_mean(df)` — averages a DataFrame of backtest runs and builds an aggregate report dict (initial/final equity, returns, max drawdown, Sharpe, Sortino, Calmar, profit factor, SQN, win rate, trade counts, etc.).

`utils/masking.py` supplies `TriangularCausalMask` (standard autoregressive causal mask) and `ProbMask` (ProbSparse mask) for the attention modules.

> Note: `utils/model.py` is a separate metrics/helper module (`eval_mae`, `eval_mape`, `eval_rmse`, `eval_ic`, `eval_rank_ic`, `eval_all_metrics`, `save_results`, `dump_config`, …). It is not imported by `main.py`; the active metric path is `utils/utils.py`.

## Citation

If you use this code or build on this work, please cite the paper:

```bibtex
@inproceedings{zhou2024fourier,
  author    = {Zhou, Junxian and Wang, Shoujin and Ou, Yuming},
  title     = {Fourier Graph Convolution Transformer for Financial Multivariate Time Series Forecasting},
  booktitle = {2024 International Joint Conference on Neural Networks (IJCNN)},
  year      = {2024},
  pages     = {1--8},
  publisher = {IEEE},
  doi       = {10.1109/IJCNN60899.2024.10650090}
}
```

## License

This project is released under the **MIT License** — Copyright (c) 2024 Junxian Zhou. See [LICENSE](LICENSE) for the full text.

## Acknowledgements

This work builds on the line of frequency-domain Transformers for time-series forecasting (e.g. FEDformer, Autoformer) and on the Fourier Graph Network (FourierGNN) idea of performing graph convolution in the spectral domain. We thank the authors of those works for inspiring the design of the Fourier Graph Convolution used in FreTransformer.
