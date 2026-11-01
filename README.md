# Hinglish Sentiment Analyzer

A research-oriented sentiment analysis project for Hindi-English code-mixed comments written in Roman script. It compares a TF-IDF baseline with neural models and provides a Gradio dashboard for single-comment and batch analysis.

> **Use with care:** These models are experimental, learn biases present in a small dataset, and must not be used for moderation or decisions about people.

## Problem statement

Hinglish spelling varies widely, and sentiment can depend on context, slang, negation, and code-switching. This project tests how classical text features and small neural models handle that problem on one published corpus.

## Objectives

- Compare a strong TF-IDF baseline with neural architectures under a shared train/validation/test protocol.
- Keep model evaluation separate from the Gradio inference interface.
- Make predictions, confidence, model disagreement, and limitations visible to users.
- Record evaluation results and error/bias probes without overstating model quality.

## Features

- Five locally available models with side-by-side predictions.
- Single-comment analysis with class probabilities and a low-confidence warning.
- TF-IDF word-feature evidence or BiLSTM attention as an explanation hint.
- Batch analysis from pasted lines or `.txt`/`.csv` files, up to 200 comments, with downloadable CSV output.
- Model-comparison and results/analytics views.
- Warnings for Devanagari input, truncation, and high out-of-vocabulary rates.
- Saved evaluation tables, plots, and an exploratory community-word bias probe.

## Models implemented

The local training and inference code implements:

1. **TF-IDF + Logistic Regression** with word and character n-grams.
2. **BiLSTM with additive attention** (word level).
3. **TextCNN** (word level).
4. **CharCNN** (character level, implemented using the TextCNN architecture).
5. **Transformer encoder trained from scratch** (word level).

The Colab notebook also contains an experimental fine-tuning workflow for pretrained transformers (mBERT, XLM-R, MuRIL, and HingBERT). Those results are not present in the saved local evaluation and those models are not part of the local Gradio app.

## Technologies

- Python 3.13 (the version used for project validation)
- PyTorch, scikit-learn, NumPy, pandas, SciPy, and joblib
- Matplotlib for evaluation plots
- Gradio for the web interface
- Jupyter/Google Colab and Hugging Face Transformers for the optional pretrained-model notebook

The local dependencies are listed in [`requirements.txt`](requirements.txt). The notebook installs its additional Colab dependencies separately.

## Project structure

```text
app/
  app.py                         Gradio dashboard
data/
  raw/                           Original IIITH corpus (local only; not tracked)
  processed/                     Train/validation/test CSVs (local only; not tracked)
figures/                         Evaluation plots
models/                          Saved model checkpoints
notebooks/
  colab_transformer_finetune.ipynb
results/                         Metrics, predictions, and analysis outputs
src/
  data.py                        Loading, normalization, and data splits
  baseline.py                    TF-IDF baseline and feature ablations
  models.py                      Neural model architectures
  train_dl.py                    Neural model training and evaluation
  predict.py                     Model inference helpers
  service.py                     Shared backend for the demo
  save_tfidf.py                  Rebuilds the saved TF-IDF model
  make_report_assets.py          Evaluation tables and figures
  bias_probe.py                  Exploratory bias probe
  metrics.py                     Shared evaluation metrics
README.md
requirements.txt
```

## Installation

Python 3.13 was used for validation. Create and activate a virtual environment from the project root.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the Python executable in `.venv\Scripts\` directly or follow your organization's PowerShell policy; do not disable security controls globally.

### macOS/Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

From the project root, run:

```bash
python app/app.py
```

The Gradio interface is served at <http://127.0.0.1:7860>. The app loads the five saved models from `models/`; the TF-IDF model also reads its evaluation score from `results/`.

To recreate the TF-IDF model file from the existing splits and verify its predictions:

```bash
python src/save_tfidf.py
```

Neural checkpoints can be regenerated with `python src/train_dl.py`. This performs model training; it is not required when the supplied checkpoints are available.

For a command-line prediction using a neural model:

```bash
python src/predict.py "ye bakwas hai" --model textcnn
```

## Example

Input:

```text
yaar ye gana bahut mast hai, sunke maza aa gaya
```

Observed output from the saved local models:

```text
TF-IDF + LogReg: positive (87.7% confidence)
Model agreement: 4 of 5 predict positive
```

This is one inference example, not an evaluation result; predictions and confidence can vary by model.

## Evaluation results

The checked-in result files report performance on a held-out test split of 388 comments. Neural results are means and population standard deviations over seeds 42, 43, and 44; the TF-IDF result is deterministic.

| Model | Test macro-F1 | Accuracy |
|---|---:|---:|
| TF-IDF + Logistic Regression (word + character) | 0.682 | 0.698 |
| BiLSTM + attention | 0.591 ± 0.013 | 0.610 |
| TextCNN (word) | 0.579 ± 0.014 | 0.608 |
| CharCNN | 0.665 ± 0.029 | 0.681 |
| Transformer (from scratch) | 0.569 ± 0.014 | 0.590 |

The paired exact McNemar test against the TF-IDF baseline reports `p = 0.8957` for CharCNN; the other scratch neural models score lower in these saved experiments. The test set is small, so modest score differences should not be overinterpreted. Detailed metrics are in `results/baseline_results.json`, `results/dl_results.json`, and `results/comparison_table.csv`.

## Screenshots

No screenshots of the running application are currently included. Add genuine screenshots of the local Gradio interface here when available; the charts in `figures/` are evaluation plots, not application screenshots.

## Data and attribution

The project uses the IIITH Hindi-English code-mixed sentiment corpus described by Prabhu, Joshi, Shrivastava, and Varma, *Towards Sub-Word Level Compositions for Sentiment Analysis of Hindi-English Code Mixed Text*, COLING 2016 ([arXiv:1611.00472](https://arxiv.org/abs/1611.00472)). The source cited by the project is <https://github.com/DrImpossible/Sub-word-LSTM>.

The cited source repository does not list a dataset licence. A local scan found phone-number-shaped strings in corpus comments (no email-pattern matches); this is a privacy signal, not proof that each match is a phone number. The corpus, processed splits, prediction/error CSVs, and attention-example figure are therefore excluded from Git by `.gitignore` pending privacy review and confirmation of redistribution rights. They remain in the local workspace and are not deleted. Confirm the applicable rights with the dataset authors before publishing corpus content or derived files. Do not treat this repository as granting rights to the dataset.

## Limitations

- The experiment uses one small corpus and one fixed split; it may not generalize to other domains or current language use.
- Saved local evaluation describes the listed corpus only. The pretrained-transformer notebook has not contributed results to the table.
- Roman-script Hinglish is the training domain. Devanagari and unfamiliar spellings can produce unreliable predictions.
- The corpus contains class and community-word bias; `results/bias_probe.csv` is exploratory, not a complete fairness assessment.
- Labels can be subjective, sarcasm and slang are difficult, and the attention display is not a guaranteed explanation.
- Confidence values are not calibrated guarantees. Do not use the system for moderation, hiring, access, or decisions about individuals.
- Dataset redistribution rights are not established by the source repository; see [Data and attribution](#data-and-attribution).

## Future scope

- Obtain and document redistribution permission for the corpus before publishing data-derived files.
- Run the pretrained-transformer notebook and report results only after verifying its outputs.
- Evaluate with repeated stratified cross-validation and independently reviewed labels.
- Expand testing across domains, spelling variants, and fairness slices.

## Author

**[Your Name]** — replace this placeholder with your preferred name and profile link.
