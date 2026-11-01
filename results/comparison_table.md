| Model | Macro-F1 | Accuracy | F1 neg | F1 neu | F1 pos | Params | Train s/run | HW |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg (word+char) | 0.682 (deterministic) | 0.698 | 0.644 | 0.744 | 0.657 | 146,235 | - | CPU |
| BiLSTM + attention | 0.591 ± 0.013 | 0.610 | 0.515 | 0.663 | 0.595 | 344,319 | 25 | CPU |
| TextCNN (word) | 0.579 ± 0.014 | 0.608 | 0.488 | 0.682 | 0.567 | 423,551 | 25 | CPU |
| CharCNN | 0.665 ± 0.029 | 0.681 | 0.610 | 0.717 | 0.668 | 59,667 | 43 | CPU |
| Transformer (from scratch) | 0.569 ± 0.014 | 0.590 | 0.473 | 0.644 | 0.591 | 188,867 | 33 | CPU |
