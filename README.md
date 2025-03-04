# BiLSTM-CRF Character-CNN NER

## Prerequisites

Ensure you have Python 3.12.7 installed. To install all necessary dependencies, use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Files Overview

- `A2-data/` - Folder containing datasets for training, validation, and testing.
- `conlleval.py` - Script for evaluating F1-score, precision, and recall. [Based on: `https://github.com/sighsmile/conlleval/tree/master`]
- `constants.py` - Stores important constants such as special tokens.
- `data.py` - Handles dataset loading and preprocessing.
- `evaluate.py` - Provides evaluation functions and writes results to output files.
- `experiment.py` - Defines and executes different model training configurations.
- `main.py` - Main script for running different model configurations via the command line.
- `model.py` - Implements the BiLSTM-CRF model, including decoding, Char-CNN, and loss functions.
- `train.py` - Contains the training loop and optimization logic.

## Usage

To see available experiment options, run:

```bash
python main.py --help
```

Available model variants:

```bash
python main.py --model {bi_lstm_crf, bi_lstm_crf_char_cnn, bi_lstm_crf_svm_loss, bi_lstm_crf_softmax_margin_loss} --model_name my_experiment
```

Example command to run the default BiLSTM-CRF model:

```bash
python main.py --model bi_lstm_crf --model_name my_experiment
```

## Model Variants

| Model Variant                     | Description                                                   |
| --------------------------------- | ------------------------------------------------------------- |
| `bi_lstm_crf`                     | Standard BiLSTM-CRF model.                                    |
| `bi_lstm_crf_char_cnn`            | BiLSTM-CRF with a character-level CNN for feature extraction. |
| `bi_lstm_crf_svm_loss`            | BiLSTM-CRF with SVM loss function.                            |
| `bi_lstm_crf_softmax_margin_loss` | BiLSTM-CRF using softmax-margin loss.                         |

## Output Files

After running an experiment, output files will be generated using the specified model name:

```
{model_name}.{pred|pred.perl|report|time|pt|hp}
```

- `{model_name}.pred.gold` - Prediction results side by side with gold label.
- `{model_name}.pred.perl` - Predictions formatted for the Perl evaluation script.
- `{model_name}.time` - Training time per batch.
- `{model_name}.pt` - Saved model weights (for PyTorch).
- `{model_name}.hp` - Hyperparameters used in the experiment.

## Notes

- To use GPU training, ensure you have a CUDA-compatible GPU and PyTorch installed with CUDA support.
- If running on CPU, training will be slower but still functional.
