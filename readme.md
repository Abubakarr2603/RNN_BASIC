# Basic RNN Sentiment Analysis

A foundational binary sentiment classification project built with **TensorFlow / Keras** using the Functional API. This repository demonstrates end-to-end NLP text preprocessing, word embeddings with zero-padding masking, sequential memory modeling via `SimpleRNN`, training performance visualization, and single-function inference.

## Project Structure

```text
RNN_BASIC/
├── rnn_basic.ipynb     # Main Jupyter Notebook (preprocessing, training, evaluation & inference)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## How It Works

1. **Dataset Creation:** 30 labeled English sentences (`15` positive labeled `1`, `15` negative labeled `0`).
2. **Tokenization & Padding:** Maps words to integer IDs (`vocab_size=1000`, `<OOV>` token for unseen words) and applies post-padding (`maxlen=5`).
3. **Word Embeddings & Masking:** Converts integer IDs into 8-dimensional dense vectors (`emb_feature=8`) with `mask_zero=True`, automatically generating a `NotEqual` mask so trailing `0` padding tokens are skipped by the RNN.
4. **Sequential Memory (`SimpleRNN`):** Reads the embedded sequence timestep-by-timestep using `4` recurrent hidden units (`rnn_unit=4`) to capture word order and context.
5. **Binary Classification:** Passes the final hidden state into a single `Dense` neuron with `sigmoid` activation to output a sentiment probability between `0.0` and `1.0`.
6. **Inference Pipeline:** Includes a reusable `predict_sentiment()` helper that handles tokenization, padding, and formatted confidence output automatically.

## Model Architecture

Built with Keras's Functional API (`8,057` total trainable parameters):

| Layer (Type) | Output Shape | Parameters | Connected To | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| `input` (`InputLayer`) | `(None, 5)` | `0` | `-` | Accepts padded integer token sequences |
| `embedding` (`Embedding`) | `(None, 5, 8)` | `8,000` | `input[0][0]` | Maps token IDs to 8-D dense vectors (`mask_zero=True`) |
| `not_equal` (`NotEqual`) | `(None, 5)` | `0` | `input[0][0]` | Auto-generated boolean mask to ignore `0` padding |
| `rnn` (`SimpleRNN`) | `(None, 4)` | `52` | `embedding[0][0]`, `not_equal[0][0]` | Processes word sequence into a 4-D hidden state |
| `output` (`Dense`) | `(None, 1)` | `5` | `rnn[0][0]` | Sigmoid classifier outputting probability (`0.0` to `1.0`) |

## Training & Results

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Training Configuration:** `25` epochs, `batch_size=8`
- **Final Training Accuracy:** `100.00%` (`loss: 0.4072`)
- **Visualization:** Plots epoch-by-epoch accuracy and loss curves using `matplotlib`

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abubakarr2603/RNN_BASIC.git
   cd RNN_BASIC
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open `rnn_basic.ipynb` in Jupyter Notebook, VS Code, or Google Colab and run all cells. You can test predictions on any new sentence or list of sentences using the built-in `predict_sentiment()` function:

```python
predict_sentiment("I love her")
predict_sentiment(["Great service and friendly staff", "Total waste of money"])
```

### Sample Output

```text
'I love her'  -->  Positive (1)  (confidence: 0.5651)
'Great service and friendly staff'  -->  Positive (1)  (confidence: 0.6242)
'Total waste of money'  -->  Negative (0)  (confidence: 0.3033)
```

## Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Jupyter Notebook**