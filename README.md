# NLP — Word Sense Disambiguation (WSD)

A BERT-based system for **Word Sense Disambiguation**: given a sentence and a set of ambiguous target words, the model predicts the correct WordNet sense (<u>synset</u>) for each target word from a list of candidate senses.

The approach is to treat WSD as a **binary classification** problem over **context-gloss pairs**: for each `<context sentence, candidate sense>` pair, the model predicts whether the candidate gloss (<u>definition</u>) matches the sense of the target word in that context. At inference time, the candidate with the highest score is selected as the predicted sense.

## 📋 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Data](#-data)
- [Requirements](#-requirements)
- [Training](#-training)
- [Usage](#-usage)
- [References](#-references)

---

## 🔎 Overview

1. **Context-Gloss Pair Construction** — For every ambiguous word in a sentence, the dataset builder pairs the sentence with the WordNet gloss (definition) of each candidate sense, optionally applying *weak supervision* by wrapping the target word in quotes to highlight it.
2. **Encoding** — Each `<sentence, gloss>` pair is tokenized with a BERT tokenizer as a single sequence `"sentence [SEP] gloss"`.
3. **Classification** — A fine-tuned BERT encoder produces a `[CLS]` embedding, which is passed through a dropout layer and a linear classifier to output a single logit (`YES`/`NO` — does this gloss match this usage?).
4. **Sense Selection** — During inference, all candidate senses for a target word are scored, and the sense with the highest score is chosen as the prediction.
5. **Coarse ↔ Fine mapping** — A precomputed mapping (`data/map/coarse_fine_defs_map.json`) links coarse-grained sense inventories to fine-grained WordNet synsets and their glosses.

## 📁 Repository structure

```
├── code/
│   ├── train.py                        # Training entry point (PyTorch Lightning + Weights & Biases)
│   ├── configs/                        # Training/runtime configuration (device, epochs, batch size, etc.)
│   └── src/
│       ├── data.py                     # WSD_Dataset (context-gloss pair construction)
│       ├── model.py                    # WSD_Model: BERT encoder + classification head
│       ├── wordnet.py                  # Helpers for retrieving WordNet synset definitions and sense keys
│       └── utils.py
├── docker/
│   ├── app.py                          # Flask inference server (POST sentences, get back predicted senses)
│   ├── model.py
│   ├── evaluate.py                     # Client script: sends a test set to the running server and computes accuracy
│   ├── simple_test.py
│   └── src/
│       └── implementation.py           # Docker model implementation (loads checkpoint, runs inference)
├── data/
│   ├── coarse-grained/                 # Coarse-grained sense-annotated (JSON)
│   ├── fine-grained/                   # Fine-grained sense-annotated (JSON)
│   └── map/
│       └── coarse_fine_defs_map.json   # Coarse-to-fine sense/gloss mapping
├── logs/                               # Server stdout/stderr logs (populated by test.sh)
├── Dockerfile                          # Builds the inference server image
├── test.sh                             # Builds the Docker image, starts the server and runs evaluation
└── requirements.txt                    # Python dependencies
```

## 📜 Data

Each split (`train` / `val` / `test`, coarse- and fine-grained) is a JSON file mapping a sentence ID to an annotated sentence, for example:

```json
{
  "d000.s032": {
    "words": ["Choose", "203", "business", "executives", "."],
    "lemmas": ["choose", "203", "business", "executive", "."],
    "pos_tags": ["VERB", "NUM", "NOUN", "NOUN", "."],
    "instance_ids": { "0": "d000.s032.t000", "3": "d000.s032.t001" },
    "senses":      { "0": ["select.v.h.01"], "3": ["executive.n.h.01"] },
    "candidates":  { "0": ["select.v.h.01", "preferred.v.h.01", "chosen.v.h.01"], "3": ["executive.n.h.01"] }
  }
}
```

- `words` / `lemmas` / `pos_tags` — the tokenized sentence and its linguistic annotations.
- `instance_ids` — positions (by token index) of the ambiguous words to disambiguate.
- `candidates` — the list of possible senses for each ambiguous word.
- `senses` — the gold/correct sense(s) for each ambiguous word (used for training and evaluation).

`data/map/coarse_fine_defs_map.json` maps each coarse-grained sense key to its corresponding fine-grained WordNet synsets and glosses.

## 🛠️ Requirements

* Ubuntu distribution: either 20.04 or the current LTS (22.04) are perfectly fine.
* [Conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community.

### Setup Environment

To evaluate the final model it will be used Docker to remove any issue pertaining the code runnability. To run *test.sh*, we need to perform two additional steps:

* Install Docker
* Setup a client

`test.sh` essentially setups a server exposing the model through a REST API and then queries this server, evaluating it.

#### 1. Install Docker

```bash
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

> ⚠️ Unfortunately, for the latter command to have effect, you need to **reboot** your Ubuntu OS and re-login. **Do it** before proceeding.

#### 2. Setup Client

The model will be exposed through a REST server, in order to call it during the evaluation we need a client. The client is written in the evaluation script and it needs some dependencies to run: Use conda to create the environment for the client.<br>
<u>Nota</u>: Python 3.9+ is required. The Docker image uses `python:3.9-slim`.

```bash
conda create -n nlp-wsd python=3.9
conda activate nlp-wsd
pip install -r requirements.txt
```

## 🏋 Training

1. Place a pretrained BERT tokenizer/model at `model/bert_tokenizer/` and `model/bert_model/` (relative to the project root), and edit `code/configs/config.yaml` as needed (device, epochs, batch size, learning rate, weak supervision, etc.).
2. From the `code/` directory, run:

```bash
cd code
python train.py
```

Training uses **PyTorch Lightning**, logs metrics to **Weights & Biases** and saves model's checkpoint under `ckpt/`.

## 🚀 Usage

### Running the inference server (Docker)

The `docker/` folder contains a self-contained Flask service that loads a trained checkpoint and serves predictions over HTTP. <br>

*test.sh* is a simple bash script. To run it:

```bash
# Build and run the server, then evaluate it against a test file
conda activate nlp-wsd
bash test.sh data/coarse-grained/test_coarse_grained.json
```

> ⚠️ Actually, you can replace *data/coarse-grained/test_coarse_grained.json* to point to a different file, as far as the target file has the same format.

This script will:
1. Build the Docker image (`Dockerfile`), which installs dependencies and copies in the model artifacts and inference code.
2. Start the container, exposing the server on a port.
3. Run `docker/evaluate.py` against the given test file, computing WSD accuracy.
4. Stop the container and dump its logs to `logs/server.stdout` / `logs/server.stderr`.

> ⚠️ Expected model artifacts inside the `model/` directory (copied into the image by the Dockerfile):
- `model/bert_tokenizer/`, `model/bert_model/` — pretrained BERT tokenizer and weights
- `model/epoch_weak.ckpt` — the fine-tuned WSD checkpoint
- `model/coarse-fine_map.data` — the serialized coarse-to-fine sense mapping

## 📚 References
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. (2019). [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805).
- Luyao Huang, Chi Sun, Xipeng Qiu, and Xuanjing Huang. (2019). [GlossBERT: BERT for word sense disambiguation with gloss knowledge](https://arxiv.org/abs/1908.07245).
- George A. Miller. (1994). [WordNet: A lexical database for English](https://aclanthology.org/H92-1116/).
- Roberto Navigli and Simone Paolo Ponzetto. (2010). [BabelNet: Building a very large multilingual semantic network](https://aclanthology.org/P10-1023/).

## 👤 Author

**Antonio Gagliardi**  
Email: [gaglia.anto95@gmail.com](mailto:gaglia.anto95@gmail.com)