# LiteASR
A lite framework focusing on ASR.

## Installation

- Clone the repo

  ```bash
  git clone https://github.com/Nazukixv/LiteASR.git
  ```

- Install Conda: https://docs.conda.io/en/latest/miniconda.html

- Create new Conda environment

  ```bash
  conda create -n liteasr python=3.7
  conda activate liteasr
  ```

- Install dependencices

  ```bash
  pip install -r requirements.txt
  ```

- Install PyTorch CUDA version

- Install LiteASR

  ```bash
  cd <liteasr-root>
  pip install -e .
  ```

## Data Preparation

LiteASR accepts Kaldi-style data sheet as input. Basically a typical data set should contain `feats.scp`, `text` and `utt2num_frames`. Both `feats.scp` and `utt2num_frames` can be generated by Kaldi.

Thus, used data set can be organized as below:

```
data/
├── train
│   ├── feats.scp
│   ├── text
│   └── utt2num_frames
├── valid
│   ├── feats.scp
│   ├── text
│   └── utt2num_frames
├── test-1
│   ├── feats.scp
│   ├── text
│   └── utt2num_frames
├── test-2
│   ├── feats.scp
│   ├── text
│   └── utt2num_frames
│   ...
└── vocab.txt
```

Then create a `my_task.yaml` under the `liteasr/config/task` as below:

```yaml
defaults:
  - asr

name: asr
vocab: /.../data/vocab.txt
train: /.../data/train
valid: /.../data/valid
test:
  - /.../data/test-1
  - /.../data/test-2
```

## Training

Run the following command

```bash
liteasr-train \
  task=my_task \
  model=my_U2 \
  criterion=my_hybrid_ctc \
  optimizer=my_noam \
  hydra.run.dir=<exp-dir>
```

Then LiteASR will start training process with the configuration automatically read from `liteasr/config/config.yaml`.

Since LiteASR utilizes [Hydra](https://github.com/facebookresearch/hydra) as configuration management, please refer to [Hydra Doc](https://hydra.cc/docs/1.1/intro/) for more detailed usage description.

## Inference

Run the following command

```bash
liteasr-infer --config-dir <exp-dir>/.hydra \
  hydra.run.dir=<exp-dir>
```
