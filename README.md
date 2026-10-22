# Non-episodic Variational Weight Imprinting for Few-Shot Learning

Code for the paper **"Non-episodic Variational Weight Imprinting for Few-Shot Learning"**, Padma V and Bhagvati C, SN Computer Science, 2023.
[DOI: 10.1007/s42979-023-01732-1](https://doi.org/10.1007/s42979-023-01732-1)

This repository contains the code used for the experiments on the
**Omniglot** and **IndicScript** datasets.

## Repository structure

```
.
├── datasets/
│   └── OneShot_Dataset.py   # dataset loading + few-shot task generation
├── models/
│   └── disent_auto_encoder.py  # DE_AutoEncoder: content/style auto-encoder + cosine classifier
├── one_shot_de_ae_train.py  # training script
├── oneshot_de_ae_test_episodic.py  # episodic few-shot evaluation
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/Padmavn/FSL-NE-VWIC.git
cd FSL-NE-VWIC
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.8.3, PyTorch 1.13.1, torchvision 0.14.1.


> **Note:** class names are parsed from file paths using `/` as the separator, so
> run the code on Linux / macOS (or WSL on Windows).

## Datasets

`datasets/OneShot_Dataset.py` is a single module shared across several of my
projects, so it contains classes for many datasets. **Only these two are used
in this paper:**

| Dataset | Class | Source |
|---|---|---|
| Omniglot | `OmniglotDataset` | [omniglot](https://github.com/floodsung/LearningToCompare_FSL/blob/master/datas/omniglot_28x28.zip) |
| IndicScript | `IndicSixPrintedDataset` | [IndicSixPrinted](https://scis.uohyd.ac.in/~chakcs/datasets/UHIndicPCwS.html) |

The other dataset classes in the file are not needed to reproduce this paper.

### Expected folder structure

Both datasets are read as `root / script(alphabet) / class(character) / image files`:

```
Omniglot/                      IndicScript/
├── Alphabet_1/                ├── Script_1/
│   ├── character01/           │   ├── class_001/
│   │   ├── img1.png           │   │   ├── img1.png
│   │   └── ...                │   │   └── ...
│   └── character02/           │   └── class_002/
└── Alphabet_2/                └── Script_2/
```

* Omniglot images are converted to grayscale and resized to 28x28. The link provided here contains the resized images
* IndicScript images are used at 50x50 (set by the transform in the training/testing scripts).
* The class folders are sorted, shuffled with a fixed seed (`random.seed(1)`) and split
  into meta-train / meta-test classes (function defaults: first 1200 Omniglot classes for training,
  first 800 for IndicScript; the training script uses 1200 for Omniglot and 800 for IndicScript).
  The split is therefore identical on every run.
* No split files (`.txt`) are needed for this paper: the train/test class split is generated
  from the seed above (`is_zsl=0` for IndicScript). The class-list files used for the zero-shot
  setting (`is_zsl=1`) belong to a different project and are not part of this repository.


## Training

`one_shot_de_ae_train.py` trains the disentangled auto-encoder (`models/disent_auto_encoder.py`)
on the training classes and saves checkpoints to `./saved_models/`. Omniglot command used for the
results in the paper:

```bash
python one_shot_de_ae_train.py -df /path/to/omniglot_resized -ds omniglot \
       -lr 0.0001 -ne 25 -bs 80 -cs 512 -lv 1000 -bvs 90 -bvc 1 \
       -nic 1 128 256 512 -noc 128 256 512 1024
```

IndicScript command (800 training classes):

```bash
python one_shot_de_ae_train.py -df /path/to/IndicSixPrinted -ds IndicSixPrinted \
       -lr 0.0001 -ne 25 -bs 50 -cs 512 -lv 1000 -bvs 10 -bvc 90 \
       -nic 1 128 256 512 -noc 128 256 512 1024
```

The script sets the input size (28 for Omniglot, 50 for IndicScript) and the first input channel
(1 for Omniglot, 3 for IndicScript) itself, so `-is` and the first value of `-nic` are overridden.

| Option | Meaning |
|---|---|
| `-df` | dataset folder (required) |
| `-ds` | `omniglot` or `IndicSixPrinted` |
| `-lr`, `-ne`, `-bs` | learning rate, number of epochs, batch size |
| `-cs` | code size; a single value (512) is used for both content and style, as in the paper |
| `-nic`, `-noc` | input / output channels of the convolutional layers |
| `-bvs`, `-bvc` | beta for the style / content KL terms |
| `-lv` | lambda: the Beta-VAE loss terms are multiplied by (1/lambda) |
| `-ss` | learning-rate scheduler step in epochs (default: no decay) |
| `-sdf` | state-dict file to start from |


Notes for reproducing the results:

* The model is built with `is_max_pool_after_last_conv=1` and `is_same_style_dim=1`; the test
  script must use the same settings.
* The reconstruction loss is summed over pixels and averaged over the batch, while the KL terms
  are averaged over batch and latent dimensions. `-bvs`, `-bvc` and `-lv` are relative to this scaling.
* The best checkpoint is chosen by validation accuracy (`*_maxacc.pth`). For Omniglot the
  validation set consists of the last 2 of the 20 images of every character and includes the
  augmented copies (rotations and affine); for IndicScript it is the last 5 images of every class
  in sorted file order.
* No random seed is set. Weight initialisation, dropout, the sampling noise
  of the VAE, data shuffling and the random augmentations therefore differ between runs, so
  results vary slightly from run to run. The class split itself is fixed (see above).
* Run on Linux or macOS (or Colab). `torchsummary` is optional and only used to print the model summary.

## Testing (episodic few-shot evaluation)

`oneshot_de_ae_test_episodic.py` loads a trained checkpoint and evaluates it on N-way K-shot episodes.
For each episode the K support images of every sampled test class are encoded, the normalised mean
content feature becomes the weight vector of a new cosine-similarity classifier, and the query images
are classified. It prints the mean accuracy with a 95% confidence interval over the test episodes.

```bash
python oneshot_de_ae_test_episodic.py -df /path/to/omniglot_resized -ds omniglot -mc 1 -cs 512 \
       -nic 1 128 256 512 -noc 128 256 512 1024 -sdf ./saved_models/<trained model>.pth \
       -nct 5 -spc 1 -oe 1 -te 1000   # example: 5-way 1-shot
```

* `-mc 1`: classical FSL (episodes contain novel classes only). `-mc 2`: generalized FSL (episodes contain
  both base and novel classes). Both are reported in the paper. `-mc 0` (validation images of the training
  classes) is also available but not used in the paper.
* `-nct` = N (classes per episode); if omitted, all test classes of the dataset are used
  (Omniglot: 423, IndicScript: 441).
* `-spc` = K (support images per class). `-qpc` = query images per class; if omitted, all remaining
  images of the class (excluding the support images) are used.
* `-oe` = outer repetitions (1 in the paper), `-te` = episodes per repetition: 1000 for N < 50 and
  100 for larger N.
* The architecture options (`-cs`, `-nic`, `-noc`, `-is`) must match the training run; `-cs` is the
  content code size used in training (512).
* The random test episodes are fixed by setting the random seed value as 1(the dataset code seeds Python’s random generator with 1), so the same checkpoint gives identical  accuracies on every run.

## Citation

```bibtex
@article{padma2023nonepisodic,
  title   = {Non-episodic Variational Weight Imprinting for Few-Shot Learning},
  author  = {Padma, V. and Bhagvati, C.},
  journal = {SN Computer Science},
  year    = {2023},
  doi     = {10.1007/s42979-023-01732-1}
}
```

## Acknowledgements

The task-generation code (`DatasetTask`, `FewShotDataset`) is adapted from
[floodsung/LearningToCompare_FSL](https://github.com/floodsung/LearningToCompare_FSL)
(Copyright (c) 2018 Flood Sung, MIT License), which is based on
[katerakelly/pytorch-maml](https://github.com/katerakelly/pytorch-maml).
The dataset classes and the remaining code are my own work.

The model architecture (`models/disent_auto_encoder.py`) is based on the master's thesis
*Few-shot Classification by Learning Disentangled Representations*
(Emiel Hoogeboom, University of Amsterdam, 2017). No code was publicly available for that work; this repository contains an
independent implementation that uses a Beta-VAE and a cosine-similarity classifier with a
different loss, in place of the thesis's VAE and episodic classifier.

The cosine-similarity classifier with normalised weights and a trainable scale follows
H. Qi, M. Brown and D. Lowe, *Low-Shot Learning with Imprinted Weights*, CVPR 2018. The in-place
weight normalisation in the classifier is adapted from the `weight_norm` method of
[YU1ut/imprinted-weights](https://github.com/YU1ut/imprinted-weights), an unofficial PyTorch
implementation of that paper.

## AI assistance disclosure

The models, algorithms and experiments are the work of the authors. Claude (Anthropic) assisted with
preparing this repository for publication:

* Writing the README, `requirements.txt` and the module docstrings/headers of all Python files
  (`datasets/OneShot_Dataset.py`, `models/disent_auto_encoder.py`, `one_shot_de_ae_train.py`,
  `oneshot_de_ae_test_episodic.py`), and removing personal paths and dated notes from comments;
* Reviewing the codebase to suggest argument validation, clean up hard-coded paths. All suggestions
  were reviewed and integrated by the authors and do not alter the core algorithms, 
  training routine, or evaluation protocol.

## License

Released under the MIT License - see [LICENSE](LICENSE). This includes the
original copyright notice of Flood Sung for the adapted code.
