# lcbench

## Steps:

1. Obtain `data_2k_lw.zip`

```
wget -O data_2k_lw.zip https://figshare.com/ndownloader/files/21188598
echo "26c8493006358a15ce12041c0bfc12a2 data_2k_lw.zip" | md5sum -c
```

2. Unzip:

```
unzip data_2k_lw.zip
```

3. Preprocess:

```python
python preprocess.py
```

Output is a `csv` labelled `data.csv` with columns that should be self-explanatory.

## Notes:

lcbench claims to contain data over 50 epochs but the actual data contains 52 epochs (0 to 51) and the meaning of the first and last epoch is not well documented.
Apparently the first (0) epoch is after initialization and no training being done, so the second (1) is actually the first (0) epoch.
The last (51) epoch often has the same performance metrics as the penultimate (50) therefore people usually drop it.
Therefore, the relevant epoch subset is considered `[1, ..., 50]`.
Moreover, the final metrics in `["log"]` do not always match the metrics in `["results"]` (maybe retraining on all train and validation data was performed but this is not documented).
Therefore, we always use the `"log"` data.

## Citation:

```
@article{lcbench,
  author = {Lucas Zimmer and Marius Lindauer and Frank Hutter},
  title = {Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2021},
  volume = {43},
  number = {9},
  pages = {3079--3090}
}
```
