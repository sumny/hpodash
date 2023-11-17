# fcnet

## Steps:

1. Obtain `fcnet_tabular_benchmarks.tar.gz`

```
wget -O fcnet_tabular_benchmarks.tar.gz http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
echo "4fcb8e1d5de1552ebab923888014b848 fcnet_tabular_benchmarks.tar.gz" | md5sum -c
```

2. Uncompress:
```
tar -xvf fcnet_tabular_benchmarks.tar.gz
```

3. Preprocess:

```python
python preprocess.py
```

Output is a `csv` labelled `data_{task}.csv` for each of the four tasks with columns that should be self-explanatory.

## Notes:

fcnet contains data over 100 epochs.
Note that each config was always evaluated on four different seeds (`repl`).
The runtime was only measured for the whole training process over the 100 epochs.
Therefore, we assume a linear runtime increase `(runtime) / 100` per epoch.
Test performance was only calculated after the training over the 100 epochs and therefore there are no test performance curves over the epochs available.
The `final_test_error` is therefore always the same for each config over the 100 epochs.

## Citation:

```
@article{fcnet,
  author = {Aaron Klein and Frank Hutter},
  title = {Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization},
  journal = {arXiv preprint arXiv:1905.04970},
  year = {2019}
}
```
