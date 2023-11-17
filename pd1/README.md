# pd1

## Steps:

1. Obtain `pd1.tar.gz`
 
```
wget -O pd1.tar.gz http://storage.googleapis.com/gresearch/pint/pd1.tar.gz
echo "e04c789cff8d5ddc9f2301e2ac02d0e3 pd1.tar.gz" | md5sum -c
```

2. Uncompress:

```
tar -xvf pd1.tar.gz
cd pd1
gzip -d pd1_matched_phase1_results.jsonl.gz
gzip -d pd1_unmatched_phase1_results.jsonl.gz
cd ..
```

3. Preprocess:

```python
python preprocess.py
```

Output is a `csv` labelled `data.csv` with columns that should be self-explanatory.

## Notes:

pd1 contains data of many diverse tasks where not the machine learning algorithm was tuned but the first order optimizer used to train the model (with the ML algorithm's own hyperparameters held constant).
The actual task is then defined by the dataset, the machine learning algorithm and the batch size.
We use the `"phase1"` data (both `"matched"` and `"unmatched"`).
Note that this can result in some unique configs being present twice for a given task.
Note that not for all unique configs both validation and test performance over the given number of epochs is available.
Also, some configs diverged.
Such configs are excluded.
Note that the number of epochs and their resolution how they were logged differs between tasks.

## Citation:

```
@article{pd1,
  author = {Zi Wang and George E. Dahl and Kevin Swersky and Chansoo Lee and Zelda Mariet and Zachary Nado and Justin Gilmer and Jasper Snoek and Zoubin Ghahramani},
  title = {Pre-trained Gaussian Processes for Bayesian Optimization},
  journal = {arXiv preprint arXiv:2109.08215},
  year = {2021}
}
```
