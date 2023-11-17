# hpodash

A collection of preprocessing scripts for getting existing pre-collected tabular HPO data into a common long `csv` format.
This is for example useful to quickly visualize learning curves for all configurations of a given task.

## Notes:

* Preprocessing is usually not very memory efficient, and you will need at least 32GB of RAM and the resulting `csv` files can grow quite large - this should be fine because if you want this long `csv` format you will run the preprocessing only once and then use the `csv` files.
* The original data sources and preprocessed `csv` files are not included in this repository, but the scripts to obtain them.
* Scripts should work fine with Python 3.10+.

## Data sources considered so far:

* fcnet
* lcbench
* pd1

## Other data sources not considered so far:

* TaskSet (https://github.com/google-research/google-research/blob/master/task_set/README.md)
* TabZilla? (https://github.com/naszilla/tabzilla)
* TabRepo (https://github.com/autogluon/tabrepo)
* Revisiting Deep Learning Models for Tabular Data (https://github.com/yandex-research/tabular-dl-revisiting-models)
* Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data (https://github.com/LeoGrin/tabular-benchmark)?
