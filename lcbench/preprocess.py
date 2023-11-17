if __name__ == "__main__":
    import json

    import pandas as pd
    import tqdm

    n_epochs = 50

    with open("data_2k_lw.json") as f:
        data = json.load(f)

    df_data = []

    for task in tqdm.tqdm(data.keys()):
        tmp = data[task]
        for config in tmp.keys():
            tmp_config = tmp[config]
            time = tmp_config["log"]["time"][1:n_epochs + 1]
            time_increase = [
                time[i] - time[i - 1] if i > 0 else time[i] for i in range(0, n_epochs)
            ]
            val_accuracy = tmp_config["log"]["Train/val_accuracy"][1:n_epochs + 1]
            val_accuracy = [x / 100 for x in val_accuracy]
            val_cross_entropy = tmp_config["log"]["Train/val_cross_entropy"][1:n_epochs + 1]
            val_balanced_accuracy = tmp_config["log"]["Train/val_balanced_accuracy"][
                1:n_epochs + 1
            ]
            test_accuracy = tmp_config["log"]["Train/test_result"][1:n_epochs + 1]
            test_accuracy = [x / 100 for x in test_accuracy]
            test_cross_entropy = tmp_config["log"]["Train/test_cross_entropy"][1:n_epochs + 1]
            test_balanced_accuracy = tmp_config["log"]["Train/test_balanced_accuracy"][
                1:n_epochs + 1
            ]

            for i in range(0, n_epochs):
                res = {
                    "OpenML_task_id": tmp_config["results"]["OpenML_task_id"],
                    "epoch": i + 1,
                    "batch_size": tmp_config["config"]["batch_size"],
                    "learning_rate": tmp_config["config"]["learning_rate"],
                    "momentum": tmp_config["config"]["momentum"],
                    "weight_decay": tmp_config["config"]["weight_decay"],
                    "num_layers": tmp_config["config"]["num_layers"],
                    "max_units": tmp_config["config"]["max_units"],
                    "max_dropout": tmp_config["config"]["max_dropout"],
                    "time": time[i],
                    "time_increase": time_increase[i],
                    "val_accuracy": val_accuracy[i],
                    "val_cross_entropy": val_cross_entropy[i],
                    "val_balanced_accuracy": val_balanced_accuracy[i],
                    "test_accuracy": test_accuracy[i],
                    "test_cross_entropy": test_cross_entropy[i],
                    "test_balanced_accuracy": test_balanced_accuracy[i],
                }
                df_data.append(res)

    df = pd.DataFrame(df_data)
    df = df.astype(
        dtype={
            "OpenML_task_id": "int64",
            "epoch": "int64",
            "batch_size": "int64",
            "learning_rate": "float64",
            "momentum": "float64",
            "weight_decay": "float64",
            "num_layers": "int64",
            "max_units": "int64",
            "max_dropout": "float64",
            "time": "float64",
            "time_increase": "float64",
            "val_accuracy": "float64",
            "val_cross_entropy": "float64",
            "val_balanced_accuracy": "float64",
            "test_accuracy": "float64",
            "test_cross_entropy": "float64",
            "test_balanced_accuracy": "float64",
        }
    )

    df.to_csv("data.csv", index=False)
