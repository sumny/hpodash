if __name__ == "__main__":
    import gc
    import json

    import h5py
    import numpy as np
    import pandas as pd
    import tqdm

    n_epochs = 100
    n_repls = 4

    tasks = [
        "naval_propulsion",
        "parkinsons_telemonitoring",
        "protein_structure",
        "slice_localization",
    ]

    for task in tasks:
        with h5py.File(f"fcnet_tabular_benchmarks/fcnet_{task}_data.hdf5", "r") as file:
            configs = []
            final_test_error = []
            runtime_increase = []
            runtime = []
            valid_mse = []
            for config in tqdm.tqdm(file.keys()):
                tmp = json.loads(config)
                x = tmp.copy()
                x.update({"task": task})
                for repl in range(1, n_repls + 1):
                    x_repl = x.copy()
                    x_repl.update({"repl": repl})
                    for epoch in range(1, n_epochs + 1):
                        x_epoch = x_repl.copy()
                        x_epoch.update({"epoch": epoch})
                        configs.append(x_epoch)
                    final_test_error_tmp = [
                        file[config]["final_test_error"][repl - 1]
                    ] * n_epochs
                    final_test_error.extend(final_test_error_tmp)
                    runtime_increase_tmp = [
                        file[config]["runtime"][repl - 1] / n_epochs
                    ] * n_epochs
                    runtime_increase.extend(runtime_increase_tmp)
                    runtime_tmp = list(np.cumsum(runtime_increase_tmp))
                    runtime.extend(runtime_tmp)
                    valid_mse_tmp = list(file[config]["valid_mse"][repl - 1])
                    valid_mse.extend(valid_mse_tmp)

        gc.collect()
        df = pd.DataFrame(configs)
        del configs
        gc.collect()
        df["final_test_error"] = final_test_error
        del final_test_error
        gc.collect()
        df["runtime_increase"] = runtime_increase
        del runtime_increase
        gc.collect()
        df["runtime"] = runtime
        del runtime
        gc.collect()
        df["valid_mse"] = valid_mse
        del valid_mse
        gc.collect()

        df = df.astype(
            dtype={
                "activation_fn_1": "object",
                "activation_fn_2": "object",
                "batch_size": "int64",
                "dropout_1": "float64",
                "dropout_2": "float64",
                "init_lr": "float64",
                "lr_schedule": "object",
                "n_units_1": "int64",
                "n_units_2": "int64",
                "task": "object",
                "repl": "int64",
                "epoch": "int64",
                "final_test_error": "float64",
                "runtime_increase": "float64",
                "runtime": "float64",
                "valid_mse": "float64",
            }
        )

        df.to_csv(f"data_{task}.csv", index=False)
        del df
        gc.collect()
