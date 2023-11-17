if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import tqdm

    def explode_arrays(row):
        n = row["n_epochs"]
        if row["test_ce_loss"] is None or row["test_error"] is None:
            return None
        else:
            return pd.DataFrame(
                {
                    "task": row["task"],
                    "lr_initial_value": row["lr_initial_value"],
                    "lr_power": row["lr_power"],
                    "lr_decay_steps_factor": row["lr_decay_steps_factor"],
                    "one_minus_momentum": row["one_minus_momentum"],
                    "valid_ce_loss": pd.Series(row["valid_ce_loss"][:n]),
                    "valid_error": pd.Series(row["valid_error"][:n]),
                    "test_ce_loss": pd.Series(row["test_ce_loss"][:n]),
                    "test_error": pd.Series(row["test_error"][:n]),
                    "epoch": pd.Series(row["epoch"][:n]),
                    "eval_time": pd.Series(row["eval_time"][:n]),
                    "eval_time_increase": pd.Series(row["eval_time_increase"][:n]),
                }
            )

    files = [
        "pd1/pd1_matched_phase1_results.jsonl",
        "pd1/pd1_unmatched_phase1_results.jsonl",
    ]
    dfs_to_concat = []

    for file in tqdm.tqdm(files):
        with open(file, "r") as fin:
            tmp = pd.read_json(fin, orient="records", lines=True)

        tmp = tmp[tmp["status"] == "done"]
        tmp.reset_index(drop=True, inplace=True)

        activation_fn = [
            tmp["hps.activation_fn"].values[i] + "_"
            if tmp["hps.activation_fn"].values[i] is not None
            else ""
            for i in range(len(tmp))
        ]
        tmp["activation_fn"] = activation_fn
        tmp["task"] = [
            tmp["dataset"][i]
            + "_"
            + tmp["model"][i]
            + "_"
            + tmp["activation_fn"][i]
            + str(tmp["hps.batch_size"][i])
            for i in range(len(tmp))
        ]
        tmp["eval_time_increase"] = tmp["eval_time"]
        tmp["eval_time"] = tmp["eval_time"].apply(
            lambda x: None if x is None else np.cumsum(x).tolist()
        )
        tmp["hps.opt_hparams.momentum"] = tmp["hps.opt_hparams.momentum"].map(
            lambda x: 1 - x
        )
        column_renaming = {
            "task": "task",
            "hps.lr_hparams.initial_value": "lr_initial_value",
            "hps.lr_hparams.power": "lr_power",
            "hps.lr_hparams.decay_steps_factor": "lr_decay_steps_factor",
            "hps.opt_hparams.momentum": "one_minus_momentum",
            "valid/ce_loss": "valid_ce_loss",
            "valid/error_rate": "valid_error",
            "test/ce_loss": "test_ce_loss",
            "test/error_rate": "test_error",
            "epoch": "epoch",
            "eval_time": "eval_time",
            "eval_time_increase": "eval_time_increase",
        }

        tmp = tmp[column_renaming.keys()]
        tmp.columns = column_renaming.values()
        tmp["n_epochs"] = [len(tmp["epoch"][i]) for i in range(len(tmp))]

        n_points = sum(
            tmp[
                [
                    tmp["test_ce_loss"][i] is not None
                    and tmp["test_error"][i] is not None
                    for i in range(len(tmp))
                ]
            ]["n_epochs"]
        )

        tmp_exploded = pd.concat(
            [explode_arrays(row) for _, row in tmp.iterrows()]
        ).reset_index(drop=True)

        if n_points != len(tmp_exploded):
            raise ValueError(
                "Exploded long dataframe does not contain the expected number of rows"
            )

        tmp_exploded = tmp_exploded.astype(
            dtype={
                "task": "object",
                "lr_initial_value": "float64",
                "lr_power": "float64",
                "lr_decay_steps_factor": "float64",
                "one_minus_momentum": "float64",
                "valid_ce_loss": "float64",
                "valid_error": "float64",
                "test_ce_loss": "float64",
                "test_error": "float64",
                "epoch": "float64",
                "eval_time": "float64",
                "eval_time_increase": "float64",
            }
        )

        dfs_to_concat.append(tmp_exploded)

    df = pd.concat(dfs_to_concat).reset_index(drop=True)
    df.to_csv("data.csv", index=False)
