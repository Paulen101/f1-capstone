"""Train the pit-timing network on the race situations (.venv-train: PyTorch, CPU). Methodology step 6.

    .venv-train/Scripts/python.exe data_pipeline/train_pit_network.py
    .venv-train/Scripts/python.exe data_pipeline/train_pit_network.py --holdout-track Monza --name pit_network_noMonza

Input: integration/output/situations/situations.csv from data_pipeline/build_situations.py. Each row is a
race situation at a decision lap with the pit-now advantage in finishing position and race time (pit now
minus the best later lap; negative: pitting now is better).

The network (integration/pit_network.py) predicts both advantages from the situation and four tyre
features from the fitted race file. Plain mean squared error, so it learns the expected advantage given
the situation: each row's target is one random future, the average over many is the expectation.

Validation: Monte Carlo runs 160-199 of every driver-race are held out (other random futures of the same
races). --holdout-track also leaves out every row of one track, to see how the network does on a track it
has never seen. Output: integration/models/<name>.npz (numpy weights, read by the simulator) and
notes/evidence/pit_network/<name>.json (settings and validation results).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.pit_network import FEATURES, PitNetwork, feature_vector, pit_now, tyre_features  # noqa: E402
from integration.write_fitted_pars import parse_pars  # noqa: E402

DATA = ROOT / "integration/output/situations/situations.csv"
VAL_FIRST_RUN = 160
TIME_CLIP_S = 60.0     # a few race-time advantages are 100-190 s (lapped cars); clipped so they do not dominate
Y_SCALE = np.array([1.0, 10.0])   # places, 10 s: both targets of similar size during training


TRAIN_ERROR_SALT = 10 ** 6   # training draws use run + salt, so they never repeat an evaluation race's error


def add_tyre_features(rows: pd.DataFrame, tyre_error: str = None) -> pd.DataFrame:
    """Tyre features of every row, from each race file's degradation model of the driver.

    tyre_error (integration/tyre_uncertainty.LEVELS): the features the strategy would compute from a wrong
    tyre model, drawn per (track, Monte Carlo run) like the evaluation's --tyre-error. The error is a pace
    offset plus a wear factor per compound, so every feature is linear in it: the features are split once
    into pace and wear parts of the true model and the error of each run is applied to those."""
    if tyre_error in (None, "exact"):
        keys = ["Race", "Driver", "Compound", "StopCompound", "TyreAge", "LapsRemaining"]
        unique = rows[keys].drop_duplicates()
        pars = {race: parse_pars((ROOT / "integration/parameters" / race).read_bytes().decode("utf-8"))
                for race in unique.Race.unique()}
        values = [tyre_features(pars[r.Race]["tireset_pars"][r.Driver], r.Compound, r.StopCompound,
                                int(r.TyreAge), int(r.LapsRemaining)) for r in unique.itertuples()]
        unique = pd.concat([unique.reset_index(drop=True), pd.DataFrame(values)], axis=1)
        return rows.merge(unique, on=keys, how="left", validate="many_to_one")
    return _wrong_tyre_features(rows, tyre_error)


def _tyre_parts(tire_pars: dict, compound: str, stop: str, age: int, laps_remaining: int) -> dict:
    """tyre_features split into pace (k_0) and wear parts: W(c, a) = degradation(c, a) - k_0(c)."""
    from integration.pit_network import degradation
    remaining = max(int(laps_remaining), 1)
    k0c, k0s = tire_pars[compound]["k_0"], tire_pars[stop]["k_0"]
    wear = lambda c, a: degradation(tire_pars, c, a) - tire_pars[c]["k_0"]
    return {"K0c": k0c, "K0s": k0s, "Rem": remaining, "Wc": wear(compound, age), "Wc1": wear(compound, age + 1),
            "Ws0": wear(stop, 0), "WsRem": wear(stop, remaining),
            "SumWc": sum(wear(compound, age + a) for a in range(remaining)),
            "SumWs": sum(wear(stop, a) for a in range(remaining))}


def _wrong_tyre_features(rows: pd.DataFrame, level: str) -> pd.DataFrame:
    from integration.tyre_uncertainty import draw
    keys = ["Race", "Driver", "Compound", "StopCompound", "TyreAge", "LapsRemaining"]
    unique = rows[keys].drop_duplicates()
    pars = {race: parse_pars((ROOT / "integration/parameters" / race).read_bytes().decode("utf-8"))
            for race in unique.Race.unique()}
    parts = [_tyre_parts(pars[r.Race]["tireset_pars"][r.Driver], r.Compound, r.StopCompound,
                         int(r.TyreAge), int(r.LapsRemaining)) for r in unique.itertuples()]
    unique = pd.concat([unique.reset_index(drop=True), pd.DataFrame(parts)], axis=1)
    out = rows.merge(unique, on=keys, how="left", validate="many_to_one")
    errors = []
    for race, run in out[["Race", "Run"]].drop_duplicates().itertuples(index=False):
        track = race.split("_")[1]
        for compound, (offset, factor) in draw(pars[race]["vse_pars"]["param_dry_compounds"], track,
                                               int(run) + TRAIN_ERROR_SALT, level).items():
            errors.append({"Race": race, "Run": run, "C": compound, "Offset": offset, "Factor": factor})
    errors = pd.DataFrame(errors)
    cur = out[["Race", "Run", "Compound"]].merge(errors.rename(columns={"C": "Compound"}),
                                                 on=["Race", "Run", "Compound"], how="left")
    stp = out[["Race", "Run", "StopCompound"]].merge(errors.rename(columns={"C": "StopCompound"}),
                                                     on=["Race", "Run", "StopCompound"], how="left")
    oc, fc = cur.Offset.to_numpy(), cur.Factor.to_numpy()
    os_, fs = stp.Offset.to_numpy(), stp.Factor.to_numpy()
    k0c, k0s, rem = out.K0c + oc, out.K0s + os_, out.Rem
    out["TyreDelta_s"] = (k0c + fc * out.Wc) - (k0s + fs * out.Ws0)
    out["WearRate_s"] = fc * (out.Wc1 - out.Wc)
    out["StopWearRate_s"] = fs * (out.WsRem - out.Ws0) / rem
    out["StayCost_s"] = rem * (k0c - k0s) + fc * out.SumWc - fs * out.SumWs
    return out.drop(columns=["K0c", "K0s", "Rem", "Wc", "Wc1", "Ws0", "WsRem", "SumWc", "SumWs"])


def load(path: Path):
    rows = add_tyre_features(pd.read_csv(path))
    rows["Track"] = rows.Race.str.extract(r"pars_(\w+?)_\d{4}")[0]
    x = np.array([feature_vector(r) for r in rows.to_dict("records")], dtype=np.float64)
    y = rows[["AdvantagePosition", "AdvantageTime_s"]].to_numpy(dtype=np.float64)
    y[:, 1] = np.clip(y[:, 1], -TIME_CLIP_S, TIME_CLIP_S)
    return rows, x, y


def build_model(n_in: int, hidden: list):
    import torch.nn as nn
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def masked_mse(pred, target):
    """Mean squared error over the targets that exist (race time is missing for 13 % of rows)."""
    import torch
    mask = ~torch.isnan(target)
    diff = torch.where(mask, pred - torch.nan_to_num(target), torch.zeros_like(pred))
    return (diff ** 2).sum(0) / mask.sum(0).clamp(min=1)


def train(x_tr, y_tr, x_va, y_va, args):
    import torch
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    mean, std = x_tr.mean(0), x_tr.std(0)
    std[std < 1e-9] = 1.0
    device = resolve_device(args.device)
    print("training on %s" % (torch.cuda.get_device_name(0) if device == "cuda" else "CPU"), flush=True)
    to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    xt, yt = to((x_tr - mean) / std), to(y_tr / Y_SCALE)
    xv, yv = to((x_va - mean) / std), to(y_va / Y_SCALE)
    model = build_model(x_tr.shape[1], args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), args.batch):
            idx = order[i:i + args.batch]
            loss = masked_mse(model(xt[idx]), yt[idx]).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            tr = masked_mse(model(xt), yt).tolist()
            va = masked_mse(model(xv), yv).tolist()
        history.append({"Epoch": epoch + 1, "TrainLoss": [round(v, 5) for v in tr],
                        "ValLoss": [round(v, 5) for v in va]})
        print("epoch %2d  train %.4f %.4f  val %.4f %.4f  (%.0f s)"
              % (epoch + 1, tr[0], tr[1], va[0], va[1], time.perf_counter() - started), flush=True)
        if sum(va) < best - 1e-5:
            best, stale = sum(va), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu(), mean, std, history


def resolve_device(choice: str) -> str:
    """auto: the NVIDIA GPU when PyTorch can use it (CUDA build installed), otherwise the CPU."""
    import torch
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: this PyTorch build cannot use the GPU (install the CUDA 12.8 build)")
    return choice


def save(model, mean, std, path: Path):
    linears = [m for m in model if hasattr(m, "weight")]
    arrays = {"features": np.array(FEATURES), "x_mean": mean, "x_std": std, "y_scale": Y_SCALE,
              "n_layers": np.array(len(linears))}
    for i, layer in enumerate(linears):
        arrays["w%d" % i] = layer.weight.detach().double().numpy()
        arrays["b%d" % i] = layer.bias.detach().double().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def mse(pred, y):
    mask = ~np.isnan(y)
    return [float(np.mean((pred[mask[:, j], j] - y[mask[:, j], j]) ** 2)) for j in range(2)]


def evaluate(name: str, rows: pd.DataFrame, pred: np.ndarray, y: np.ndarray, train_rows: pd.DataFrame,
             y_train: np.ndarray) -> dict:
    """Error against simple references, and agreement of the pit decision with the run-averaged one.

    constant      the training mean of each target
    lookup        the training mean of the same (race, driver, lap) cell: knows the race and the lap but
                  not the situation in this run; only defined for races seen in training
    decision      per (race, driver, lap) cell, pit_now() on the mean predicted advantage against pit_now()
                  on the mean true advantage over the validation runs
    """
    result = {"Rows": int(len(rows)), "MSE": mse(pred, y)}
    constant = np.nanmean(y_train, axis=0)
    result["MSE_constant"] = mse(np.tile(constant, (len(y), 1)), y)
    cells = ["Race", "Driver", "Lap"]
    train_cells = pd.DataFrame(y_train, columns=["P", "T"]).assign(
        **{c: train_rows[c].to_numpy() for c in cells}).groupby(cells).mean()
    lookup = train_cells.reindex(pd.MultiIndex.from_frame(rows[cells])).to_numpy()
    if not np.isnan(lookup).all():
        known = ~np.isnan(lookup).any(1)
        result["MSE_lookup"] = mse(lookup[known], y[known])
        result["MSE_model_on_lookup_rows"] = mse(pred[known], y[known])
    frame = rows[cells].assign(PredP=pred[:, 0], PredT=pred[:, 1], TrueP=y[:, 0], TrueT=y[:, 1])
    cell = frame.groupby(cells).mean().dropna()
    model_pit = [pit_now(p, t) for p, t in zip(cell.PredP, cell.PredT)]
    true_pit = [pit_now(p, t) for p, t in zip(cell.TrueP, cell.TrueT)]
    result["DecisionCells"] = len(cell)
    result["DecisionAgreement"] = float(np.mean(np.array(model_pit) == np.array(true_pit)))
    result["PitShareModel"] = float(np.mean(model_pit))
    result["PitShareTrue"] = float(np.mean(true_pit))
    print("%s: rows %d  MSE pos %.3f time %.1f (constant %.3f %.1f)  decision agreement %.1f %%"
          % (name, len(rows), result["MSE"][0], result["MSE"][1], result["MSE_constant"][0],
             result["MSE_constant"][1], 100 * result["DecisionAgreement"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--name", default="pit_network_v1")
    parser.add_argument("--holdout-track", help="leave every row of this track out of training")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    rows, x, y = load(args.data)
    print("loaded %d rows, %d features (%.0f s)" % (len(rows), x.shape[1], time.perf_counter() - started))
    in_val = (rows.Run >= VAL_FIRST_RUN).to_numpy()
    on_holdout = (rows.Track == args.holdout_track).to_numpy() if args.holdout_track else np.zeros(len(rows), bool)
    train_mask = ~in_val & ~on_holdout
    val_mask = in_val & ~on_holdout

    model, mean, std, history = train(x[train_mask], y[train_mask], x[val_mask], y[val_mask], args)
    model_path = ROOT / "integration/models" / (args.name + ".npz")
    save(model, mean, std, model_path)

    # the numpy network the simulator uses must give the same predictions as the PyTorch one
    network = PitNetwork(model_path)
    import torch
    sample = x[:2000]
    with torch.no_grad():
        torch_pred = model(torch.tensor((sample - mean) / std, dtype=torch.float32)).double().numpy() * Y_SCALE
    numpy_gap = float(np.abs(network.predict(sample) - torch_pred).max())
    assert numpy_gap < 1e-3, numpy_gap

    pred = network.predict(x)
    report = {"Name": args.name, "Settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "Features": FEATURES, "TrainRows": int(train_mask.sum()), "History": history,
              "NumpyVsTorchMaxDiff": numpy_gap,
              "Validation": evaluate("validation runs", rows[val_mask], pred[val_mask], y[val_mask],
                                     rows[train_mask], y[train_mask]),
              "ValidationByTrack": {}}
    for track in sorted(rows.Track.unique()):
        mask = val_mask & (rows.Track == track).to_numpy()
        if mask.any():
            report["ValidationByTrack"][track] = evaluate("  " + track, rows[mask], pred[mask], y[mask],
                                                          rows[train_mask], y[train_mask])
    if args.holdout_track:
        report["HeldOutTrack"] = evaluate("held-out track " + args.holdout_track, rows[on_holdout],
                                          pred[on_holdout], y[on_holdout],
                                          rows[train_mask], y[train_mask])
    report["Seconds"] = round(time.perf_counter() - started)
    out = ROOT / "notes/evidence/pit_network" / (args.name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("saved %s and %s (%.0f s)" % (model_path.relative_to(ROOT), out.relative_to(ROOT), report["Seconds"]))


if __name__ == "__main__":
    main()
