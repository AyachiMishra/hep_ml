import argparse
import numpy as np
from hep_ml.reweight import GBReweighter


def run_reweight(source, target):
    try:
        Xs = np.load(source)
        Xt = np.load(target)
    except Exception as e:
        print(f"[ERROR] Failed to load files: {e}")
        return

    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)
    if Xt.ndim == 1:
        Xt = Xt.reshape(-1, 1)

    if Xs.shape[1] != Xt.shape[1]:
        print("[ERROR] Source and target must have same number of features")
        return

    print("[INFO] Training GBReweighter...")
    reweighter = GBReweighter()
    reweighter.fit(Xs, Xt)

    print("[INFO] Predicting weights...")
    weights = reweighter.predict_weights(Xs)

    print("[SUCCESS] First 10 weights:")
    print(weights[:10])


def main():
    parser = argparse.ArgumentParser(prog="hep-ml", description="hep-ml CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Reweight command
    rw = subparsers.add_parser("reweight", help="Run reweighting using GBReweighter")
    rw.add_argument("--source", required=True, help="Path to source .npy file")
    rw.add_argument("--target", required=True, help="Path to target .npy file")

    args = parser.parse_args()

    if args.command == "reweight":
        run_reweight(args.source, args.target)
    else:
        parser.print_help()
