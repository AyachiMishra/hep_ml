import argparse

def main():
    parser = argparse.ArgumentParser(prog="hep-ml")
    subparsers = parser.add_subparsers(dest="command")

    # Example: dataset info
    info = subparsers.add_parser("info")
    info.add_argument("file", help="Path to data file")

    args = parser.parse_args()

    if args.command == "info":
        print(f"Inspecting {args.file}")
