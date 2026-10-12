"""python -m deciax.serve"""

import argparse

from deciax.serve.app import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m deciax.serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    serve(args.host, args.port, tuple(args.models or ["laya"]))


if __name__ == "__main__":
    main()
