from __future__ import annotations

import click
import pandas as pd


@click.command()
@click.argument("filename")
@click.option("-b", "--branches", multiple=True, default=[])
def print_parquet(filename, branches):
    events = pd.read_parquet(filename)
    print(events)
    print("columns")
    print(events.columns.to_numpy())
    if len(branches) > 0:
        for b in branches:
            if b in events.columns:
                print(b, events[b])
            else:
                print(f"Branch {b} not found")


if __name__ == "__main__":
    print_parquet()
