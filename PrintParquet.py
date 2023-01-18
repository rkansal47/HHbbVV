import pandas as pd
import click


@click.command()
@click.argument("filename")
@click.option("-b", "--branches", multiple=True, default=[])
def print_parquet(filename, branches):
    # At least one jet with Txbb > 0.8
    filters = [
        [
            ("('ak8FatJetParticleNetMD_Txbb', '0')", ">=", 0.8),
        ],
        [
            ("('ak8FatJetParticleNetMD_Txbb', '1')", ">=", 0.8),
        ],
    ]
    events = pd.read_parquet(filename, filters=filters)
    print(events.columns)
    if len(branches) > 0:
        for b in branches:
            if b in events.columns:
                print(b, events[b])
            else:
                print(f"Branch {b} not found")


if __name__ == "__main__":
    print_parquet()
