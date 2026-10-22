from pathlib import Path
import numpy as np


def main():

    dirIn = Path("/usr/WS1/smalley5/downstream-3dclouds-subset")

    for f in dirIn.rglob("*.npz"):
        with np.load(f, allow_pickle=True) as archive:
            goes_chip, ec_data = archive["chip"], archive["data"].item()
        print(ec_data["Cloud_mask"].shape)


if __name__ == "__main__":
    main()
