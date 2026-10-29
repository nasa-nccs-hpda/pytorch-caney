import pandas as pd
import matplotlib.pyplot as plt


def main():
    modis_vis_csv = "modis_vis_losses.csv"
    goes_vis_csv = "goes_vis_losses.csv"
    goes_ir_csv = "goes_ir_losses.csv"

    modis_vis_df = pd.read_csv(modis_vis_csv)
    goes_vis_df = pd.read_csv(goes_vis_csv)
    goes_ir_df = pd.read_csv(goes_ir_csv)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.hist(
        modis_vis_df["Losses"].values,
        bins=10,
        histtype="step",
        color="black",
        density=True,
        label="MODIS VIS",
    )
    ax.hist(
        goes_vis_df["Losses"].values,
        bins=50,
        histtype="step",
        color="blue",
        density=True,
        label="GOES VIS",
    )
    ax.hist(
        goes_ir_df["Losses"].values,
        bins=50,
        histtype="step",
        color="red",
        density=True,
        label="GOES IR",
    )

    ax.legend(loc="best")

    ax.set_ylabel("Density")
    ax.set_xlabel("L1 Loss from Pretrained MIM")

    fig.savefig(
        "giant_pretrained_MIM_loss_comparision.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    main()
