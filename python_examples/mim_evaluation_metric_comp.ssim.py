import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

FILES = {
    "MODIS Vis": "full_ssim_per_image.csv",
    "GOES Vis": "full_ssim_per_image.goes_vis.csv",
    "GOES IR": "full_ssim_per_image.goes_ir.csv",
}


def main():
    # Long format: one row per (source, image, channel)
    frames = []
    for label, path in FILES.items():
        df = pd.read_csv(path, index_col="image")
        long_df = df.reset_index().melt(
            id_vars="image", var_name="channel", value_name="value"
        )
        long_df["source"] = label
        frames.append(long_df)
    data = pd.concat(frames, ignore_index=True)
    data["type"] = data["channel"].str.split().str[0]  # "refl" or "BT"

    # Reflectance (%) and brightness temperature (K) have different units,
    # so plot them on separate axes, with the files side by side in each channel
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    panels = [
        (axes[0], "refl", "% reflectance", "Reflective channels", "RMSE"),
        (axes[1], "BT", "K", "Emissive channels", "RMSE"),
    ]
    for ax, kind, units, title, metric in panels:
        sns.boxplot(
            data=data[data["type"] == kind], x="channel", y="value", hue="source", ax=ax
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel(f"{metric}")
        ax.set_title(title)
        ax.legend(title=None)

    fig.savefig(
        "evaluation_metrics/full_ssim_boxplots.png", dpi=200, bbox_inches="tight"
    )
    plt.show()

    return
    full_rmse_modis_csv = "full_rmse_per_image.csv"
    full_ssim_modis_csv = "full_ssim_per_image.csv"
    # One row per image, one column per channel (e.g. "refl 0", "BT 5")
    modis_rmse_df = pd.read_csv(full_rmse_modis_csv, index_col="image")

    modis_ssim_df = pd.read_csv(full_ssim_modis_csv, index_col="image")

    # Reflectance (%) and brightness temperature (K) have different units,
    # so plot them on separate axes
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    sns.boxplot(data=modis_rmse_df.filter(like="refl"), ax=axes[0, 0], color="blue")
    axes[0, 0].set_xlabel("Channel")
    axes[0, 0].set_ylabel("Full-image RMSE (% reflectance)")
    axes[0, 0].set_title("Reflective channels")

    sns.boxplot(data=modis_rmse_df.filter(like="BT"), ax=axes[0, 1], color="blue")
    axes[0, 1].set_xlabel("Channel")
    axes[0, 1].set_ylabel("Full-image RMSE (K)")
    axes[0, 1].set_title("Emissive channels")

    sns.boxplot(data=modis_ssim_df.filter(like="refl"), ax=axes[1, 0], color="blue")
    axes[1, 0].set_xlabel("Channel")
    axes[1, 0].set_ylabel("Full-image SSIM")
    axes[1, 0].set_title("Reflective channels")

    sns.boxplot(data=modis_ssim_df.filter(like="BT"), ax=axes[1, 1], color="blue")
    axes[1, 1].set_xlabel("Channel")
    axes[1, 1].set_ylabel("Full-image SSIM")
    axes[1, 1].set_title("Emissive channels")

    fig.suptitle(f"Full-image RMSE per image ({len(modis_rmse_df)} images)")
    # fig.tight_layout()
    # fig.savefig("full_rmse_boxplots.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
