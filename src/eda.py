"""EDA: writes reports/eda_summary.txt and several figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .data import load_data


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    lines = [
        f"Rows: {len(df)}   Columns: {df.shape[1]}",
        "\nMissing values per column:\n" + df.isna().sum().to_string(),
        "\nTarget balance (proportion):\n"
        + df[config.TARGET].value_counts(normalize=True).round(3).to_string(),
        "\nSex distribution (1=male, 0=female):\n"
        + df["sex"].value_counts(normalize=True).round(3).to_string(),
        "\nAge summary:\n" + df["age"].describe().round(1).to_string(),
        "\nNumeric feature summary:\n"
        + df[config.NUMERIC].describe().round(2).to_string(),
    ]
    text = "\n".join(lines)
    (config.REPORT_DIR / "eda_summary.txt").write_text(text)
    print(text)

    # 1. target balance
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = df[config.TARGET].value_counts().sort_index()
    ax.bar(["No disease (0)", "Disease (1)"], counts.values, color=["#1f77b4", "#d62728"])
    ax.set_title("Target balance")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_target_balance.png", dpi=120)
    plt.close(fig)

    # 2. missing values
    fig, ax = plt.subplots(figsize=(6, 3))
    miss = df.isna().sum()
    ax.bar(miss.index, miss.values)
    ax.set_title("Missing values per column")
    plt.setp(ax.get_xticklabels(), rotation=60)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_missing.png", dpi=120)
    plt.close(fig)

    # 3. numeric histograms split by target
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, config.NUMERIC):
        for label, grp in df.groupby(config.TARGET):
            ax.hist(grp[col].dropna(), bins=20, alpha=0.6, label=f"target={label}")
        ax.set_title(col)
        ax.legend()
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_numeric_by_target.png", dpi=120)
    plt.close(fig)

    # 4. positive rate per categorical level
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for ax, col in zip(axes, config.CATEGORICAL):
        rate = df.groupby(col)[config.TARGET].mean()
        ax.bar(rate.index.astype(str), rate.values)
        ax.set_ylim(0, 1)
        ax.set_title(f"Disease rate by {col}")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_categorical_rates.png", dpi=120)
    plt.close(fig)

    # 5. correlation heatmap (numeric + target)
    cols = config.NUMERIC + [config.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60)
    ax.set_yticklabels(cols)
    fig.colorbar(im)
    ax.set_title("Correlation")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "eda_correlation.png", dpi=120)
    plt.close(fig)

    print("\nEDA figures saved to", config.FIG_DIR)


if __name__ == "__main__":
    main()
