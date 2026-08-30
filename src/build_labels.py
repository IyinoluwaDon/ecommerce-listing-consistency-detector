"""
build_labels.py

Constructs the binary "compliant / non-compliant" moderation task on top of the
Rakuten France multimodal product dataset.

Ground truth for real policy violations does not exist in any public e-commerce
dataset (for obvious reasons). Instead we build a defensible, well-documented
synthetic proxy: text-image MISMATCH detection.

Label 0 (compliant)     -> title/description and image genuinely correspond
Label 1 (non-compliant) -> the image has been swapped with one from a
                            semantically distant category, simulating a
                            mislabeled / fraudulent / recycled-image listing.

This makes fusion a hard requirement: a text-only or image-only model has no
way to know the pairing is broken. Only a joint model reasoning over both
modalities together can catch it.

Usage:
    python build_labels.py --input_dir ../data --output_dir ../data --swap_frac 0.25 --seed 42
"""
import argparse
import numpy as np
import pandas as pd


# Hand-grouped from inspecting sample designations per prdtypecode.
# Used only to pick "semantically distant" categories for swapping, so
# mismatches are learnable rather than adjacent/ambiguous (e.g. swapping
# two book categories would be a much harder and noisier mismatch signal).
CATEGORY_GROUPS = {
    "books_media": [10, 2280, 2403, 2705, 2522],
    "toys_games_figures": [40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300,
                            1301, 1302, 1320, 2462, 2905],
    "home_furniture_garden": [1560, 1920, 1940, 2060, 2220, 2582, 2583, 2585],
}
CODE_TO_GROUP = {code: g for g, codes in CATEGORY_GROUPS.items() for code in codes}


def build(input_dir: str, output_dir: str, swap_frac: float, seed: int):
    rng = np.random.default_rng(seed)

    X = pd.read_csv(f"{input_dir}/X_train_update.csv", index_col=0)
    Y = pd.read_csv(f"{input_dir}/Y_train_CVw08PX.csv", index_col=0)
    df = X.join(Y).reset_index(drop=True)
    df["group"] = df["prdtypecode"].map(CODE_TO_GROUP)

    assert df["group"].isna().sum() == 0, "Unmapped prdtypecode found — update CATEGORY_GROUPS."

    n = len(df)
    n_swap = int(n * swap_frac)
    swap_idx = rng.choice(n, size=n_swap, replace=False)

    df["label"] = 0  # 0 = compliant (genuine pair)
    df["orig_imageid"] = df["imageid"]
    df["orig_productid"] = df["productid"]

    # For each row selected for swapping, pick a donor row from a DIFFERENT group
    for idx in swap_idx:
        own_group = df.at[idx, "group"]
        donor_pool = df.index[df["group"] != own_group]
        donor_idx = rng.choice(donor_pool)
        df.at[idx, "imageid"] = df.at[donor_idx, "orig_imageid"]
        df.at[idx, "productid_for_image"] = df.at[donor_idx, "orig_productid"]
        df.at[idx, "label"] = 1  # 1 = non-compliant (mismatched pair)

    # For non-swapped rows, the image to load is just their own original image
    df["productid_for_image"] = df["productid_for_image"].fillna(df["orig_productid"])
    df["productid_for_image"] = df["productid_for_image"].astype(df["orig_productid"].dtype)

    print(f"Total rows: {n}")
    print(f"Mismatched (label=1): {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    print(f"Genuine   (label=0): {(df['label']==0).sum()} ({100*(df['label']==0).mean():.1f}%)")
    print()
    print("Class balance by original category (label=1 rate should be ~swap_frac across all):")
    print(df.groupby("group")["label"].mean())

    out_cols = ["designation", "description", "productid", "imageid",
                "orig_productid", "orig_imageid", "productid_for_image",
                "prdtypecode", "group", "label"]
    out_path = f"{output_dir}/train_with_mismatch_labels.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../data")
    parser.add_argument("--output_dir", default="../data")
    parser.add_argument("--swap_frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.input_dir, args.output_dir, args.swap_frac, args.seed)
