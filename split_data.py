from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    base_dir = Path(__file__).resolve().parent
    df = pd.read_csv("HR_Dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates() 

    train, remaining = train_test_split(
        df, test_size=0.30, stratify=df["left"], random_state=42
    )

    val, test = train_test_split(
        remaining, test_size=0.50, stratify=remaining["left"], random_state=42
    )

    output_dir = base_dir / "data"
    output_dir.mkdir(exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        data.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(data)} rows | left rate: {data['left'].mean():.2%}")


if __name__ == "__main__":
    main()