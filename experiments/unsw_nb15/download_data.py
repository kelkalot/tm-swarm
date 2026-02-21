#!/usr/bin/env python
"""Download UNSW-NB15 training and testing CSV files."""
import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Direct download URLs (UNSW Research mirrors)
FILES = {
    "UNSW_NB15_training-set.csv": "https://research.unsw.edu.au/projects/unsw-nb15-dataset/UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv": "https://research.unsw.edu.au/projects/unsw-nb15-dataset/UNSW_NB15_testing-set.csv",
}

# Fallback: try nids-datasets package
def try_nids_package():
    try:
        from nids_datasets import data
        print("Using nids-datasets package...")
        df = data.read(dataset='UNSW-NB15', subset='Flow')
        train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
        df.to_csv(train_path, index=False)
        print(f"  Saved: {train_path} ({len(df)} rows)")
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  nids-datasets failed: {e}")
        return False


def download_file(url, dest):
    """Download with progress."""
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"  Saved: {dest} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in FILES if os.path.exists(os.path.join(DATA_DIR, f))]
    if len(existing) == len(FILES):
        print("All files already downloaded.")
        for f in FILES:
            size_mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1024 / 1024
            print(f"  {f}: {size_mb:.1f} MB")
        return

    # Try direct download first
    print("Downloading UNSW-NB15 dataset...")
    all_ok = True
    for fname, url in FILES.items():
        dest = os.path.join(DATA_DIR, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {fname}")
            continue
        if not download_file(url, dest):
            all_ok = False

    if all_ok:
        print("\nDone! All files downloaded.")
        return

    # Fallback: try nids-datasets
    print("\nDirect download failed. Trying nids-datasets package...")
    if try_nids_package():
        print("Done via nids-datasets.")
        return

    print("\nERROR: Could not download dataset.")
    print("Please download manually from: https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print(f"Place CSV files in: {DATA_DIR}")
    sys.exit(1)


if __name__ == "__main__":
    main()
