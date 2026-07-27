"""Download the UCI Default of Credit Card Clients dataset to data/uci_default.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e23mon.data import download_uci  # noqa: E402

if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "data/uci_default.csv"
    print(f"saved {download_uci(dest)}")
