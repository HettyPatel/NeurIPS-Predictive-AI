"""Train the prior on all public benchmarks and build the Codabench zip.  ->  submission.zip

Run from the repo root (after features.py):  python train_and_zip.py
Then check it with the organizers' script:   python <paiec_baseline>/check_submission_zip.py submission.zip
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
sys.path.insert(0, str(SUB))
from model import Model                     # noqa: E402
from evaluate import usable_benchmarks      # noqa: E402

FILES = ["model.py", "weights.npz", "meta.json", "models.txt", "requirements.txt",
         "submission_config.json"]          # the config holds the API key; it is gitignored, never commit it

if __name__ == "__main__":
    m = Model().fit(usable_benchmarks())     # all 160 benchmarks: the final model trains on everything we have
    m.save(SUB)
    (SUB / "models.txt").write_text("sentence-transformers/all-MiniLM-L6-v2\n")
    (SUB / "requirements.txt").write_text("sentence-transformers\ntorch\nnumpy\nanthropic\nhuggingface_hub\n")
    out = ROOT / "submission.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(SUB / name, name)        # flat: files at the zip root, as the platform expects
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB): {FILES}")
