# Dataset

Place the NSL-KDD files here (not redistributed in this repo):

- `KDDTrain+.txt` (required for real training)
- `KDDTest+.txt` (optional: enables evaluation on the official test set)

Source: Canadian Institute for Cybersecurity, University of New Brunswick
(NSL-KDD dataset). Without these files, `python ml_nids.py` falls back to a
clearly labelled synthetic dataset that is for demos only.
