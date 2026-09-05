# Dataset

The training dataset is intentionally not committed to this repository.

The default configuration expects:

`/content/drive/MyDrive/INKER_Confidence_Detector/datasets/confidence_statements1.json`

The JSON file must contain two top-level mappings:

- `confident`
- `unconfident`

Each mapping is organized by the same topic names. The current replication
uses the 27-topic confident/unconfident statement collection used by the
original Colab experiment.

Change `paths.project_dir` or `paths.dataset` in `configs/default.yaml` if
your file lives somewhere else.
