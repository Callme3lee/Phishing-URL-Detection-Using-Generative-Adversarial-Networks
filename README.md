# Phishing URL Classification Using Generative Adversarial Networks

Graduation project by **Ali AbdulMajid Alganbour**, Department of Computer Engineering, Faculty of Engineering, University of Tripoli.

This repository contains the source code, experiment definitions, aggregated evidence, final report, and browser prototype for a study of phishing-URL classification and defensive GAN augmentation. Large raw datasets, intermediate arrays, and historical training checkpoints are intentionally stored outside Git so the repository remains practical to clone.

## Main result

The experiments do not assume that synthetic augmentation must improve every metric. The final controlled result is narrower: filtered 10% MS-HardGAN-C augmentation improved the selected HYB-MS detector on several frozen internal and low-prevalence external evaluations, while other generators, ratios, classifiers, and shifted datasets produced mixed or negative results. The repository preserves those negative results as part of the evidence.

## Repository contents

| Path | Contents |
|---|---|
| `Version_1/` | Original 2M-dataset implementation and early GAN experiments |
| `Version_2/` | Controlled single-dataset experiments and distribution-shift analysis |
| `Version_3/` | Final multi-source, hybrid, GAN, calibration, and evaluation implementation |
| `extension/` | Runnable Phishing URL Guard v0.6.0 with the final ONNX ensemble |
| `results/` | Aggregated metrics, reports, and result graphs; row-level URL predictions are excluded |
| `report/` | LaTeX source and the compiled graduation report |
| `data/` | Dataset card and machine-readable artifact manifest |
| `models/` | Final model card and deployment-model description |
| `reproducibility/` | Environment, artifact, training, and evaluation instructions |
| `scripts/` | Repository and external-artifact verification tools |

## What is intentionally excluded

- Raw or redistributability-unverified URL datasets
- Participant browsing exports and row-level URL predictions
- Historical Random Forest `.joblib` checkpoints
- Generated and processed `.npy`, `.npz`, and SQLite files
- Intermediate PyTorch and GAN checkpoints
- `node_modules`, caches, obsolete report drafts, and old extension packages

The excluded files are not needed to read the results or run the packaged browser extension. Exact training reproduction requires the external datasets described in [`data/DATASET_CARD.md`](data/DATASET_CARD.md). Final archive download links will be recorded in [`data/artifacts_manifest.json`](data/artifacts_manifest.json) after the permitted artifacts are deposited.

## Quick browser demonstration

1. Open `chrome://extensions` in a Chromium-based browser.
2. Enable **Developer mode**.
3. Select **Load unpacked** and choose the `extension/` directory.
4. Confirm that the version is `0.6.0`.

The extension performs local inference using three included ONNX models. It supports English and Arabic. Caution is advisory; High Risk and Very High Risk interrupt navigation in Normal mode. It is a research prototype, not a substitute for browser security controls.

## Reproduce the experiments

Read [`reproducibility/REPRODUCIBILITY.md`](reproducibility/REPRODUCIBILITY.md). In summary:

1. Obtain only the datasets for which you have permission.
2. Put them at the exact paths in `data/artifacts_manifest.json`.
3. Run `python3 scripts/verify_artifacts.py`.
4. Create an isolated Python environment and install the requirements for the selected version.
5. Run the preparation, training, and evaluation commands documented by that version.

Neural and GAN training is configured for a CUDA-capable NVIDIA GPU. Random Forest and linear SVM are CPU baselines.

## Verify the repository

```bash
python3 scripts/check_repository.py
python3 scripts/verify_artifacts.py
```

The first command verifies that the public repository contains no oversized or excluded training artifacts. The second verifies externally downloaded data using recorded SHA-256 hashes.

## Results and report

- [`results/graphs/00_Supervisor_Selection`](results/graphs/00_Supervisor_Selection) contains the principal presentation figures.
- [`results/graphs/Complete_Archive`](results/graphs/Complete_Archive) preserves the complete result-graph hierarchy.
- [`results/final_evaluation`](results/final_evaluation) contains the final aggregated comparison.
- [`report/Graduation_Report.pdf`](report/Graduation_Report.pdf) is the compiled report.

Results from different datasets should not be compared as if they came from one test. The report identifies which comparisons share frozen rows, labels, seeds, thresholds, and evaluation procedures.

## Safety and privacy

Do not visit live phishing URLs for demonstration or testing. Use offline, defanged, or locally controlled examples. Raw user-evaluation exports are not included. Before publishing a fork, review the publication checklist and the licences of every external dataset.

## Citation and licence

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). A software licence has not been selected because that decision must be made by the copyright holder before publication; see [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
