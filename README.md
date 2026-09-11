# Field Instance Segmentation in Spain

### Evaluating Mask R-CNN Across Agricultural Landscape Morphologies

An end-to-end GeoAI case study using **Fields of the World (FTW) Spain** to
evaluate agricultural field instance segmentation and investigate how model
performance varies across different field morphologies.

Rather than reporting only an overall accuracy score, the project asks:

> **How does agricultural field morphology relate to instance-segmentation
> performance across heterogeneous agricultural landscapes in Spain?**

---

## Key Finding

Field size showed the strongest observed relationship with held-out
segmentation performance.

- **Spearman ρ = +0.478** between median field area and IoU
- Mean IoU increased from **0.317** for the smallest-field quartile to
  **0.573** for the largest-field quartile

![Held-out IoU by field-size quartile](outputs/test_evaluation/field_size_quartile_iou_boxplot.png)

For this three-epoch Spain baseline, segmentation performance was
substantially lower in landscapes dominated by smaller agricultural fields.

This is an **association, not a causal claim**.

---

## Project Overview

The project combines deep-learning-based field segmentation with
morphology-aware error analysis.

**Dataset:** Fields of the World — Spain  
**Imagery:** Sentinel-2 RGB + NIR (4 bands)  
**Model:** Mask R-CNN with ResNet-50 FPN  
**Task:** Agricultural field instance segmentation  
**Training:** 3-epoch compute-conscious baseline  
**Final evaluation:** 216 independent held-out tiles

The FTW train/validation/test structure was preserved throughout the
experiment.

---

## Held-Out Test Results

| Metric | Score |
|---|---:|
| Mean IoU | **0.4938** |
| Median IoU | 0.5122 |
| Mean Precision | **0.8321** |
| Mean Recall | 0.5115 |
| Mean F1 / Dice | **0.6551** |

The model showed relatively strong precision but moderate recall, suggesting
that **under-detection and missed fields were more prominent than excessive
false positives**.

Prediction/test validation confirmed:

- 216 test tiles
- 216 prediction sets
- 0 pairing problems

---

## Morphology-Aware Analysis

Ground-truth field instances were used to derive four chip-level morphology
variables:

- median field area
- median compactness
- median shape complexity
- number of morphology-valid fields

After morphology quality control, **197 of 216 held-out tiles** contained
valid morphology summaries.

Spearman correlations with IoU showed:

| Variable | Spearman ρ |
|---|---:|
| Median field area | **+0.478** |
| Median compactness | +0.121 |
| Median shape complexity | -0.137 |
| Number of fields | -0.123 |

Field size was therefore the clearest morphology-related performance pattern
observed in this experiment.

---

## Failure Analysis

Representative small- and large-field landscapes were inspected using
satellite imagery, ground-truth masks, and model predictions.

![Small vs large field examples](outputs/test_evaluation/small_vs_large_field_examples.png)

Smaller-field examples showed more missed detections and weaker recovery of
small or fragmented field regions.

Larger-field examples generally showed stronger spatial agreement, although
segmentation and boundary errors remained.

These examples support interpretation of the quantitative results but are
treated as qualitative illustrations rather than independent statistical
evidence.

---

## Methodological Notes

Morphology extraction required explicit quality control because agricultural
field masks can contain:

- disconnected components belonging to the same instance
- fields truncated by chip boundaries
- very small objects with unstable morphology measurements

The workflow therefore included component-aware processing, boundary-object
screening, and minimum-size filtering before morphology-performance analysis.

The final analysis uses **tile-level segmentation metrics and chip-level
morphology summaries**.

A separate object-matching benchmark was not performed.

---

## Repository Structure

```text
ftw_field_segmentation/
├── README.md
├── environment.yml
├── docs/
│   ├── analysis_plan.md
│   └── results.md
├── notebooks/
│   ├── ftw_spain_training_validation.ipynb
│   └── FTW_Spain_Heldout_Test_Inference.ipynb
├── scripts/
│   ├── 01_screen_spain_masks.py
│   ├── 02_screen_spain_morphology.py
│   ├── 03_check_spain_imagery_coverage.py
│   ├── 04_build_spain_dataset_index.py
│   ├── 05_inspect_spain_samples.py
│   ├── 06_build_chip_morphology_features.py
│   ├── 07_explore_morphology_distributions.py
│   ├── 08_validate_spain_test_pairs.py
│   ├── 09_evaluate_spain_test_metrics.py
│   ├── 10_plot_field_size_vs_iou.py
│   └── 11_visualize_size_failure_cases.py
├── outputs/
│   ├── morphology/
│   ├── qc_samples/
│   └── test_evaluation/
└── models/
    └── training_summary.txt
```

Large datasets, model checkpoints, prediction rasters, debug outputs, and
local archives are excluded from version control.

---

## Reproducibility

### Local analysis environment

The local environment supports morphology processing, quality control,
evaluation, and visualization.

```bash
conda env create -f environment.yml
conda activate ftw-field-segmentation
```

### Model training and inference

Mask R-CNN training and final held-out CUDA inference were performed in
**Google Colab using an NVIDIA Tesla T4 GPU**.

The repository includes the final notebooks documenting these workflows:

- `notebooks/ftw_spain_training_validation.ipynb`
- `notebooks/FTW_Spain_Heldout_Test_Inference.ipynb`

Model checkpoints and generated prediction rasters are not included in the
repository because of their size.

---

## Documentation

For methodological details:

[`docs/analysis_plan.md`](docs/analysis_plan.md)

For complete held-out results, morphology analysis, and limitations:

[`docs/results.md`](docs/results.md)

---

## Limitations

This project should be interpreted as a controlled case study rather than a
general segmentation benchmark.

Key limitations include:

- training was limited to three epochs
- hyperparameters were not extensively optimized
- only one model architecture was evaluated
- only Spain was studied
- cross-country generalization was not tested

The observed field-size relationship may reflect spatial resolution,
small-object representation, training duration, model architecture,
landscape characteristics, or interactions between these factors.

The current experiment does not isolate these mechanisms.

---

## Future Research

The results motivate a broader question:

> **Is field-size sensitivity a general property of satellite-based instance
> segmentation, or does it depend on regional agricultural landscape
> characteristics?**

Answering this would require controlled cross-region and/or cross-model
experiments and is outside the scope of the current project.

---

## Data

This project uses the **Fields of the World (FTW)** dataset, with **Spain** selected as the case study.

- **Dataset:** Fields of the World (FTW)
- **Region:** Spain
- **Imagery:** Sentinel-2 RGB + NIR
- **Source:** [Fields of the World](https://fieldsofthe.world/)
- **Repository:** [Fields of the World — GitHub](https://github.com/fieldsoftheworld/fieldsoftheworld)

Raw imagery and labels are not redistributed in this repository. Only derived analytical outputs required to document the workflow and results are included.

---

## License

This project is licensed under the [MIT License](LICENSE).