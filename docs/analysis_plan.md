# FTW Spain Field Segmentation — Analysis Plan

## 1. Research Question

**How does agricultural field morphology relate to instance-segmentation
performance across heterogeneous agricultural landscapes in Spain?**

The project evaluates a Mask R-CNN baseline on Fields of the World (FTW)
Spain and investigates whether segmentation performance varies with
agricultural field morphology.

The analytical focus is:

> Morphology-aware segmentation performance and failure analysis.

Spain is treated as a case study. The objective is to identify performance
patterns, not to establish causal relationships between field morphology
and model performance.

---

## 2. Study Area and Dataset

**Study area:** Spain  
**Dataset:** Fields of the World (FTW)

### Inputs

- Sentinel-2 imagery
- RGB + NIR (4 bands)
- One temporal window
- 256 × 256 image chips
- Agricultural field instance masks

The FTW train/validation/test structure was preserved to maintain separation
between model development and held-out evaluation.

The final independent test set contains **216 tiles**.

---

## 3. Model and Experimental Configuration

### Architecture

- Mask R-CNN
- ResNet-50 FPN backbone
- 4-channel input
- 2 classes
- Instance labels enabled

### Training configuration

- Epochs: 3
- Batch size: 2
- Learning rate: 0.005
- Google Colab
- NVIDIA Tesla T4 GPU

The experiment is treated as a **compute-conscious baseline** rather than an
optimized model benchmark.

No extensive hyperparameter search or architecture comparison was performed.

The canonical trained model is:

`models/best_model.pth`

Model checkpoints are retained locally and excluded from the public
repository.

---

## 4. Morphology Variables

Morphology features were derived from ground-truth agricultural field
instances and summarized at chip level.

### Field area

Primary chip-level variable:

`median_area_ha`

### Compactness

Calculated as:

`4πA / P²`

where:

- A = field area
- P = field perimeter

Chip-level variable:

`median_compactness`

### Shape complexity

Calculated as:

`1 / compactness`

Chip-level variable:

`median_shape_complexity`

### Local field density

Represented by:

`n_morphology_fields`

This records the number of morphology-valid field instances within each chip.

---

## 5. Morphology Quality Control

Morphology screening identified several issues requiring explicit QC.

### Disconnected components

Some instance IDs contain multiple disconnected raster components.

Components belonging to the same instance ID were handled together before
morphology calculation.

### Chip boundaries

Fields touching chip boundaries may represent incomplete objects.

Boundary-touching instances were excluded where complete field geometry was
required.

### Very small instances

Very small label objects can produce unstable morphology measurements.

A pixel-based minimum-size rule was therefore applied during morphology QC.

Only chips containing valid morphology measurements were included in the
final morphology-performance analysis.

---

## 6. Evaluation Design

Final model evaluation was performed on the independent **216-tile held-out
test set**.

The workflow included:

1. CUDA-based prediction generation
2. test/prediction pair validation
3. tile-level segmentation metric calculation
4. morphology-performance joining
5. failure-pattern analysis
6. field-size stratification and visualization

Primary segmentation metrics:

- IoU
- Precision
- Recall
- F1 / Dice

The final analysis uses **tile-level segmentation metrics and chip-level
morphology summaries**.

A separate object-matching benchmark was not performed.

Detailed numerical results are reported in:

`docs/results.md`

---

## 7. Morphology × Performance Analysis

The main analysis evaluates relationships between held-out segmentation
performance and:

- median field area
- median compactness
- median shape complexity
- number of morphology-valid fields

Continuous relationships are examined using Spearman rank correlation.

Field-area quartiles are additionally used for interpretation and
visualization.

Representative prediction examples are used to inspect morphology-related
failure patterns.

No causal relationship is inferred from these associations.

---

## 8. Experimental Scope

The project intentionally uses a narrow experimental design:

- one country
- one dataset
- one model architecture
- one temporal window
- one spectral configuration
- one fixed training configuration
- one held-out evaluation workflow

### In scope

- FTW Spain
- Sentinel-2 RGB + NIR
- Mask R-CNN
- instance segmentation
- agricultural field morphology
- held-out evaluation
- failure analysis
- Python
- Colab GPU

### Out of scope

- cross-country comparison
- architecture benchmarking
- SAM / SAM-3
- temporal stacking
- extensive hyperparameter tuning
- change detection
- web application

> **Scope rule:** Spain + one model + one research question + clean evaluation.

---

## 9. Interpretation Boundaries

Observed morphology-performance relationships are treated as
**associations rather than causal effects**.

Potential performance differences may reflect interacting factors such as:

- spatial resolution and object scale
- limited training duration
- model architecture
- landscape complexity
- regional or dataset characteristics

The current experiment does not isolate these mechanisms.

Results should therefore be interpreted as specific to the FTW Spain case
study and the model/configuration used here.

---

## 10. Project Workflow

### Phase 0 — Feasibility
- [x] compute feasibility screening
- [x] Spain mask and imagery QC
- [x] morphology feasibility screening
- [x] Spain-only experimental decision

### Phase 1 — Experimental Setup
- [x] finalize research question
- [x] define project scope
- [x] validate dataset structure
- [x] define morphology QC
- [x] freeze baseline configuration

### Phase 2 — Model Training
- [x] prepare training data
- [x] train Mask R-CNN baseline
- [x] save model checkpoints
- [x] record validation performance

### Phase 3 — Held-Out Evaluation
- [x] generate CUDA predictions
- [x] validate test/prediction pairs
- [x] calculate segmentation metrics
- [x] characterize failure behavior

### Phase 4 — Morphology Analysis
- [x] create morphology features
- [x] apply morphology QC
- [x] join morphology and performance
- [x] analyze morphology-performance relationships
- [x] perform field-size quartile analysis
- [x] inspect representative failure cases

### Phase 5 — Analysis Completion
- [x] complete held-out evaluation
- [x] complete morphology-performance analysis
- [x] generate final result visualizations
- [x] document methodological limitations
- [x] preserve invalid CPU inference as a local debugging artifact
- [x] freeze the final analytical scope

---

## 11. Analysis Status

**Analysis complete.**

Model training, held-out evaluation, morphology analysis, and failure
analysis are finalized.

No additional model training or analytical experiments are planned for this
case study unless a methodological error is identified.

Repository documentation and publication are handled separately from the
analytical workflow.