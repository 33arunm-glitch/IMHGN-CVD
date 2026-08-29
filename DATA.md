# Data specification

## Dataset scope

The manuscript describes two public cardiovascular datasets:

- **DF1**: 1,190 patient records, 11 principal predictive variables in the harmonized cardiovascular schema.
- **DF2**: 1,000 patient records, with a closely related feature set and an additional source-specific variable.
- **Combined analysis set**: 2,190 records after harmonization.

The manuscript reports class proportions separately for DF1, DF2, and the combined set. Those dataset-specific proportions should not be replaced by one global percentage when describing the individual sources.

This repository does not redistribute the source datasets. Researchers must download them from the original repositories cited in the manuscript and comply with the source terms.

## Expected target

Binary label:

```text
0 = non-CVD / normal
1 = CVD / heart disease
```

The target column name is configurable through `config.yaml`.

## Canonical feature names

The repository uses the following canonical names where available:

```text
age
sex
chest_pain_type
resting_bp
cholesterol
fasting_bs
resting_ecg
max_hr
exercise_angina
oldpeak
st_slope
target
```

Dataset-specific names are normalized through the alias map in `config.yaml`.

## Modalities

The final implementation uses exactly three modalities.

### Clinical

Typical fields:

```text
chest_pain_type
resting_bp
cholesterol
fasting_bs
resting_ecg
max_hr
oldpeak
st_slope
```

### Demographic

```text
age
sex
```

### Lifestyle / activity-related

```text
exercise_angina
```

No genetic modality is required by this implementation.

## Harmonization

The loader:

1. normalizes column names;
2. applies configured aliases;
3. validates the target;
4. retains configured modality features that are actually available;
5. concatenates DF1 and DF2 using aligned canonical columns.

A missing feature in one source is represented as missing and is imputed only after fold construction.

## Missing-value handling

The canonical implementation uses `sklearn.experimental.enable_iterative_imputer` with `IterativeImputer` as an MICE-style multivariate iterative imputation procedure.

Imputers are fitted on training data only.

## Encoding

Object/category columns are converted to deterministic integer category codes by the preprocessing component before iterative imputation. For source datasets in which categorical variables are already integer encoded, their original numerical encoding is preserved.

## Scaling

Min-Max scaling is fitted to each training modality and applied to its validation/test counterpart.

## RFE and PCA

Both are supported as optional, modality-preserving transformations.

- RFE operates independently within each modality.
- PCA, when enabled, operates independently within each modality and retains the configured explained-variance fraction.

The default configuration leaves both disabled because the exact source-feature treatment must match the final manuscript specification. Enable them only if they are retained in the submitted Methods section.

## Class balancing

SMOTE is applied only to the transformed training partition.

Validation and test partitions are never oversampled.

## Leakage control

The following objects are fitted only on training partitions:

```text
category mappings
iterative imputers
scalers
RFE selectors
PCA transforms
SMOTE synthesis
ABO hyperparameter selection
```

The outer test fold is used only for final fold evaluation.
