# Downstream Task: MODIS Land Cover Classification

**_NOTE: The Getting Started section must be completed [first.](requirements/README.md)_**

## Objective

The objective of this task is to leverage the advanced capabilities of SatVision-TOA
to perform land cover segmentation using MODIS satellite data. This task aims to accurately 
classify and delineate different types of land cover (such as forests, urban areas, agricultural lands, 
and water bodies) from MODIS imagery. By utilizing SatVision-TOA's pre-trained knowledge on a vast array 
of satellite imagery, including its ability to handle all-sky conditions and coarse-resolution data, 
this tasks seeks to demonstrate its capabilities for land cover mapping. The approach will involve 
fine-tuning SatVision-TOA on a labeled dataset of MODIS images representing various land cover types, then 
applying the model to segment and classify land cover across larger geographic areas.

## Downstream Task Development

### 1. Setting up input data

For this task, we take MOD021KM data available in Earthdata or NCCS Explore, and generate
daily composites from the available swaths. This will provide us with the necessary input
data to feed into the land cover model. For this demonstration task, we will simply select
a couple of days not used by the pre-training of the model, selecting Summer months for
leaf on dates to easy the differentiation of leaf on features.

a. Running composites

```bash
```

b. Clipping and selecting the tiles of interest

```bash
```

### 2. Extracting labels

As the labels, we leverage MCD12Q1 (MODIS Land Cover Type Yearly L3 Global 500m SIN Grid)
as our ground truth for training. Here we select a single year for the given tiles we
want to analyze.

a. Download the label data

```bash
```

b. Clipping and selecting the tiles of interest

```bash
```

### 3. Extraction training and validation dataset

We then select different training datasets to understand the performance
of these models under different conditions. We have a dataset with 100, 
500, 1000, and 5000 samples to use for fine-tuning our model.

```bash
```

### 4. Fine-tuning SatVision-TOA for land cover classification

In this task, we go ahead and train our fine-tuned models using the
generated datasets.

```bash
```

### 5. Validation

Then, we proceed to validate our models and produce additional metrics.
Visualizations can be generated with a jupyter notebook after the
metric CSV files are producing during the validation step of the 
pipeline.

```bash
```
