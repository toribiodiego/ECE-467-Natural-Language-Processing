# GoEmotions Dataset Analysis

This document provides comprehensive statistics and analysis of the GoEmotions dataset, including sample distributions, multi-label characteristics, and per-emotion breakdowns.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Multi-Label Distribution](#multi-label-distribution)
3. [Per-Emotion Statistics](#per-emotion-statistics)
4. [Split Distribution](#split-distribution)
5. [Label Co-occurrence Patterns](#label-co-occurrence-patterns)
6. [Text Length Distributions](#text-length-distributions)
7. [Split Consistency Validation](#split-consistency-validation)
8. [Data Files Reference](#data-files-reference)

---

## Dataset Overview

**Source:** [HuggingFace Datasets - go_emotions](https://huggingface.co/datasets/go_emotions)

**Dataset Characteristics:**
- **Total Samples:** 58,009 Reddit comments
- **Emotion Labels:** 28 total (27 emotions + neutral)
- **Label Type:** Multi-label (samples can have multiple emotions)
- **Domain:** Social media text (Reddit comments)
- **Language:** English

**Dataset Splits:**
- **Train:** 43,410 samples (74.8%)
- **Validation:** 5,426 samples (9.4%)
- **Test:** 5,427 samples (9.4%)

**Emotion Categories:**

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

---

## Multi-Label Distribution

### Overall Statistics

**Across all splits:**
- **Single-label samples:** ~83.8% (majority have exactly one emotion)
- **Two-label samples:** ~14.8% (multiple emotions, but not excessive)
- **Three+ label samples:** ~1.3% (rare but present)

**Key Finding:** 16.2% of samples contain multiple emotions, making this a true multi-label classification problem.

### Per-Split Breakdown

| Split      | 1 Label (%) | 2 Labels (%) | 3+ Labels (%) | Total Samples |
|------------|-------------|--------------|---------------|---------------|
| Train      | 83.6        | 15.1         | 1.3           | 43,410        |
| Validation | 83.8        | 14.9         | 1.3           | 5,426         |
| Test       | 84.6        | 14.3         | 1.2           | 5,427         |

**Observation:** Distribution is consistent across splits, indicating proper stratification.

**Data Source:** `output/stats/multi_label_stats.csv`

---

## Per-Emotion Statistics

### Frequency Distribution

Emotions ranked by total frequency (descending):

| Rank | Emotion         | Total Count | % of Dataset |
|------|-----------------|-------------|--------------|
| 1    | neutral         | 17,772      | 30.6%        |
| 2    | admiration      | 5,122       | 8.8%         |
| 3    | approval        | 3,687       | 6.4%         |
| 4    | gratitude       | 3,372       | 5.8%         |
| 5    | annoyance       | 3,093       | 5.3%         |
| 6    | amusement       | 2,895       | 5.0%         |
| 7    | curiosity       | 2,723       | 4.7%         |
| 8    | disapproval     | 2,581       | 4.4%         |
| 9    | love            | 2,576       | 4.4%         |
| 10   | optimism        | 1,976       | 3.4%         |
| ...  | ...             | ...         | ...          |
| 28   | grief           | 96          | 0.2%         |

**Class Imbalance:**
- **Most frequent:** neutral (17,772 samples)
- **Least frequent:** grief (96 samples)
- **Imbalance ratio:** 185:1 (neutral vs. grief)

**Implications:**
- Class-weighted loss may be necessary for rare emotions
- Per-class threshold tuning likely needed for optimal F1 scores
- Evaluation metrics should use macro-averaging to avoid bias toward frequent classes

**Data Source:** `output/stats/per_emotion_supports_by_split.csv`

### Multi-Label Characteristics by Emotion

Distribution of single-label vs. multi-label samples for each emotion:

| Emotion      | Total | 1 Label (%) | 2 Labels (%) | 3+ Labels (%) |
|--------------|-------|-------------|--------------|---------------|
| neutral      | 17772 | 90.1        | 9.4          | 0.5           |
| fear         | 764   | 72.4        | 23.3         | 4.3           |
| amusement    | 2895  | 70.7        | 26.4         | 2.9           |
| gratitude    | 3372  | 70.5        | 25.9         | 3.6           |
| disapproval  | 2581  | 70.1        | 27.1         | 2.8           |
| surprise     | 1330  | 67.8        | 28.9         | 3.2           |
| admiration   | 5122  | 66.1        | 29.9         | 4.0           |
| remorse      | 669   | 65.3        | 29.6         | 5.1           |
| embarrass.   | 375   | 65.6        | 29.1         | 5.3           |
| ...          | ...   | ...         | ...          | ...           |
| grief        | 96    | 49.0        | 41.7         | 9.4           |
| pride        | 142   | 47.2        | 45.8         | 7.0           |

**Key Observations:**

1. **Neutral is most often single-label** (90.1%)
   - Neutral samples rarely co-occur with other emotions
   - May be easier to predict due to clear semantic distinction

2. **Rare emotions have high multi-label rates**
   - Grief: 51.0% multi-label (only 49.0% single-label)
   - Pride: 52.8% multi-label
   - Rare emotions often co-occur with more common emotions

3. **Multi-label complexity varies by emotion**
   - Some emotions are "standalone" (neutral, fear, amusement)
   - Others are frequently combined (grief, pride, nervousness)

**Implications for Modeling:**
- Models may learn to predict frequent standalone emotions more easily
- Rare, multi-label emotions may require careful threshold tuning
- Co-occurrence patterns should be analyzed for error diagnosis

**Data Source:** `output/stats/per_emotion_multilabel.csv`

---

## Split Distribution

### Per-Emotion Counts by Split

Example emotions (full table in CSV):

| Emotion    | Train  | Val  | Test | Total  | Train % | Val % | Test % |
|------------|--------|------|------|--------|---------|-------|--------|
| neutral    | 14,219 | 1,766| 1,787| 17,772 | 80.0    | 9.9   | 10.1   |
| admiration | 4,130  | 488  | 504  | 5,122  | 80.6    | 9.5   | 9.8    |
| approval   | 2,939  | 397  | 351  | 3,687  | 79.7    | 10.8  | 9.5    |
| gratitude  | 2,662  | 358  | 352  | 3,372  | 78.9    | 10.6  | 10.4   |
| grief      | 77     | 13   | 6    | 96     | 80.2    | 13.5  | 6.3    |

**Observations:**

1. **Consistent split proportions**
   - Most emotions maintain ~80% train, ~10% val, ~10% test
   - Stratification appears effective for common emotions

2. **Rare emotion variability**
   - Grief test split (6.3%) is lower than expected
   - Small sample size (96 total) makes perfect stratification difficult
   - May lead to higher variance in test metrics for rare emotions

**Implications:**
- Evaluation on rare emotions may have higher confidence intervals
- Cross-validation could provide more robust estimates for rare classes
- Report per-class metrics with sample size context

**Data Source:** `output/stats/per_emotion_supports_by_split.csv`

---

## Label Co-occurrence Patterns

### Overview

Understanding which emotions frequently appear together helps inform multi-label modeling strategies and error analysis. The co-occurrence matrix captures these relationships across all dataset splits.

### Key Statistics

- **Total label occurrences:** 63,812 (sum of all emotions across all samples)
- **Total co-occurrences:** 10,321 (unique emotion pairs appearing together)
- **Co-occurrence rate:** 16.2% of occurrences involve multiple labels

### Most Common Co-occurring Pairs

Based on `label_cooccurrence.csv`, the most frequently co-occurring emotion pairs are:

1. **anger + annoyance** (348 co-occurrences)
   - Related negative emotions often triggered by similar contexts

2. **admiration + love** (236 co-occurrences)
   - Positive emotions expressing appreciation and affection

3. **joy + love** (high correlation in positive contexts)

### Emotion Relationship Patterns

**Standalone emotions:**
- **neutral** (90.1% single-label) - rarely co-occurs with other emotions
- **gratitude** (high single-label percentage) - typically expressed alone

**Co-occurring emotions:**
- **Negative cluster:** anger, annoyance, disappointment, disapproval often appear together
- **Positive cluster:** admiration, joy, love, excitement show strong correlations
- **Ambiguous emotions:** surprise, realization frequently combine with other emotions

### Implications for Modeling

1. **Threshold tuning**
   - Emotions with high co-occurrence need coordinated thresholds
   - Standalone emotions can use independent thresholds

2. **Error analysis**
   - Confusing correlated emotions (e.g., anger/annoyance) is less severe than unrelated errors
   - Models should learn to predict emotion clusters appropriately

3. **Evaluation metrics**
   - Hamming loss may be too strict for correlated emotions
   - F1 scores better capture multi-label performance
   - Per-pair accuracy could measure correlation learning

**Data Source:** `output/stats/label_cooccurrence.csv`

**Visualization:** `output/figures/02_label_cooccurrence.png` (heatmap)

---

## Text Length Distributions

### Overview

Understanding text length distributions is critical for selecting appropriate `max_seq_length` values during model training. These statistics show character and token counts across all dataset splits, using the RoBERTa tokenizer.

### Key Statistics

**Character lengths:**
- **Mean:** ~68 characters across all splits
- **Median:** ~65 characters
- **95th percentile:** ~132 characters
- **Range:** 2-703 characters (train split)

**Token lengths (RoBERTa tokenizer):**
- **Mean:** ~19 tokens across all splits
- **Median:** 18 tokens
- **95th percentile:** 33 tokens
- **Range:** 3-1437 tokens (train split, outlier)

### Truncation Analysis

Percentage of samples that would be truncated at common `max_seq_length` values:

| Split      | max=128 | max=256 | max=512 |
|------------|---------|---------|---------|
| Train      | 0.01%   | 0.00%   | 0.00%   |
| Validation | 0.00%   | 0.00%   | 0.00%   |
| Test       | 0.02%   | 0.00%   | 0.00%   |

**Key finding:** GoEmotions comments are very short. Even with `max_seq_length=128`, less than 0.02% of samples require truncation. Most comments are under 33 tokens (95th percentile).

### Implications for Modeling

1. **Max sequence length selection**
   - `max_seq_length=128` is more than sufficient (captures 99.99% of samples)
   - Using 256 or 512 wastes memory and compute with no benefit
   - Shorter sequences enable larger batch sizes and faster training

2. **Efficiency optimization**
   - Short texts favor efficient models (DistilBERT, RoBERTa-base)
   - Dynamic padding can further reduce memory usage
   - Longer context models (Longformer, BigBird) not needed

3. **Split consistency**
   - Length distributions are nearly identical across splits
   - No drift in text length between train/val/test
   - Model performance differences won't be due to length mismatches

**Data Source:** `output/stats/text_length_statistics.csv`, `output/stats/truncation_analysis.csv`

**Visualization:** `output/figures/03_text_length_distributions.png` (histograms by split)

---

## Split Consistency Validation

### Overview

Statistical validation to ensure train/validation/test splits have consistent distributions. Distribution drift between splits can lead to poor generalization and unreliable model evaluation.

### Validation Tests

**Module:** `src.analysis.split_consistency`

**Command:**
```bash
python -m src.analysis.split_consistency
```

### 1. Label Distribution Consistency

**Test:** Chi-square test of independence

**Null Hypothesis:** Label proportions are consistent across splits

**Results:**
- Chi-square statistic: 52.82
- P-value: 0.52 (>> 0.05 threshold)
- Degrees of freedom: 54
- **Conclusion:** PASS - Label distributions are statistically consistent

**Largest differences (percentage points):**
1. approval: 0.68% (train: 5.75%, val: 6.22%, test: 5.55%)
2. disapproval: 0.62% (train: 3.96%, val: 4.58%, test: 4.22%)
3. curiosity: 0.60% (train: 4.29%, val: 3.89%, test: 4.49%)
4. amusement: 0.58% (train: 4.56%, val: 4.75%, test: 4.17%)
5. neutral: 0.56% (train: 27.82%, val: 27.68%, test: 28.24%)

All differences are minor (< 1%) and not statistically significant.

### 2. Text Length Distribution Consistency

**Test:** Kolmogorov-Smirnov two-sample tests

**Null Hypothesis:** Text length distributions are equivalent

**Results:**
- Train vs Validation: KS stat = 0.012, p-value = 0.45
- Train vs Test: KS stat = 0.010, p-value = 0.67
- Validation vs Test: KS stat = 0.014, p-value = 0.67
- **Conclusion:** PASS - All comparisons show consistent distributions

**Length statistics:**
| Split      | Mean  | Std   | Median | Min | Max |
|------------|-------|-------|--------|-----|-----|
| Train      | 68.40 | 36.72 | 65     | 2   | 703 |
| Validation | 68.24 | 36.91 | 64     | 5   | 187 |
| Test       | 67.82 | 36.32 | 65     | 5   | 184 |

### 3. Multi-Label Distribution Consistency

**Test:** Chi-square test of independence

**Null Hypothesis:** Multi-label proportions (1 label, 2 labels, 3+ labels) are consistent

**Results:**
- Chi-square statistic: 3.27
- P-value: 0.51 (>> 0.05 threshold)
- **Conclusion:** PASS - Multi-label distributions are consistent

**Distribution breakdown:**
| Split      | 1 Label | 2 Labels | 3+ Labels |
|------------|---------|----------|-----------|
| Train      | 83.6%   | 15.1%    | 1.3%      |
| Validation | 83.8%   | 14.9%    | 1.3%      |
| Test       | 84.6%   | 14.3%    | 1.2%      |

### Validation Summary

**Result:** 3/3 consistency tests passed

**Conclusion:** All splits have statistically consistent distributions across label proportions, text lengths, and multi-label patterns. No evidence of distribution drift that would compromise model evaluation.

**Implications:**
- Model performance differences between splits reflect true generalization, not dataset bias
- Validation and test metrics are reliable indicators of production performance
- No need for stratified sampling or distribution corrections

**Data Exports:**
- `output/stats/split_label_proportions.csv` - Per-label proportions across splits
- `output/stats/split_length_statistics.csv` - Text length statistics per split

---

## Data Files Reference

### CSV Exports

All statistics are exported to CSV files for reproducibility and downstream analysis.

#### `output/stats/multi_label_stats.csv`

**Purpose:** Overall multi-label distribution across splits

**Columns:**
- `Split`: Dataset split name (Train, Validation, Test)
- `1 Label (%)`: Percentage of samples with exactly 1 emotion
- `2 Labels (%)`: Percentage of samples with exactly 2 emotions
- `3+ Labels (%)`: Percentage of samples with 3 or more emotions
- `Total Samples`: Total number of samples in split

**Generated by:** `python -m src.data.multilabel_stats`

**Use cases:**
- Reporting dataset characteristics in papers/presentations
- Understanding multi-label complexity
- Caption text for multi-label distribution figures

---

#### `output/stats/per_emotion_multilabel.csv`

**Purpose:** Per-emotion multi-label breakdown showing how often each emotion appears alone vs. with others

**Columns:**
- `emotion`: Emotion category name
- `total_frequency`: Total number of samples containing this emotion
- `1_label_count`: Number of samples with only this emotion (single-label)
- `1_label_pct`: Percentage of samples that are single-label
- `2_labels_count`: Number of samples with exactly 2 emotions including this one
- `2_labels_pct`: Percentage of samples that are two-label
- `3plus_labels_count`: Number of samples with 3 or more emotions including this one
- `3plus_labels_pct`: Percentage of samples that are three-plus-label

**Sorting:** Emotions sorted by total_frequency (descending)

**Generated by:** `python -m src.visualization.class_distribution`

**Use cases:**
- Understanding which emotions are standalone vs. co-occurring
- Identifying emotions that may need special threshold tuning
- Analyzing model errors in multi-label context
- Documenting dataset characteristics for ablation studies

**Data integrity:** All percentages sum to 100% for each emotion. Counts sum to total_frequency.

---

#### `output/stats/per_emotion_supports_by_split.csv`

**Purpose:** Per-emotion sample counts for each dataset split (train/val/test)

**Columns:**
- `emotion`: Emotion category name
- `train_count`: Number of samples in training split
- `val_count`: Number of samples in validation split
- `test_count`: Number of samples in test split
- `total_count`: Total samples across all splits

**Sorting:** Emotions sorted by total_count (descending)

**Generated by:** `python -m src.visualization.class_distribution`

**Use cases:**
- Joining with per-class metrics for performance analysis
- Analyzing class imbalance across splits
- Computing per-class sample weights for training
- Verifying stratification quality
- Reporting confidence intervals for rare-class metrics

**Data integrity:** train_count + val_count + test_count = total_count for all emotions

---

#### `output/stats/split_label_proportions.csv`

**Purpose:** Per-label proportion analysis across splits for consistency validation

**Columns:**
- `label`: Emotion category name
- `train_count`: Sample count in training split
- `train_pct`: Percentage of training samples with this label
- `val_count`: Sample count in validation split
- `val_pct`: Percentage of validation samples with this label
- `test_count`: Sample count in test split
- `test_pct`: Percentage of test samples with this label
- `max_diff_pct`: Maximum difference in proportions across splits

**Sorting:** Labels sorted by max_diff_pct (descending)

**Generated by:** `python -m src.analysis.split_consistency`

**Use cases:**
- Identifying labels with potential distribution drift
- Verifying split stratification quality
- Quality assurance for dataset splitting
- Documenting split consistency in research papers

**Interpretation:** Small max_diff_pct values (< 1%) indicate good split consistency

---

#### `output/stats/split_length_statistics.csv`

**Purpose:** Text length summary statistics for each dataset split

**Columns:**
- `split`: Dataset split name (train, validation, test)
- `count`: Number of samples in split
- `mean`: Mean character length
- `std`: Standard deviation of character length
- `median`: Median character length
- `min`: Minimum character length
- `max`: Maximum character length
- `q25`: 25th percentile (Q1)
- `q75`: 75th percentile (Q3)

**Sorting:** By split (train, validation, test)

**Generated by:** `python -m src.analysis.split_consistency`

**Use cases:**
- Comparing text length distributions across splits
- Detecting potential length-based sampling bias
- Quality assurance for dataset splitting
- Informing tokenization strategy

**Interpretation:** Similar statistics across splits indicate consistent length distributions

---

#### `output/stats/label_cooccurrence.csv`

**Purpose:** Label co-occurrence matrix showing how frequently emotion pairs appear together

**Format:** CSV matrix with emotions as both rows and columns

**Columns:**
- `label`: Emotion name (row label)
- Remaining columns: One per emotion, containing co-occurrence counts

**Matrix properties:**
- **Diagonal:** Self-occurrence counts (how many times each emotion appears overall)
- **Off-diagonal:** Co-occurrence counts (how many samples contain both emotions)
- **Symmetric:** Matrix[i,j] = Matrix[j,i] (co-occurrence is bidirectional)

**Sorting:** Emotions in alphabetical order

**Generated by:** `python -m src.visualization.label_cooccurrence`

**Use cases:**
- Identifying emotion correlations and patterns
- Understanding which emotions frequently co-occur
- Informing multi-label modeling strategies
- Analyzing error patterns (e.g., confusing correlated emotions)
- Designing ablation studies on emotion relationships

**Example insights:**
- Most common pair: anger + annoyance (348 co-occurrences)
- Emotions with high self-occurrence but low co-occurrence are typically standalone
- Emotions with high off-diagonal counts indicate strong relationships

**Data integrity:**
- All counts non-negative
- Matrix is symmetric (cooccurrence[i,j] = cooccurrence[j,i])
- Diagonal values match total_count in per_emotion_supports_by_split.csv

---

#### `output/stats/text_length_statistics.csv`

**Purpose:** Text length statistics (character and token counts) per dataset split

**Columns:**
- `split`: Dataset split name (train, validation, test)
- `num_samples`: Total number of samples in split
- `char_min`: Minimum character count
- `char_max`: Maximum character count
- `char_mean`: Mean character count
- `char_median`: Median character count
- `char_std`: Standard deviation of character counts
- `char_q25`: 25th percentile character count
- `char_q75`: 75th percentile character count
- `char_q90`: 90th percentile character count
- `char_q95`: 95th percentile character count
- `char_q99`: 99th percentile character count
- `token_min`: Minimum token count
- `token_max`: Maximum token count
- `token_mean`: Mean token count
- `token_median`: Median token count
- `token_std`: Standard deviation of token counts
- `token_q25`: 25th percentile token count
- `token_q75`: 75th percentile token count
- `token_q90`: 90th percentile token count
- `token_q95`: 95th percentile token count
- `token_q99`: 99th percentile token count

**Tokenizer:** RoBERTa-base tokenizer

**Sorting:** By split (train, validation, test)

**Generated by:** `python -m src.analysis.text_length_analysis`

**Use cases:**
- Selecting appropriate `max_seq_length` for training
- Understanding memory and compute requirements
- Comparing text length consistency across splits
- Identifying outliers or truncation needs
- Informing batch size and padding strategies

**Key insight:** 95th percentile is only 33 tokens, so `max_seq_length=128` is more than sufficient

---

#### `output/stats/truncation_analysis.csv`

**Purpose:** Truncation rates at common `max_seq_length` values per split

**Columns:**
- `split`: Dataset split name (train, validation, test)
- `max_seq_length`: Maximum sequence length tested (128, 256, 512)
- `truncation_rate_pct`: Percentage of samples that exceed max_seq_length
- `samples_truncated`: Number of samples that would be truncated
- `total_samples`: Total number of samples in split

**Sorting:** By split, then by max_seq_length

**Generated by:** `python -m src.analysis.text_length_analysis`

**Use cases:**
- Quantifying information loss from truncation
- Justifying `max_seq_length` choice in ablation studies
- Reporting truncation rates in papers/documentation
- Comparing truncation across splits for consistency

**Key finding:** Truncation rate is <0.02% even at `max_seq_length=128`

---

### Visualizations

#### `output/figures/class_distribution_stacked.png`

**Shows:** All 28 emotions with multi-label breakdown (1-label, 2-labels, 3+ labels)

**Purpose:** Complete view of dataset distribution

**Y-axis scale:** ~18,000 (dominated by neutral's 17,772 samples)

**Use case:** Comprehensive dataset overview including neutral

---

#### `output/figures/class_distribution_stacked_no_neutral.png`

**Shows:** 27 emotions (excluding neutral) with multi-label breakdown

**Purpose:** Better visual clarity of non-neutral emotions

**Y-axis scale:** ~5,000 (scales to admiration's 5,122 samples)

**Use case:** Detailed view of multi-label patterns in non-neutral emotions

**Note:** This is a visualization-only exclusion. Neutral is retained in all dataset processing, training, and CSV exports. See `design_decisions.md` for rationale.

---

## Generating Statistics

All statistics can be regenerated from scratch using the following commands:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate multi-label distribution statistics
python -m src.data.multilabel_stats

# Generate class distribution visualizations and per-emotion statistics
python -m src.visualization.class_distribution
```

**First run:** ~30-40 seconds (downloads dataset from HuggingFace Hub)

**Subsequent runs:** ~8-10 seconds (uses cached dataset from `~/.cache/huggingface/datasets`)

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
