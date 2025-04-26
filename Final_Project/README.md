## Multi-Label Emotion Classification: Comparing Transformer Models

This directory contains my final project for the **Natural Language Processing (ECE-467)** course, which focused on exploring multi-label emotion classification using transformer models. Specifically, the GoEmotions<sup>[1](#footnote1)</sup><sup>, [2](#footnote2)</sup> dataset by Google Research was used to evaluate and compare the effectiveness and efficiency of various transformer architectures.

### Objective  
To perform multi-label emotion classification on textual data, leveraging pre-trained transformer models to detect nuanced emotional expressions. The GoEmotions dataset comprises Reddit comments labeled with one or more emotions across 27 distinct categories.

**Transformer Models Evaluated:**
- **BERT & BERT-Large**
- **RoBERTa & RoBERTa-Large**
- **DistilBERT** (efficient variant)
- **SqueezeBERT** (efficient variant)

Each model was fine-tuned and optimized using systematic hyperparameter sweeps with *Weights & Biases to achieve the best performance, balancing computational efficiency and predictive accuracy.

### Directory Structure

```
.
├── Dataset_Analysis.ipynb
├── Final_Project.ipynb
├── Final_Project_Presentation.pdf
├── Final_Project_Report.pdf
└── README.md
```

- **`Dataset_Analysis.ipynb`**  
  - Exploratory analysis of the GoEmotions dataset, including label distributions, tokenization statistics, and co-occurrence of emotion labels.

- **`Final_Project.ipynb`**
  - Full implementation of model training, validation, and testing pipelines. Includes hyperparameter tuning, multi-label classification setup, and model evaluations.

- **`Final_Project_Presentation.pdf`**  
  - Slides summarizing the project goals, methodology, key results, and conclusions.

- **`Final_Project_Report.pdf`**  
  - Comprehensive report detailing the dataset, methods, experiments, and complete findings from the project.

### Results

- **Best Performing Models:**  
  RoBERTa-Large and RoBERTa consistently outperformed BERT-based models, achieving higher AUC scores and lower validation losses, indicating better optimization and generalization capabilities.

- **Lightweight Model Insights:**  
  DistilBERT provided a good trade-off between performance and efficiency, significantly outperforming SqueezeBERT in accuracy metrics. However, SqueezeBERT remained a viable option when computational speed and memory usage are critical.
<br>

<div align="center">

| Model            | AUC Score | Validation Loss |
|------------------|-----------|-----------------|
| RoBERTa-Large    | 0.957     | 0.089           |
| RoBERTa-Base     | 0.956     | 0.088           |
| BERT-Large       | 0.953     | 0.097           |
| DistilBERT       | 0.948     | 0.096           |
| SqueezeBERT      | 0.944     | 0.093           |
</div>

<br>

### Replication

All experiments were systematically tracked using **Weights & Biases (W&B)** for reproducibility. Hyperparameter settings, GPU utilization, and detailed training metrics are logged for transparency and ease of replication.

- For more detailed insights into methods, experimental setup, and complete analysis, please refer to the **Final Project Report** available in this repository.



## References 

<a name="footnote1">1</a>: [GoEmotions: A Dataset for Fine-Grained Emotion Classification](https://arxiv.org/pdf/2005.00547)

<a name="footnote2">2</a>: [Google Research Blog: GoEmotions - A Dataset for Fine-Grained Emotion Classification](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)

