

> This repository contains the code for **ECE 467: Natural Language Processing**, a 3-credit graduate-level course at The Cooper Union for the Advancement of Science and Art, offering an exploration of both traditional NLP methods and modern deep learning techniques.


## Natural Language Processing
**Course, Fall 2024**  
**Instructor:** Professor Carl Sable   
**Syllabus:** [View](http://faculty.cooper.edu/sable2/courses/fall2024/ece467)


### Overview

This course provides a practical introduction to natural language processing, emphasizing both traditional linguistic methods and modern deep-learning approaches. Through focused study of key textbook chapters, seminal research papers, and hands-on projects, students gain the theoretical foundations and practical skills needed to build advanced NLP systems.



### Material

The primary resource for the course is the in-progress draft of [*Speech and Language Processing, 3rd Edition*](https://web.stanford.edu/~jurafsky/slp3/) by Daniel Jurafsky and James H. Martin. This textbook lays the groundwork for understanding both statistical and deep-learning methods in NLP. 

- Traditional NLP techniques such as text categorization, syntax, and semantics.
- Deep learning fundamentals with feedforward networks, recurrent architectures (including LSTMs), and sequence-to-sequence models.
- Modern advances with transformers, BERT, GPT variations, and reinforcement learning from human feedback.

In addition, the curriculum integrates various published papers and online materials that cover essential topics, including:

<div align="center">

| **Topic**                   | **Paper** |
|-----------------------------|-----------|
| **ELMo**                    | [ELMo](https://arxiv.org/abs/1802.05365) |
| **Transformers**            | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **BERT**                    | [BERT](https://arxiv.org/abs/1810.04805) |
| **Large Language Models**   | [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| **Large Language Models**   | [GPT-3](https://arxiv.org/abs/2005.14165) |
| **RLHF**                    | [InstructGPT](https://arxiv.org/abs/2203.02155) |
| **Ethics of NLP**           | [Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922) |
</div>

### Repository Structure

```
.
├── Final_Project
│   ├── Dataset_Analysis.ipynb
│   ├── Final_Project.ipynb
│   ├── Final_Project_Presentation.pdf
│   ├── Final_Project_Report.pdf
│   └── README.md
├── P02
│   ├── P02.ipynb
│   ├── P02_Report.pdf
│   └── README.md
└── README.md
```
- **`P02.ipynb`** – *LSTM-Based Experiments*  
  - Showcases text classification using recurrent architectures and various embedding strategies.

- **`Dataset_Analysis.ipynb`** – *Exploratory Analysis*  
  - Presents data-loading steps, distributions, and initial insights to guide model development.

- **`Final_Project.ipynb`** – *Model Implementation*  
  - Demonstrates the classification pipeline, from training scripts to evaluation metrics, using transformer-based models.


### Final Project

The project, *Multi-Label Emotion Classification Using Transformer Architectures*, investigates how different transformer models perform on the task of detecting multiple emotions in text. Leveraging the GoEmotions dataset—which captures the nuanced expression of 27 distinct emotions—the goal is to evaluate how variations in architecture, from robust models like BERT and RoBERTa to efficient alternatives like DistilBERT and SqueezeBERT, affect classification accuracy and efficiency. By comparing these approaches, the project seeks to uncover innovative strategies for optimizing transformer-based NLP systems for fine-grained emotion detection.