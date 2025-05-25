# Applications of Generative AI in Emergency Department Admission Evaluation

This repository contains the code and documentation for our UChicago Applied Data Science Capstone project.

## Project Overview

We develop a generative AI system to predict hospital admissions in emergency departments (EDs) at the point of triage. Our approach combines structured data (e.g., vitals, demographics) with unstructured clinical text (triage notes) using transformer-based models and large language models (LLMs). The goal is to enable real-time, context-aware predictions to support better resource allocation and reduce patient wait times.

## Current Status

- **Synthetic Data**  
  Synthetic structured and unstructured triage data have been cleaned, merged, and prepared for modeling.

- **Modeling**  
  Apollo-7B and OpenBioLLM-8B have been fine-tuned on admission prediction tasks.

- **Preliminary Results** 

| Model                  | Eval Loss | Accuracy | Precision (Class 1) | Recall (Class 1) | Specificity (Class 1) | F1 (Class 1) | AUROC  |
|------------------------|-----------|----------|----------------------|------------------|------------------------|--------------|--------|
| BioClinicalBERT        | 0.6976    | 0.6390   | 0.6483               | 0.6698           | 0.6055                 | 0.6589       | 0.5515 |
| Apollo-7B (baseline)   | 9.0324    | 0.3159   | 0.3163               | 0.9964           | 0.0000                 | 0.4801       | 0.4991 |
| Apollo-7B (fine-tuned) | 0.6599    | 0.6636   | 0.3836               | 0.1004           | 0.9251                 | 0.1591       | 0.5355 |
| OpenBioLLM-8B (base)   | 1.0971    | 0.6409   | 0.2874               | 0.0896           | 0.8968                 | 0.1366       | 0.5041 |
| OpenBioLLM-8B (fine)   | 0.6735    | 0.6784   | 0.4804               | 0.1756           | 0.9118                 | 0.2572       | 0.5502 |
