# Patronizing and Condescending Language Detection
This repository contains the code and report for detecting patronizing and condescending language. The task was Subtask 1 of Task 4 from the SemEval 2022 competition.

## Contents
- `Final.ipynb`: Jupyter notebook containing the final model code and experiments.
- `Community Experiment.ipynb`: Experiments based on community label.
- `DataAnalysisAndBaseline.ipynb`: Initial data analysis and baseline models.
- `report.pdf`: Final report submitted for the coursework.
Overview
The goal is to build a model that outperforms the RoBERTa baseline F1 score of 0.49 on detecting patronizing language.

Our final DeBERTa model achieves a F1 score of 0.58 on the official dev set

## Methodology
### 1) Data Analysis:
We analyzed label distribution, text length correlation, and other data characteristics. We also qualitatively assessed dataset difficulty and subjectivity.
### 2) Modeling:
We experimented with BERT, ALBERT, DistilBERT, BART, and DeBERTa transformers using HuggingFace Transformers. Hyperparameter tuning included tweaking learning rate and training epochs. In addition to the RoBERTa baseline, we created and compared against two simple Bag of Words based models Naive Bayes and SVM which achieved F1-scores of 0.37 and 0.31 respectively. To counter class imbalance, we tried upsampling and downsampling, augmentation through backtranslation, and paraphrasing data augmentation via GPT. We ran experiments as suggested by the task including the use of the community label.
### 3) Analysis:
We analyzed model performance on patronizing language types and input text lengths. We also evaluated model behavior on different target groups in the data. More information can be found in the report.

## Requirements
Python
PyTorch
Pandas
Matplotlib
scikit-learn
Transformers
HuggingFace Datasets
