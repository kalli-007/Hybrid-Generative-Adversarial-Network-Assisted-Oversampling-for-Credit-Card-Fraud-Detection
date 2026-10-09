# Validated Metrics for the Included Dataset

These results were calculated from the included 5,000-record dataset using the
application's fixed stratified 80% training / 20% testing process. Values are not
hardcoded in the web interface.

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 99.40% | 92.19% | 98.33% | 95.16% | 98.50% |
| Decision Tree | 99.10% | 91.80% | 93.33% | 92.56% | 96.09% |
| Random Forest + SMOTE | 99.90% | 100.00% | 98.33% | 99.16% | 99.62% |
| Gradient Boosting + SMOTE | 99.70% | 96.72% | 98.33% | 97.52% | 98.90% |
| Hybrid GAN + Gradient Boosting | 99.90% | 100.00% | 98.33% | 99.16% | 98.78% |

Different uploaded datasets will produce different measured results. To display
these validated values, upload `sample_data/credit_card_training.csv` and click
**Train all algorithms** in the administrator module.

