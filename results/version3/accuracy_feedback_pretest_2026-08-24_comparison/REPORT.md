# Comparison of the two fresh snapshots

Both snapshots used the same frozen real-only and +10% GAN ensembles and unchanged thresholds. They share no exact URLs or hostnames.

The GAN model improved accuracy, precision, recall, F1 and PR-AUC and reduced FPR/FNR in both independent cohorts. The strict augmentation rule failed on the 5% cohort only because ROC-AUC changed by -0.13%, then passed on the 10% cohort where ROC-AUC changed by +0.41%.

The second cohort should not be interpreted as proof that increasing phishing prevalence improves the model. It uses a different phishing slice: 1,000 still-online PhishTank URLs verified June 16–25, while the first cohort used newer eligible OpenPhish/PhishTank URLs. The difference in recall therefore reflects sample distribution, not prevalence alone.
