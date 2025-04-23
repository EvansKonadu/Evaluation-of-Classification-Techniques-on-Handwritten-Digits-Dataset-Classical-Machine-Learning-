# Handwritten Digit Classification: Comparative Analysis of Machine Learning Models

**Author**: Evans Konadu 

**Date**: 2025  

---

## Introduction
This report evaluates eight classification techniques on the Scikit-learn handwritten digits dataset. The models were tested with both default and hyperparameter-tuned configurations, using an 80/20 train-test split. Unnormalised data was selected after preliminary comparisons showed marginally better performance over normalised data (StandardScaler). Key evaluation metrics included accuracy, precision, recall, F1-score, and ROC AUC.

---

## Classification Methods
The following models were implemented and compared:  

- **Logistic Regression [1]**  
- **Gaussian Naïve Bayes [2]**  
- **Random Forest Classifier [3]**  
- **K-Nearest Neighbours (KNN) [4, 9]**  
- **Decision Tree Classifier [5, 12]**  
- **Support Vector Classifier (SVC) [6, 11]**  
- **Gradient Boosting [7]**  
- **Linear Discriminant Analysis (LDA) [8]**  

---
----

## Experimental Setup
An 80:20 train-test split was used. For model evaluation, both default and optimised hyperparameters were considered using GridSearchCV. Overfitting was monitored via cross-validation and evaluation of prediction accuracy.

**1. Cross -Validation Comparison with Test and Training Sets**

![image](https://github.com/user-attachments/assets/f196eab7-944b-4b82-9748-7c53624ab512)

*The bar chart compares the train, cross-validation, and test accuracy of each model in both default and tuned models to help in identifying the best-performing model.*

------
**2. Evaluating All Trained Models to Select the Top 8 Performers (Default vs. Tuned) Using Parity Plots**
- Below is the Comparison of Model Performance: Default vs. Tuned Hyperparameters

![image](https://github.com/user-attachments/assets/48380207-4a42-49e0-8cd1-63986bc514ad)

*Each subplot presents the comparison of the different model performance between default and tuned hyperparameters for each model. The point closest to the reference line (line of best fit) indicates the model with the better performance between the two trained versions (default and optimised)*

------
**3. Distance from Best-Fit Line**
| Classification Model              | Default Models Distance | Tuned Models Distance |
|-----------------------------------|-------------------------|-----------------------|
| Gaussian Naive Bayes              | 0.0145                  | 0.0028                |
| Logistic Regression               | 0.0129                  | 0.0172                |
| Random Forest Classifier          | 0.0196                  | 0.0103                |
| K-Nearest Neighbors               | 0.013                   | 0.0157                |
| Decision Tree                     | 0.1218                  | 0.106                 |
| Support Vector Classifier         | 0.0983                  | 0.0908                |
| Gradient Boosting                 | 0.0334                  | 0.0103                |
| Linear Discriminant Analysis      | 0.0668                  | 0.0049                |

**Selected Models Based On Distances from Best Fit Line**
- **Model:** Gaussian Naive Bayes, **Best Version:** Tuned, **Distance from Best Fit Line:** 0.0028
- **Model:** Logistic Regression, **Best Version:** Default, **Distance from Best Fit Line:** 0.0172
- **Model:** Random Forest Classifier, **Best Version:** Tuned, **Distance from Best Fit Line:** 0.0177
- **Model:** K-Nearest Neighbors, **Best Version:** Default, **Distance from Best Fit Line:** 0.0103
- **Model:** Decision Tree, **Best Version:** Tuned, **Distance from Best Fit Line:** 0.1060
- **Model:** Support Vector Classifier, **Best Version:** Default, **Distance from Best Fit Line:** 0.0083
- **Model:** Gradient Boosting, **Best Version:** Tuned, **Distance from Best Fit Line:** 0.0103
- **Model:** Linear Discriminant Analysis, **Best Version:** Tuned, **Distance from Best Fit Line:** 0.0049

-----------------
-------------

## Model Configurations

### Models with Default Parameters  
*(Best-performing without tuning)*  

| Model                  | Parameters                          |
|------------------------|-------------------------------------|
| Logistic Regression    | Default settings optimal            |
| K-Nearest Neighbours   | Default settings optimal            |
| Support Vector Machine | Default settings optimal            |

### Hyperparameter-Tuned Models  
Key tuned parameters and rationales:  

#### Random Forest Classifier  
| Parameter          | Value  | Rationale                          |
|--------------------|--------|------------------------------------|
| `n_estimators`     | 200    | Increased trees for robustness     |
| `max_depth`        | None   | Full tree growth                   |
| `min_samples_split`| 2      | Minimal samples to split nodes     |

#### Decision Tree Classifier  
| Parameter          | Value  | Rationale                          |
|--------------------|--------|------------------------------------|
| `criterion`        | gini   | Gini index for split quality       |
| `max_depth`        | 10     | Prevent overfitting                |
| `min_samples_split`| 3      | Balance between bias and variance  |

#### Gradient Boosting  
| Parameter             | Value  | Rationale                          |
|-----------------------|--------|------------------------------------|
| `learning_rate`       | 0.05   | Controls tree contribution         |
| `max_depth`           | 2      | Limits complexity                  |
| `n_estimators`        | 8000   | Extensive boosting stages          |

*(Full hyperparameter ranges detailed in Appendix I **Figure H**)*  

------
---

## Results  

### Evaluation Metrics  
**Table 1**: Performance metrics across classifiers (highest values in bold):  

| Model                   | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC AUC    |
|-------------------------|----------|-------------------|-----------|--------|----------|------------|
| **Support Vector (SVC)**| **0.989**| **0.989**         | **0.988** | **0.989** | **0.988** | **0.99998** |
| Gradient Boosting       | 0.983    | 0.985             | 0.984     | 0.985    | 0.984     | 0.99965    |
| K-Nearest Neighbours    | 0.981    | 0.981             | 0.980     | 0.981    | 0.979     | 0.99691    |
| Random Forest           | 0.975    | 0.976             | 0.975     | 0.976    | 0.975     | 0.99967    |
| Logistic Regression     | 0.967    | 0.969             | 0.968     | 0.969    | 0.967     | 0.99913    |
| LDA                     | 0.961    | 0.963             | 0.960     | 0.963    | 0.960     | 0.99894    |
| Gaussian Naïve Bayes    | 0.928    | 0.927             | 0.926     | 0.927    | 0.924     | 0.99608    |
| Decision Tree           | 0.819    | 0.819             | 0.820     | 0.819    | 0.817     | 0.90890    |

---
-----

## Visualisations  

### ROC AUC Scores  
![image](https://github.com/user-attachments/assets/737e023d-ba6d-4280-b0f3-0f154a60bc72)

**Figure 1**: Bar plot displaying ROC AUC scores across classifiers.  

----------

### Confusion Matrices  

![image](https://github.com/user-attachments/assets/2d3e8d43-1293-495f-87e2-e3e9cb814016)

**Figure 2**: Confusion matrices ranked by classifier performance.  

-------------

### ROC Curves  

![image](https://github.com/user-attachments/assets/d110661e-8697-4451-9bcf-7afe2883976c)

**Figure 3**: ROC AUC curves illustrating classifier performance.  

-----
---

## Discussion  
- **Best Model**: The **Support Vector Classifier (SVC)** outperformed others (98.9% accuracy, AUC≈1), aligning with Vapnik’s theory[6] that SVMs excel in high-dimensional spaces by maximising margin boundaries.  

- **Ensemble Methods**: Gradient Boosting (98.3%) and Random Forest (97.5%) demonstrating how ensemble methods reduce variance via aggregation[10]. Hyperparameter tuning (e.g., `n_estimators=8000`) was critical.  
- **Challenges**: Digit "8" was consistently misclassified, particularly by Gaussian Naïve Bayes and Decision Tree, suggesting overlapping features with similar digits.  

---
---

## Dependencies  
- Python 3.x  
- Scikit-learn  
- NumPy, Pandas, Matplotlib  

---
---

## Future Work  
- Explore advanced feature extraction (e.g., CNNs) to address misclassification of ambiguous digits.  
- Test hybrid models combining SVC with ensemble techniques.  

---
-----

## Coding Contributors  
The implementation was a collaborative effort by:  
- Fanny Namondo Ngomba (K2441388)  
- Evans Konadu (K2436512)  
- Ashley Daud (K2441726)  
- Ohemaa Pokuaa Boadu (K2447214)  

*Note: The report is an individual analysis by Evans Konadu.*  

---
----

## References  
[1] D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed. Hoboken, NJ: Wiley, 2013.

[2] F. Sabry, *Naïve Bayes Classifier: Fundamentals and Applications*. One Billion Knowledgeable, 2023. Available: [https://www.google.co.uk/books/edition/Naive_Bayes_Classifier/DPTGEAAAQBAJ?hl=en&gbpv=1](https://www.google.co.uk/books/edition/Naive_Bayes_Classifier/DPTGEAAAQBAJ, vol. 45, no. 1, pp. 4–33, 2001.

[4] F. Sabry, *K Nearest Neighbor Algorithm: Fundamentals and Applications*. One Billion Knowledgeable, 2023. Available: https://books.google.co.uk/books?id=BMrGEAAAQBAJ&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false. [Accessed: 25 Feb. 2025].

[5] J. R. Quinlan, "Induction of decision trees," *Mach. Learn.*, vol. 1, no. 1, pp. 82–107, 1986.

[6] V. Vapnik, *The Nature of Statistical Learning Theory*, New York: Springer, 1995.

[7] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.

[8] G. J. McLachlan, *Discriminant Analysis and Statistical Pattern Recognition*, Hoboken, NJ: Wiley-Interscience, 2004.

[9] T. M. Cover and P. E. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Inf. Theory*, vol. 13, no. 1, pp. 21–27, 1967.

[10] C. Zhang and Y. Ma, *Ensemble Machine Learning: Methods and Applications*. Springer, 2012.

[11] C.-W. Hsu, C.-C. Chang, and C.-J. Lin, "A Practical Guide to Support Vector Classification," 2016. Available: [https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf](https://www.csie.ntu2025].

[12] M. Azad, I. Chikalov, S. Hussain, M. Moshkov, and B. Zielosko, "Decision Trees with Hypotheses for Recognition of Monotone Boolean Functions and for Sorting," *Synthesis Lectures on Intelligent Technologies*, pp. 73–80, 2022. Available: [https://doi.org/10.1007/978-3-031-08585-7_6](https://doi.org/ch. 2025].

[13] Scikit-learn developers, "sklearn.datasets.load_digits," *Scikit-learn Documentation*. [Online]. Available: [load_digits — scikit-learn 1.6.1 documentation](https://scikit-learn25].

------
------

# Appendix I

**Figure H: Summary of Grid Search Tuning Ranges and Baseline Parameter Settings for Classification Models**

| Classifier                  | Tuned Parameters (Range Used for Grid Search)                                                                 | Baseline (Default) Parameter Settings (Scikit-learn Documentation) |
|-----------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| **Logistic Regression**     | • C: [0.001, 0.01, 0.1, 1, 10, 100]                                                                           | • max_iter = 10000<br>• C = 1.0<br>• solver = 'lbfgs'<br>• penalty = 'l2' |
| **Gaussian Naïve Bayes**    | • var_smoothing: np.logspace(-9, 0, 10)                                                                       | • var_smoothing = 1e-9                                             |
| **Random Forest Classifier**| • n_estimators: [50, 100, 200]<br>• max_depth: [None, 10, 20]<br>• min_samples_split: [2, 5, 10]               | • n_estimators = 100<br>• max_depth = None<br>• min_samples_split = 2 |
| **K-Nearest Neighbours**    | • n_neighbors: [3, 5, 7, 9]<br>• weights: ['uniform', 'distance']<br>• algorithm: ['auto', 'ball_tree', 'kd_tree', 'brute']<br>• p: [1, 2] | • n_neighbors = 5<br>• weights = 'uniform'<br>• algorithm = 'auto'<br>• p = 2 |
| **Decision Tree Classifier**| • max_depth: [5, 10, 15, 20, None]<br>• min_samples_split: [3, 5, 10, 20]<br>• criterion: ['gini', 'entropy']<br>• max_features: [None, 'sqrt', 'log2'] | • criterion = 'gini'<br>• max_depth = None<br>• min_samples_split = 2<br>• max_features = None |
| **Support Vector Classifier**| • C: [0.1, 1, 10]<br>• kernel: ['linear', 'rbf', 'poly']<br>• gamma: ['scale', 'auto']                        | • C = 1.0<br>• kernel = 'rbf'<br>• gamma = 'scale'                 |
| **Gradient Boosting Classifier**| • n_estimators: [120, 130, 8000, 10000, 20000]<br>• learning_rate: [0.01, 0.05, 0.1]<br>• max_depth: [1, 2]<br>• max_features: ['sqrt', 'log2']<br>• subsample: [0.5, 0.6, 0.9]<br>• validation_fraction: [0.1]<br>• n_iter_no_change: [10] | • n_estimators = 100<br>• learning_rate = 0.1<br>• max_depth = 3<br>• max_features = None<br>• subsample = 1.0<br>• validation_fraction = 0.1<br>• n_iter_no_change = None |
| **Linear Discriminant Analysis**| • solver: ['svd', 'lsqr', 'eigen']<br>• shrinkage: ['auto', 0.05, 0.01, 0.04, 0.1, 0.5]<br>• n_components: [None, 5, 10, 15] | • solver = 'svd'<br>• shrinkage = None<br>• n_components = None    |
