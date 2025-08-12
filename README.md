# Decision-Trees-and-Random-Forests
# Task 5 — Decision Trees and Random Forests

## Objective
Learn and implement tree-based models (Decision Trees & Random Forests) for classification.

## Dataset
Heart Disease Dataset from Kaggle  
Link: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## Steps Performed
1. Loaded and explored dataset.
2. Split data into train/test sets.
3. Trained a **Decision Tree Classifier** and visualized it using `plot_tree`.
4. Controlled overfitting by limiting tree depth.
5. Trained a **Random Forest Classifier** and compared accuracy.
6. Interpreted **feature importances** using Random Forest.
7. Evaluated both models using **cross-validation**.

## Results
| Model                   | Accuracy | Cross-Validation Accuracy |
|-------------------------|----------|---------------------------|
| Decision Tree           | ~XX%     | ~XX%                      |
| Decision Tree (Limited) | ~XX%     | ~XX%                      |
| Random Forest           | ~XX%     | ~XX%                      |

Random Forest performed better due to ensemble learning and bagging.

## How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn graphviz
python task5.py
