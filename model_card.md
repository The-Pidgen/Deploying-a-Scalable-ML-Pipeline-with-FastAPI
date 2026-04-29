# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classification model built using a Logistic Regression algorithm from
scikit-learn. It predicts whether an individual earns more than $50k per year based on census
demographic data. This model is trained using a processed dataset with categorical variables
encoded using OneHotEncoder and labels binarized using LabelBinarizer.
## Intended Use
This model is intended for educational and demonstration purposes as part of a machine learning
pipeline project. It is designed to classify income levels based on data features and should
not be used for production or real-world decision making.
## Training Data
The model is trained on the Census Income dataset (census.csv). The dataset contains demographic
information such as age, occupation, and work hours per week. The training data is split into
training and testing sets using an 80/20 split. 
## Evaluation Data
The evaluation data consists of the held-out test split (20% of the dataset). This dataset was
not used during model training and is used to assess model performance on unseen data.
## Metrics

The model is evaluated using precision, recall, and F1-score. 
On the test dataset, the model achieved:
    Precision: 0.7351
    Recall: 0.5563
    F1 Score: 0.6333

Additionally, perfromance was evaluated across categorical slices of data to assess fairness
and consistency across subgroups.

## Ethical Considerations
The dataset contains demographic information that may reflect historical biases. As a result,
model predictions may unintentionally reinforce or reflect existing societal biases. Care should
be taken when interpreting results, especially for sensitive attributes such as race, sex, or
native country.
## Caveats and Recommendations
The model is a relatively simple Logistic Regression classifier and may not capture complex
relationships in the data. Performance varies across different categorical slices, indicating 
potential bias and uneven performance across subgroups. Future improvements could include more
advanced models, feature engineering, and bias mitigation techniques.