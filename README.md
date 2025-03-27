# Cybersecurity-Threat-Classification-Using-Machine-Learning


# Cybersecurity Threat Classification Using Machine Learning

This project implements a machine learning solution for classifying cybersecurity threats based on network traffic data. It follows a complete pipeline from data preprocessing to model evaluation and visualization.

## Requirements

- Python 3.6 or higher
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

You can install all required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

This project uses network intrusion detection data. The default filename in the code is `network_intrusion_detection.csv`. You should update this to match your actual dataset file.

## How to Run the Code

### Option 1: Jupyter Notebook

1. Make sure you have Jupyter Notebook installed:
   ```bash
   pip install jupyter
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open the `cybersecurity_threat_classification.ipynb` file in Jupyter

4. Update the dataset filename in the first code cell to match your dataset

5. Run each cell in order by selecting it and pressing Shift+Enter

### Option 2: Python Script

If you prefer to run the script directly:

1. Update the dataset filename in the script:
   ```python
   data = pd.read_csv('your_dataset_filename.csv')
   ```

2. Run the script:
   ```bash
   python cybersecurity_threat_classification.py
   ```

## Understanding the Code

The code implements the following steps:

1. **Data Loading**: Loads the dataset
2. **Data Preprocessing**:
   - Handles missing values
   - Converts categorical features to numeric
   - Converts continuous target values to discrete classes
3. **Feature Selection**: Identifies the most important features
4. **Feature Normalization**: Scales the selected features
5. **Model Training**: Trains Random Forest and SVM models
6. **Model Evaluation**: Calculates accuracy, precision, recall, and F1-score
7. **Visualization**: Creates comparison charts and confusion matrices

## Output

The code will generate:

1. Printed evaluation metrics in the console/notebook
2. Three visualizations:
   - Model performance comparison chart
   - Confusion matrix for the Random Forest model
   - Feature importance chart

## Troubleshooting

- **"Unknown label type: continuous"**: This error occurs if you're trying to use classification models on continuous target values. The code includes a step to convert continuous values to discrete classes.
- **Memory issues**: If you encounter memory problems with large datasets, try reducing the number of features or using a smaller subset of the data for training.
- **Long training time**: SVM can be slow on large datasets. Consider using a smaller subset or increasing the `max_iter` parameter.

