# House Price Regression Exercise

This is a practice project for learning linear regression with a Kaggle dataset.

## Project Structure

```
Houses_regression/
│
├── data/                   # Place your dataset files here
│   └── README.md          # Instructions for data
│
├── notebooks/             # Jupyter notebooks for analysis
│   └── house_price_regression.ipynb  # Main exercise notebook
│
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── model.py           # Model building and training
│   ├── preprocessing.py   # Data preprocessing functions
│   └── visualization.py   # Visualization functions
│
├── tests/                 # Test files
│   ├── __init__.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## Getting Started

### 1. Setup Environment

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Data

1. Download a house prices dataset from Kaggle (e.g., "House Prices - Advanced Regression Techniques")
2. Place the CSV file(s) in the `data/` directory
3. See `data/README.md` for more details

### 3. Start Working

You can work in two ways:

**Option A: Jupyter Notebook (Recommended for beginners)**
```bash
jupyter notebook notebooks/house_price_regression.ipynb
```

**Option B: Python Scripts**
- Implement functions in the `src/` modules
- Run your model with `python src/model.py`

### 4. Complete the Exercise

Follow the TODO comments in the code to:
- Load and explore the data
- Preprocess and clean the data
- Build a linear regression model
- Train and evaluate the model
- Visualize the results

## Learning Objectives

By completing this exercise, you will learn:
- How to load and explore datasets with pandas
- Data preprocessing and feature engineering
- Building linear regression models with scikit-learn
- Evaluating model performance
- Creating visualizations with matplotlib/seaborn

## Notes

- All data files (*.csv) are gitignored to avoid committing large files
- The code structure is intentionally empty - fill in the TODOs as you learn
- Write tests for your functions in the `tests/` directory

## Resources

- [Scikit-learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Good luck with your exercise!
