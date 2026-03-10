# StyleSense: Fashion Forward Forecasting

**StyleSense** is an end-to-end machine learning pipeline designed to predict customer recommendations for an online women's clothing retailer. By analyzing customer reviews, product categories, and demographic data, the system automatically determines whether a customer is likely to recommend a product.

The repository features a heterogeneous data pipeline capable of processing text, categorical, and numerical data simultaneously. Key highlights include:
* **NLP Integration:** Efficient tokenization and lemmatization using SpaCy.
* **Feature Engineering:** Custom transformers for text-length features and linguistic attributes.
* **Model Optimization:** XGBoost classifier tuned via `RandomizedSearchCV`.

---

## Getting Started

These instructions will help you set up a copy of the project on your local machine for development and testing.

### Dependencies

To run this project, you will need **Python 3.x** and the following libraries:
* `pandas`
* `scikit-learn`
* `xgboost`
* `spacy`
* `joblib`

### Installation

1.  **Clone the repository** to your local machine.
2.  **Navigate to the project directory** in your terminal.
3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the SpaCy language model**:
    ```bash
    python -m spacy download en_core_web_sm
    ```
5.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

---

## Usage & Testing

Testing is performed by executing the notebook cells sequentially within the `starter/` directory.

### Pipeline Breakdown
1.  **Base Model Testing:** Run the "Training Pipeline" section to verify that the `ColumnTransformer` (integrating the SpaCy processor) trains the base model.
2.  **Optimization Testing:** Run the "Fine-Tuning Pipeline" section to execute the `RandomizedSearchCV` and evaluate the F1-Macro score.

### Reproducing Results
1.  Open `starter/starter.ipynb`.
2.  **Run all cells** from top to bottom.
3.  The serialized models will be automatically exported to the `models/` directory:
    * `models/base_xgboost_pipeline.pkl`
    * `models/tuned_xgboost_pipeline.pkl`

---

## Project Structure

```text
├── starter/
│   ├── starter.ipynb        # Main notebook (EDA, architecture, training)
│   └── data/
│       └── reviews.csv      # Anonymized customer reviews dataset
├── models/                  # Output directory for trained .pkl files
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

```

---

## Built With

* [Scikit-Learn](https://scikit-learn.org/) - Pipelines and hyperparameter tuning.
* [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting framework.
* [spaCy](https://spacy.io/) - Industrial-strength Natural Language Processing.
* [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis.

---

## License

This project is licensed under the MIT License - see the `LICENSE.txt` file for details.
