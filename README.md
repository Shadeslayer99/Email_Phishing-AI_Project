# Phishing Email Detection

## Project Overview

### Objective

Develop a robust system for detecting phishing emails using advanced natural language processing (NLP) techniques and machine learning algorithms to mitigate phishing threats by accurately identifying phishing attempts from email text data.

### Key Components

1. **Data Preprocessing:**
   - Text normalization, cleaning, tokenization, stop word removal, and removal of URLs and HTML tags

2. **Feature Extraction:**
   - Word embeddings (TF-IDF, Word2Vec, GloVe) and sequence padding

3. **Model Training:**
   - Machine learning models: Logistic Regression, SVM, Random Forest
   - Deep learning models: RNN, LSTM

4. **Model Evaluation:**
   - Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
   - Cross-validation for robustness


### Technical Stack

- **Programming Language:** Python 3.6+
- **Libraries and Frameworks:** TensorFlow, Keras, NLTK, Scikit-learn, NumPy, Pandas
- **Development Environment:** Jupyter Notebook
- **Version Control:** Git

This project demonstrates proficiency in data preprocessing, feature engineering, model training, evaluation, and deployment, showcasing a comprehensive approach to solving real-world problems using state-of-the-art NLP and machine learning techniques.

## Installation and Setup

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/Shadeslayer99/Email_Phishing-AI_Project/edit/main/README.md
    ```

2. **Install Dependencies:**

    Make sure you have Python 3.6+ installed. You can install the required packages using pip:

    ```sh
    pip install -r requirements.txt
    ```

    Note: Create a `requirements.txt` file listing all the necessary packages, such as TensorFlow, NLTK, and any other dependencies used in the notebook.

3. **Download Pre-trained Model:**

    Ensure you have the pre-trained model file used in the notebook. Update the path to the model in the notebook if necessary.

## Usage

1. **Run the Notebook:**

    Open the `SCAI1Master.ipynb` notebook in Jupyter and run all the cells to preprocess the text and detect phishing emails.

2. **Text Preprocessing Steps:**

    The notebook includes the following preprocessing steps:
    - Lowercasing the text
    - Removing special characters and numbers
    - Tokenizing the text
    - Removing stop words
    - Removing URLs and HTML tags

3. **Phishing Detection:**

    The preprocessed text is then fed into a pre-trained model to detect phishing emails. The model outputs a binary classification (1 for phishing, 0 for non-phishing).

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a Pull Request.

## Author

- Anish Kondamadugula

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
