# Spam Filter Project

## Overview
This project implements a spam filter using a Naïve Bayes classifier. It processes and classifies text messages as either "Spam" or "Not Spam" using natural language processing (NLP) techniques.

## Features
- Loads and preprocesses a labeled dataset of text messages
- Converts text into numerical representations using CountVectorizer and TfidfTransformer
- Trains a Multinomial Naïve Bayes model
- Evaluates the model with accuracy and classification reports
- Allows testing of new messages for spam detection

## Installation
### Prerequisites
Ensure you have Python installed on your machine. Install the required dependencies using:

```sh
pip install pandas numpy scikit-learn
```

## Usage
1. Clone this repository:
   ```sh
   git clone https://github.com/YanaYerokhina/Spam-Filter.git
   cd Spam-Filter
   ```
2. Ensure you have a dataset named `spam.csv` in the same directory.
3. Run the script:
   ```sh
   python spam_filter.py
   ```
4. The program will display model accuracy and classification metrics.
5. Test custom messages to check if they are classified as spam or not.

## Example Output
- Model accuracy score
- Precision, recall, and F1-score for spam classification
- Classification results for test messages

## Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

