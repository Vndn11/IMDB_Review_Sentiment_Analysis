Naïve Bayes Sentiment Analysis on IMDB Review Dataset


1. Introduction
This project aims to implement a Naïve Bayes classifier for sentiment analysis on the IMDB movie reviews dataset. The primary goal is to classify movie reviews as either positive or negative based on their textual content. The Naïve Bayes algorithm, a probabilistic classification technique based on Bayes' Theorem, is used to determine the likelihood of a given review belonging to a specific sentiment category.

The implementation follows a structured approach involving multiple phases, including data preprocessing, feature extraction, model training, evaluation, and real-time user interaction. Various text-processing techniques such as tokenization, stopword removal, lemmatization, and feature engineering are utilized to transform raw text into structured numerical representations suitable for machine learning models. The final model is evaluated based on standard performance metrics such as accuracy, precision, recall, F1-score, and ROC curve analysis.



2. Dataset Overview (Link : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/code)
The dataset used in this project is the IMDB Movie Review Dataset, which consists of large-scale textual data containing reviews of movies along with their corresponding sentiment labels (positive or negative). The dataset is structured into two primary components:

Review Text: This consists of textual reviews written by users about various movies. The reviews contain subjective opinions, making sentiment classification a challenging task.
Sentiment Labels: Each review is labeled as either positive (indicating a favorable review) or negative (indicating an unfavorable review).
The dataset is split into two parts for model training and evaluation:

Training Set: A specified percentage of the dataset (e.g., 80% or 70%) is used for training the Naïve Bayes classifier. This portion is used to learn the underlying patterns in the data.
Test Set: The remaining portion of the dataset (e.g., 20% or 30%) is used to evaluate the trained model. The test set provides an unbiased assessment of how well the model generalizes to unseen data.
This dataset is widely used in sentiment analysis research due to its large size, real-world nature, and inherent challenges in text classification.


3. Preprocessing Steps
Before training the Naïve Bayes model, it is crucial to preprocess the text data to improve classification accuracy. The following preprocessing techniques are applied:



3.1 Lowercasing
To ensure uniformity, all text is converted to lowercase, eliminating any inconsistencies due to case sensitivity.

<img width="299" alt="image" src="https://github.com/user-attachments/assets/316c1259-b957-4b96-a2bc-485c1d7cd047" />

This step helps to standardize the text, ensuring that words like "Great" and "great" are treated as the same word.



3.2 Remove HTML Tags
Many reviews contain HTML tags that are unnecessary for sentiment analysis. Removing them cleans up the text:


<img width="434" alt="image" src="https://github.com/user-attachments/assets/8434158f-12ca-4cc8-8091-af277ce93fa7" />

HTML tags, often found in web-based datasets, do not contribute meaningfully to the sentiment classification task.

3.3 Remove URLs
Since some reviews contain website links (URLs), these need to be removed:

<img width="522" alt="image" src="https://github.com/user-attachments/assets/6674c50f-16e5-4810-b93e-a7c7d622df83" />

URLs do not carry sentiment-related information, so removing them ensures that the model is not distracted by irrelevant text.

3.4 Remove Punctuation
Punctuation marks such as commas, periods, exclamation marks, and question marks are removed since they do not contribute significantly to sentiment meaning:

<img width="425" alt="image" src="https://github.com/user-attachments/assets/4303c958-e036-46fc-8e1e-298a747d351e" />

Removing punctuation helps simplify text analysis, ensuring that words are not split by unnecessary characters.

3.5 Handling Chatwords
Commonly used internet slang and abbreviations are expanded into their full forms to maintain textual clarity:

<img width="373" alt="image" src="https://github.com/user-attachments/assets/2901c0f8-ac89-46bc-96fa-65b11ef735d1" />

Expanding chat words such as "LOL" → "Laughing Out Loud" ensures that the model understands the actual sentiment conveyed in informal text.

3.6 Remove Stopwords
Commonly used stopwords such as "is," "the," "and," are removed since they do not contribute to sentiment:

<img width="470" alt="image" src="https://github.com/user-attachments/assets/fdff3fc3-0eed-4d1b-b242-c305b6565d53" />

Stopword removal helps the model focus on meaningful words rather than frequently occurring function words.

3.7 Remove Emojis
Emojis can sometimes interfere with text analysis, so they are removed:

<img width="356" alt="image" src="https://github.com/user-attachments/assets/c371ab94-f111-4122-8e1e-26864af42b54" />

While emojis may contain sentiment, this model does not process them explicitly, so they are removed for consistency.

3.8 Tokenization and Lemmatization
Each review is split into individual tokens (words), and words are reduced to their root forms to improve model efficiency:

<img width="662" alt="image" src="https://github.com/user-attachments/assets/05221f88-22d4-4bed-ba3a-4435c246f5cc" />

Lemmatization helps to standardize words by converting them to their base forms (e.g., running → run).

4. Feature Engineering
To transform text into numerical values for machine learning, the Bag-of-Words (BoW) representation is used.

4.1 Create Vocabulary
Extracts unique words from all reviews:

<img width="401" alt="image" src="https://github.com/user-attachments/assets/d51656d4-abee-4f8c-8ccb-26ee1af0054d" />


<img width="341" alt="image" src="https://github.com/user-attachments/assets/853ccaa9-252e-45da-ac43-2d31e5028405" />

This vocabulary serves as the reference for feature extraction.


4.2 Create Binary Bag of Words Representation
Converts each review into a sparse binary vector, indicating word presence:

<img width="557" alt="image" src="https://github.com/user-attachments/assets/6b63dcc4-acda-48ae-88f2-16f91da0009b" />


<img width="409" alt="image" src="https://github.com/user-attachments/assets/21bc6d8d-4a0c-44b5-a612-84dcf42b7ece" />

This ensures that each review is converted into a numerical format for the classifier.


5. Training Naïve Bayes Classifier
Once the data is preprocessed and transformed into a numerical format using Binary Bag-of-Words (BoW), we train the Naïve Bayes classifier.
5.1 Overview of Naïve Bayes for Sentiment Analysis
The Naïve Bayes algorithm is a probabilistic classifier based on Bayes' Theorem, assuming that features (words in this case) are independent given the class label. The classifier calculates the probability of a review being positive or negative based on the occurrence of words in the review text.

<img width="520" alt="image" src="https://github.com/user-attachments/assets/497d5bc0-ed23-46c7-8adc-1676e8f95fe1" />


5.2 Implementing Naïve Bayes Training
To train the Naïve Bayes model, we count the occurrences of words in each class and compute probabilities. We use Laplace Smoothing (Add-1 Smoothing) to avoid zero probabilities.

<img width="869" alt="image" src="https://github.com/user-attachments/assets/a1c15aef-9849-4502-8b35-fd50ffb26221" />




In this implementation:

The word occurrences are stored for each class.
Laplace smoothing prevents zero probabilities by initializing word counts to 1 instead of 0.
Log probabilities are used to avoid numerical underflow when multiplying small probabilities.

6. Making Predictions with Naïve Bayes
Once the classifier is trained, it can predict sentiment labels for new reviews based on their word composition.

<img width="520" alt="image" src="https://github.com/user-attachments/assets/96902cdf-0bf1-49ab-9465-65de7c591cda" />


Explanation:
This function calculates log probabilities for each sentiment class.
It selects the class with the highest probability as the predicted label.



7. Model Evaluation
Once the model is trained and predictions are made, we evaluate its performance using various classification metrics.

7.1 Performance Metrics
To assess how well the model classifies reviews, we compute the following metrics:

Accuracy: Overall correctness of the model.
Precision: Correctly predicted positive reviews out of all predicted positives.
Recall (Sensitivity): Correctly identified positive reviews out of all actual positives.
Specificity: Correctly identified negative reviews out of all actual negatives.
F1-Score: Harmonic mean of precision and recall.

<img width="675" alt="image" src="https://github.com/user-attachments/assets/08b60397-1697-4600-bfbd-74c8e75abac8" />

<img width="317" alt="image" src="https://github.com/user-attachments/assets/f6123df2-6760-498f-8187-048cb9451f24" />




7.2 Confusion Matrix Visualization

The confusion matrix provides insight into correct and incorrect predictions.


<img width="515" alt="image" src="https://github.com/user-attachments/assets/38f75ff6-bef5-4cb1-bced-8d7f58d2e3c4" />

<img width="451" alt="image" src="https://github.com/user-attachments/assets/111933e1-d5c5-41d6-817e-52c9fa435f87" />



7.3 ROC Curve
The ROC Curve (Receiver Operating Characteristic) shows the trade-off between the true positive rate (TPR) and false positive rate (FPR).

<img width="584" alt="image" src="https://github.com/user-attachments/assets/82d7f727-3b23-4334-8e30-6820c6a87cd3" />

<img width="492" alt="image" src="https://github.com/user-attachments/assets/87808ba1-25e4-407b-bd76-bb491caf04ef" />



8. User Interaction for Real-Time Sentiment Prediction
The model allows users to input custom reviews and get real-time sentiment predictions.

<img width="556" alt="image" src="https://github.com/user-attachments/assets/9f323928-e0a2-4da4-b69b-c0bbd292a3dd" />



9. Summary of Results
The Naïve Bayes classifier successfully classifies IMDB movie reviews with an accuracy of approximately 85.99%. The key observations are:

The model performs well in distinguishing positive and negative reviews.
Precision and recall values indicate a well-balanced classification approach.
ROC curve analysis confirms that the model has a strong discriminatory ability.
Training with 70% of the data provided slightly better performance than training with 80%, indicating that too much training data may introduce noise.
Conclusion
This implementation demonstrates the effectiveness of Naïve Bayes for sentiment analysis. With proper text preprocessing, feature extraction, and probability-based classification, the model successfully analyzes the sentiment of IMDB reviews. Future improvements could include handling negations, incorporating TF-IDF, or using deep learning models for comparison.



















