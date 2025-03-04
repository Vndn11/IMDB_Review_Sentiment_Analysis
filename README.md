Naïve Bayes Sentiment Analysis on IMDB Review Dataset


1. Introduction
This project implements a Naïve Bayes classifier for sentiment analysis on the IMDB movie reviews dataset. The classifier determines whether a review is positive or negative based on text-based feature extraction and probability-based classification. The implementation follows a step-by-step text preprocessing, feature engineering, model training, and evaluation process.

2. Dataset Overview
The dataset used is the IMDB Dataset.csv, which consists of movie reviews along with their sentiment labels (positive/negative).
Training Set: A user-defined percentage (e.g., 80% or 70%) of the dataset.
Test Set: The remaining portion of the dataset (e.g., 20% or 30%).

3. Preprocessing Steps
To prepare the text for analysis, several preprocessing steps are applied:

3.1 Lowercasing
All text is converted to lowercase to maintain uniformity:

<img width="299" alt="image" src="https://github.com/user-attachments/assets/316c1259-b957-4b96-a2bc-485c1d7cd047" />

3.2 Remove HTML Tags
Removes unnecessary HTML tags:

<img width="434" alt="image" src="https://github.com/user-attachments/assets/8434158f-12ca-4cc8-8091-af277ce93fa7" />

3.3 Remove URLs
Eliminates hyperlinks:

<img width="522" alt="image" src="https://github.com/user-attachments/assets/6674c50f-16e5-4810-b93e-a7c7d622df83" />

3.4 Remove Punctuation
Removes all punctuation marks to clean text:

<img width="425" alt="image" src="https://github.com/user-attachments/assets/4303c958-e036-46fc-8e1e-298a747d351e" />

3.5 Handling Chatwords
Common internet slang (chatwords) are replaced with their full forms:

<img width="373" alt="image" src="https://github.com/user-attachments/assets/2901c0f8-ac89-46bc-96fa-65b11ef735d1" />

3.6 Remove Stopwords
Common stopwords (e.g., "is", "the", "and") are removed using NLTK:

<img width="470" alt="image" src="https://github.com/user-attachments/assets/fdff3fc3-0eed-4d1b-b242-c305b6565d53" />

3.7 Remove Emojis
Removes emoji characters:

<img width="356" alt="image" src="https://github.com/user-attachments/assets/c371ab94-f111-4122-8e1e-26864af42b54" />

3.8 Tokenization and Lemmatization
Each review is split into words (tokens) and converted to their root form:

<img width="662" alt="image" src="https://github.com/user-attachments/assets/05221f88-22d4-4bed-ba3a-4435c246f5cc" />

4. Feature Engineering
To convert text into a numerical representation, Bag-of-Words (BoW) is used.

4.1 Create Vocabulary
Extracts unique words from the dataset:

<img width="401" alt="image" src="https://github.com/user-attachments/assets/d51656d4-abee-4f8c-8ccb-26ee1af0054d" />


<img width="341" alt="image" src="https://github.com/user-attachments/assets/853ccaa9-252e-45da-ac43-2d31e5028405" />


4.2 Create Binary Bag of Words Representation
Converts each review into a sparse binary vector:

<img width="557" alt="image" src="https://github.com/user-attachments/assets/6b63dcc4-acda-48ae-88f2-16f91da0009b" />


<img width="409" alt="image" src="https://github.com/user-attachments/assets/21bc6d8d-4a0c-44b5-a612-84dcf42b7ece" />


5. Training Naïve Bayes Classifier
Using Bayes' Theorem, the probability of a class given the words in the review is calculated:

<img width="869" alt="image" src="https://github.com/user-attachments/assets/a1c15aef-9849-4502-8b35-fd50ffb26221" />

<img width="520" alt="image" src="https://github.com/user-attachments/assets/96902cdf-0bf1-49ab-9465-65de7c591cda" />


6. Model Evaluation
6.1 Calculate Metrics:

<img width="675" alt="image" src="https://github.com/user-attachments/assets/08b60397-1697-4600-bfbd-74c8e75abac8" />

<img width="317" alt="image" src="https://github.com/user-attachments/assets/f6123df2-6760-498f-8187-048cb9451f24" />


6.2 Confusion Matrix:

<img width="515" alt="image" src="https://github.com/user-attachments/assets/38f75ff6-bef5-4cb1-bced-8d7f58d2e3c4" />

<img width="451" alt="image" src="https://github.com/user-attachments/assets/111933e1-d5c5-41d6-817e-52c9fa435f87" />


6.3 ROC Curve:

<img width="584" alt="image" src="https://github.com/user-attachments/assets/82d7f727-3b23-4334-8e30-6820c6a87cd3" />

<img width="492" alt="image" src="https://github.com/user-attachments/assets/87808ba1-25e4-407b-bd76-bb491caf04ef" />


7. Real-Time Sentiment Prediction
Allows user input to classify sentences in real-time:

<img width="556" alt="image" src="https://github.com/user-attachments/assets/9f323928-e0a2-4da4-b69b-c0bbd292a3dd" />


8. Summary
-> Preprocessing: Cleaned text data.
-> Feature Engineering: Converted text into a binary bag-of-words.
-> Model Training: Used Naïve Bayes classifier with Laplace smoothing.
-> Evaluation: Achieved 85.99% accuracy with good sensitivity and specificity.
-> User Interaction: Allowed real-time predictions for new sentences.



















