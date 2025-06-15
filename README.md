# ML-Phisher-URL-Safety-Classifier
PhishGuard is a machine learning-based project designed to detect phishing websites by analyzing URLs and extracting key features. Using a Random Forest Classifier trained on a phishing dataset, the system accurately classifies URLs as either legitimate or phishing. 

🔐 Phishing URL Detection Using Machine Learning
This project focuses on detecting phishing URLs using a supervised machine learning model. The system is trained on a dataset containing both legitimate and phishing URLs, which are represented by numerical features. A Random Forest classifier is used to predict whether a given URL is legitimate (good) or phishing (bad).

📂 Dataset
The dataset used: PhiUSIIL_Phishing_URL_Dataset.csv
Features include:

Length: Length of the URL

NumDots: Number of dots in the URL

HasAt: Presence of @ symbol

HasHttps: Presence of https

HasHttp: Presence of http

NumDigits: Number of digits in the URL

NumSpecialChar: Number of special characters

label: Class label (phishing or legitimate)

💡 Features
Feature engineering from URL string

Label encoding

Random Forest classifier

Accuracy and classification report output

Real-time prediction by user input

🛠️ Tech Stack
Python 🐍

pandas

NumPy

scikit-learn

Regular Expressions (re)

