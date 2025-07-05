# 🧠 **Comment Categorization & Reply Assistant Tool**

A Natural Language Processing (NLP) mini project that classifies user comments (e.g., from social media or product announcements) into actionable categories like praise, hate, constructive feedback, spam, emotional, etc.

## 🚀 Features

### _Classifies text comments into 8 categories:_

* Praise

* Support

* Constructive Criticism

* Hate/Abuse

* Threat

* Emotional

* Irrelevant/Spam

* Question/Suggestion

#### Suggests automatic replies for each type

#### Upload your own comment CSVs for batch processing

#### Visualizes distribution of classified comments

#### Built with Python, Streamlit, and scikit-learn

## 🏗️ Project Structure

#### comment_classifier_deployable/
>── app.py                 #Streamlit frontend script
>
>── models/                #Pre-trained model + TF-IDF vectorizer
>
>>   ── logistic_model.pkl
>
>>   ── tfidf_vectorizer.pkl
>
>── data.csv               #Sample dataset (comment, label)

## 💻 How to Run Locally

#### 1. Clone the Repo or Unzip Project
   
`cd path/to/comment_classifier_deployable`


#### 2. (Optional) Create Virtual Environment

`python -m venv venv`

* Activate (Windows):  
`venv\Scripts\activate`

* Activate (Mac/Linux):  
`source venv/bin/activate`

#### 3. Install Dependencies

`pip install streamlit pandas scikit-learn joblib spacy matplotlib nltk`  
`python -m nltk.downloader stopwords`  
`python -m spacy download en_core_web_sm`  

#### 4. Run the Streamlit App

Open Command Prompt/ Terminal

`streamlit run app.py`

## 📊 Example CSV Format for Upload  
comment  
`"Amazing work!"`  
`"This is terrible."`  
`"Can you make one about AI?`  

##💡 Future Improvements  
* Add multilingual support

* Deploy to Streamlit Cloud or Hugging Face Spaces

* Integrate OpenAI API for smart replies

## 🧑‍💻 Author

___Kajol Dalmia___

## 📜 License

This project is for educational purpose only.



