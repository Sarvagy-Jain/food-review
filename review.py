import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')


def load_count_vectorizer(filename):
    with open(filename, 'rb') as cv_file:
        saved_data = pickle.load(cv_file)
        cv = CountVectorizer(vocabulary=saved_data['vocabulary'])
        cv.__dict__.update(saved_data)
        return cv

# Load the pre-fitted CountVectorizer and model
with open('model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

example_filename = 'your_count_vectorizer.pkl'
cv = load_count_vectorizer(example_filename)

# Streamlit UI
st.title('Restaurant Review System')

# Input for user review
sample_review = st.text_area('Enter your restaurant review here:')

if st.button('Predict'):
    if sample_review:
        sample_review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sample_review)
        sample_review=sample_review.lower()
        sample_review_words=sample_review.split()
        review_words=[word for word in sample_review_words if not word in set(stopwords.words('english'))]
        ps=PorterStemmer()
        final_review=[ps.stem(word) for word in sample_review_words]
        final_review=' '.join(final_review)
        temp=cv.transform([final_review]).toarray()
        # Make prediction using the loaded model
        prediction = model.predict(temp)

        # Display the result
        if prediction[0] == 1:
            st.success('Positive Review!')
        else:
            st.error('Negative Review!')
    else:
        st.warning('Please enter a review before predicting.')


# In[ ]:




