import pandas as pd
import numpy as np
import string
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime



def de_emojify(text):
    regex_pattern = re.compile(pattern = "["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               u"\u23ea"
                           "]+", flags = re.UNICODE)
    return regex_pattern.sub(r'', text)

class engage_pred():
    def __init__(self) -> None:
        self.data = pd.read_csv("ChatGPT_training_first20000_v1.csv")
    
    # def date_process(self):
    #     dates = np.array(self.data["date"])
    #     print(datetime.strptime(dates[0],"%Y-%m-%d"))
    def text_process(self):
        cleaned_text = self.data["tweet"].str.lower()
        cleaned_text = cleaned_text.replace(to_replace=r'rt @.+? ', value="", regex=True)
        cleaned_text = cleaned_text.replace(to_replace=r'@.+? ', value="", regex=True)
        cleaned_text = cleaned_text.replace(to_replace=r'(htpps|http).+? ', value="", regex=True)
        cleaned_text = cleaned_text.replace(to_replace=r'(htpps|http).+', value="", regex=True)
        cleaned_text = cleaned_text.replace(to_replace=r'\^[a-zA-Z]\s+', value=' ', regex=True)
        cleaned_text = cleaned_text.replace(to_replace=r'\s+', value=' ', regex=True)
        

        test = []
        for sentence in cleaned_text:
            for punctuation in string.punctuation:
                if punctuation in sentence:
                    sentence = sentence.replace(punctuation, "")
                    
            test += [sentence]
        
    print(string.punctuation)

x = engage_pred()
# x.date_process()
x.text_process()