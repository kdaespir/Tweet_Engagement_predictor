import pandas as pd
import numpy as np
import string
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, get_scorer_names
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime
import pickle as pkl
import matplotlib.pyplot as plt


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
    def __init__(self, dataset) -> None:
        # Loads the dataset
        self.data = pd.read_csv(dataset)

        # Replaces the NaN values in the location data with unknown, for encoding purposes
        self.data["location"] = self.data["location"].replace(np.nan, "Unknown")
        
        # Encodes each of the differnet locations listed in the location data
        enc = OrdinalEncoder()
        enc_loc = enc.fit_transform(np.array(self.data["location"]).reshape(-1,1))
        self.data["location_coded"] = enc_loc

        # Encodes each of the users as a numerical value
        enc_user = enc.fit_transform(np.array(self.data["user"]).reshape(-1,1))
        self.data["user_coded"] = enc_user

        # places the text data in tweet in a seperate variable for later use
        self.cleaned_text = self.data["tweet"]

    def feat_creation(self):
        # Creates a feature based on the length of the tweet
        tweet_len = [len(text) for text in self.data["tweet"]]

        # Creates a feature based on the length of the users description
        user_desc_len = [len(str(text)) for text in self.data["userdescription"]]

        #Creates a featrue based on the number of hashtags in the tweet
        num_hasht = [text.count("#") for text in self.data["tweet"]]

        #Creates a feature based on the number of @s found in the tweet
        num_ats = [text.count("@") for text in self.data["tweet"]]

        # Creates a feature which is the differnece between the number of days since minus the account creation date
        date_differ = []
        for date, user_create in zip(self.data["date"], self.data["usercreated"]):
            post_date = datetime.strptime(date[:10], "%Y-%m-%d")
            user_date = datetime.strptime(user_create[:10], "%Y-%m-%d")
            difference = post_date - user_date
            date_differ += [difference.days]

        # Stores each of the created features in the dataset
        self.data["tweet_length"] = tweet_len
        self.data["user_desc_length"] = user_desc_len
        self.data["number of #"] = num_hasht
        self.data["number of @s"] = num_ats
        self.data["postdate-usercreate"] = date_differ

        pass
    def mean_engagement_add(self, training_data, test_data, mode):
        # This function finds the mean engagement for each user in the training data and creates the feature in the training dataset.
        # For the testing dataset this function takes the mean engagement for each user in the training dataset references the test dataset
        # if a user in the test dataset is also in the training dataset the the mean enagement score from the training dataset is used in the test dataset.
        # if not, then the mean engagment for the whole training dataset is ussed instead

        # loads the training and testing data
        training = pd.read_csv(training_data)
        testing = pd.read_csv(test_data)

        # iteratres through each row in the dataset and stores the engagement score for a tweet in the posters directory
        user_engage = {}
        for user, engagement in zip(training["user"], training["log_engagements"]):
            if user not in user_engage:
                user_engage[user] = [engagement]
            else:
                user_engage[user] += [engagement]
        
        # iterates through each row in the dataset and calculates the users mean engagement score
        mean_engage_by_row = [np.mean(user_engage[user]) for user in training["user"]]

        # creates a new feature which is the users mean engagement score in the dataset
        self.data["user_mean_engagement"] = mean_engage_by_row


        if mode == "export":
            # checks for each unique user in the training dataset and creates a directory made of of mean user scores and stores the scores in
            # a seperate list
            mean_engage_by_user = {}
            mean_engage_datset = []
            for user in self.data["user"].unique():
                mean_engage_by_user[user] = np.mean(user_engage[user])
                mean_engage_datset += [np.mean(user_engage[user])]

            # calcluates the mean engagement for the dataset
            mean_engage_datset = np.mean(mean_engage_datset)

            # creates a list of the unique users in the training dataset
            users_in_training = list(mean_engage_by_user.keys())

            # checks if each user in the testing dataset is found in the training dataset. if so that users mean engagment score
            # in the training data is added to a list test_data_mean_engage_row. if not then the mean engagment score of the entire
            # training dataset is add to the list
            test_data_mean_engage_row = []
            for user in testing["user"]:
                if user in users_in_training:
                    test_data_mean_engage_row += [mean_engage_by_user[user]]
                else:
                    test_data_mean_engage_row += [mean_engage_datset]

            # creates the mean engagment score feature in the testing dataset
            testing["user_mean_engagement"] = test_data_mean_engage_row
            
            # creates the mean engagment score feature for the training dataset 
            training["user_mean_engagement"] = mean_engage_by_row

            # outputs the alteretd testing dataset to a new csv file 
            # testing.to_csv("ChatGPT_test_v1_w_engage.csv", index=False)

    def text_process(self, feature):
        # loads the data
        _ = self.data[feature]

        # removes non ascii characters
        printable = set(string.printable)
        _ = ["".join(filter(lambda x: x in printable, text)) for text in _]
        
        # removes new lines
        self.cleaned_text = [x.replace("\n","") for x in _]
        
        # converts the list of texts to a series
        self.cleaned_text = pd.Series(self.cleaned_text)
        
        # Removes all retweets at other users
        # self.cleaned_text = self.cleaned_text.replace(to_replace=r'rt @[A-Za-z0-9_]+', value="", regex=True)

        # Removes all @s that are not retweets
        # self.cleaned_text = self.cleaned_text.replace(to_replace=r'@[A-Za-z0-9_]+', value="", regex=True)

        # Removes all hashtags from the texts
        # self.cleaned_text = self.cleaned_text.replace(to_replace=r'#[A-Za-z0-9_]+', value="", regex=True)
        
        # Removes all links that are found in the beginning or middle of the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+? ', value="", regex=True)
        
        # Removes all links that are found at the end of the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'(htpps|http).+', value="", regex=True)

        #removes all single characters from the text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'\^[a-zA-Z]\s+', value=' ', regex=True)

        # Removes all numeric characters from the texts
        self.cleaned_text = self.cleaned_text.replace(to_replace=r"[0-9]+", value="", regex=True)

        # substitutes multiple blank characters with a single one
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'\s+', value=' ', regex=True)
        
        # Removes beginning of text, if the text begins with a blank charcter
        self.cleaned_text = self.cleaned_text.replace(to_replace=r"(^ +)", value="", regex=True)

        # Removes punctuation from text
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'[^\w\s]', value="", regex=True)
        
        #Removes Underscores
        self.cleaned_text = self.cleaned_text.replace(to_replace=r'[_]+', value="", regex=True)
    

        # Coverts all the texts into word tokens
        text_tokens = [word_tokenize(sentence) for sentence in self.cleaned_text]
        
        # removes all stop words from the tokens
        remove_stops = [word for word in text_tokens if not word in stopwords.words()]
        

        # Joins all the tokenized words back into a sentence
        process_words = [" ".join(words) for words in remove_stops]
        
        # replaces the tweets in the dataframe with the processed tweets
        self.data[feature] = process_words

    def xy_split(self, mode):
        if mode == "training":
            self.datax = self.data.drop(["date", "tweet", "url", "user", "usercreated", "location", "userdescription", "log_engagements"], axis=1)
            self.datay = self.data["log_engagements"]
        if mode == "testing":
            self.datax = self.data.drop(["date", "tweet", "url", "user", "usercreated", "location", "userdescription"], axis=1)
        

    def feature_sel(self):
        model = SelectKBest(f_classif)
        anova_feat = model.fit(self.datax, self.datay)
        
        scores = anova_feat.scores_
        anovaf_scores = list(zip(scores, self.datax.columns))
        self.anovaf_incl = sorted(anovaf_scores, key= lambda tup: tup[0], reverse=True)
        print(self.anovaf_incl)

    def drop_unimp_feat(self):
        self.datax = self.datax.drop(["user_coded", "location_coded"], axis=1)

    def export_df(self):
        _ = pkl.dump(self.datax,open("pc4_datax.pkl", "wb"))

    def vectorize(self):
        tfidf = TfidfVectorizer(binary=True, stop_words="english")

    def training_testing(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.datax, self.datay, random_state=0, test_size=0.3)

    

    def models(self, method, model):

        rf = RandomForestRegressor()
        boost = GradientBoostingRegressor()
        svm = SVR()
        if method == "cross_val":
            cv = KFold(10)
            rf_cross_val_mse = cross_val_score(rf, self.datax, self.datay, cv=cv, scoring="neg_mean_squared_error")
            rf_cross_val_r2 = cross_val_score(rf, self.datax, self.datay, cv=cv, scoring="r2")
            rf_cv_score_mse = np.mean(abs(rf_cross_val_mse))
            rf_cv_score_r2 = np.mean(rf_cross_val_r2)

            boost_cross_val_mse = cross_val_score(boost, self.datax, self.datay, cv=cv, scoring="neg_mean_squared_error")
            boost_cross_val_r2 = cross_val_score(boost, self.datax, self.datay, cv=cv, scoring="r2")
            boost_cv_score_mse = np.mean(abs(boost_cross_val_mse))
            boost_cv_score_r2 = np.mean(boost_cross_val_r2)

            svm_cross_val_mse = cross_val_score(svm, self.datax, self.datay, cv=cv, scoring="neg_mean_squared_error")
            svm_cross_val_r2 = cross_val_score(svm, self.datax, self.datay, cv=cv, scoring="r2")
            svm_cv_score_mse = np.mean(abs(svm_cross_val_mse))
            svm_cv_score_r2 = np.mean(svm_cross_val_r2)

            with open("PC4_cv_results_meanTest.txt", "w") as f:
                f.writelines(f"The scores for the models are:\nRF:\nMSE: {rf_cv_score_mse}\nr2: {rf_cv_score_r2}\n\n\
                    \nBoost:\nMSE: {boost_cv_score_mse}\nr2: {boost_cv_score_r2}\n\nSVM:\nMSE: {svm_cv_score_mse}\nr2: {svm_cv_score_r2}")
            exit()
        
        if method == "training":
            rf.fit(self.xtrain, self.ytrain)
            rf_pred = rf.predict(self.xtest)
            rf_r2 = r2_score(self.ytest, rf_pred)
            rf_mse = mean_squared_error(self.ytest, rf_pred)

            
            boost.fit(self.xtrain, self.ytrain)
            boost_pred = boost.predict(self.xtest)
            boost_r2 = r2_score(self.ytest, boost_pred)
            boost_mse = mean_squared_error(self.ytest, boost_pred)

            
            svm.fit(self.xtrain, self.ytrain)
            svm_pred = svm.predict(self.xtest)
            svm_r2 = r2_score(self.ytest, svm_pred)
            svm_mse = mean_squared_error(self.ytest, svm_pred)

            _ = pd.Series(rf_pred).to_csv("output_training_PC4.csv", index=False)

            with open("training_results.txt", "w") as f:
                f.writelines(f"Random forest:\nR2: {rf_r2}\nMSE: {rf_mse}\n\nBoosting:\nR2: {boost_r2}\n\
                            MSE: {boost_mse}\n\nSVM:\nR2: {svm_r2}\nMSE: {svm_mse}")
                f.close()
            exit()

        if method == "export_training_model":
            rf.fit(self.datax, self.datay)
            boost.fit(self.datax, self.datay)
            svm.fit(self.datax, self.datay)

            
            _ = pkl.dump(rf, open("rf_ChatGPT_PC4_model.pkl", "wb"))
            _ = pkl.dump(boost, open("boost_ChatGPT_PC4_model.pkl", "wb"))
            _ = pkl.dump(svm, open("svm_ChatGPT_PC4_model.pkl", "wb"))

            exit()

        if method == "pred_output":
            model = pkl.load(open(f"{model}_ChatGPT_PC4_model.pkl", "rb"))
            datax = pkl.load(open("pc4_datax.pkl", "rb"))
            # datay = pkl.load("pc4_datax.pkl", "rb")
            self.model_pred = model.predict(datax)
            output_data = pd.Series(self.model_pred)
            output_data.to_csv("output_PC4.csv", index=False)
    
def plotting(mode):

    if mode == "training":
        data = pd.read_csv("output_training_PC4.csv")
        plt.hist(data, color="blue", bins=30)
        plt.title("Distribution of training tweets")
    
    if mode == "testing":
        data = pd.read_csv("output_PC4.csv")
        plt.hist(data, color="blue", bins=30)
        plt.title("Distribution of testing tweets")

    plt.ylabel("# of Tweets")
    plt.xlabel("Log(Engagement) score")
    plt.show()
# plotting(mode="testing")        

def testing():
    data = pd.read_csv("ChatGPT_training_v1.csv")
    # datax = data.drop(["log_engagements"], axis=1)
    # datay = data["log_engagements"]

    xtrain, xtest = train_test_split(data, random_state=0, test_size= 0.3)
    _ = xtrain.to_csv("testing_mean_engage_train.csv", index=False)
    _ = xtest.to_csv("testing_mean_engage_testing.csv", index=False)

# testing()

def testing2():
    train = pd.read_csv("testing_mean_engage_train.csv")
    test = pd.read_csv("testing_mean_engage_testing.csv")

    xtrain = train.drop(["date", "tweet", "url", "user", "usercreated", "location", "userdescription", "log_engagements"], axis=1)
    ytrain = train["log_engagements"]

    xtest = test.drop(["date", "tweet", "url", "user", "usercreated", "location", "userdescription", "log_engagements"], axis=1)
    ytest = test["log_engagements"]

    rf = RandomForestRegressor()
    boost = GradientBoostingRegressor()
    # svm = SVR()

    rf.fit(xtrain, ytrain)
    rf_pred = rf.predict(xtest)
    rf_r2 = r2_score(ytest, rf_pred)
    rf_mse = mean_squared_error(ytest, rf_pred)

    # _ = pd.Series(rf_pred)
    # _ = _.to_csv("")

    boost.fit(xtrain, ytrain)
    boost_pred = boost.predict(xtest)
    boost_r2 = r2_score(ytest, boost_pred)
    boost_mse = mean_squared_error(ytest, boost_pred)


    # svm.fit(xtrain, ytrain)
    # svm_pred = svm.predict(xtest)
    # svm_r2 = r2_score(ytest, svm_pred)
    # svm_mse = mean_squared_error(ytest, svm_pred)

    print((f"Random forest:\nR2: {rf_r2}\nMSE: {rf_mse}\n\nBoosting:\nR2: {boost_r2}\n\
                    MSE: {boost_mse}\n\nSVM:\n"))#R2: {svm_r2}\nMSE: {svm_mse}""))
    # with open("training_results.txt", "w") as f:
    #     f.writelines(f"Random forest:\nR2: {rf_r2}\nMSE: {rf_mse}\n\nBoosting:\nR2: {boost_r2}\n\
    #                 MSE: {boost_mse}\n\nSVM:\nR2: {svm_r2}\nMSE: {svm_mse}")
    #     f.close()
    exit()
# testing2()
_ = input("Training or Testing?: ")
if _.lower() == "training":
# if 1 == 1:
    x = engage_pred("ChatGPT_training_v1.csv")
    x.feat_creation()
    # x.mean_engagement_add("ChatGPT_training_v1.csv","ChatGPT_test_v1.csv","")
    x.xy_split("training")
    x.feature_sel()
    x.drop_unimp_feat()
    x.training_testing()
    x.models("training", "boost")
    # x.plotting(_)

if _.lower() == "testing":
    y = engage_pred("ChatGPT_test_v1.csv")
    y.feat_creation()
    y.xy_split(_)
    # y.drop_unimp_feat()
    y.export_df()
    y.models("pred_output", "rf")
    # y.plotting(_)

if _.lower() == "export_train":
    x = engage_pred("testing_mean_engage_train.csv")
    x.feat_creation()
    x.mean_engagement_add("testing_mean_engage_train.csv","testing_mean_engage_testing.csv", "export")
    # x.xy_split()
    # x.feature_sel()
    # x.drop_unimp_feat()
    # x.training_testing()
    # x.models("cross_val", "boost")
    # x.plotting(_)

# Features to Create and test
# tweet length
# age of account vs the date of the post
# mean engagement fo rthe account
# lemngth of user description
# number of hastags in the weet
#ratio of friends to followers
# number of times the tweet mentioned someone else, number of @ signs 

# Find a way to include previous log engagments. maybe on the load of the testing data
# we check to see if user from testing data found in training data, if so we use that users mean enagge score
# if they are not foubd in the training dataset then we can take the mean score of all users and fill in


# getting the mean engagement scores for the test data is done
# need to validate by taking the mean for training split and filling in for test split
# then can do cross validation 

# the copy datasets have been made
# time to run cross validation