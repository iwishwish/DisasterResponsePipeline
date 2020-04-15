import sys
import re
import pickle
import pandas as pd
import numpy as np

import sqlalchemy
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    load data from database
    
    INPUT:
    database_filepath - (str)
        database file path 
    
    OUTPUT:
    X - (pandas series) 
        X for modeling, each element is original text of message
    Y - (pandas dataframe) 
        Y for modeling, each column represents result of one caegory.
    category_names - (list) 
        category names list 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessagesVsCategories', engine)
    X = df.message 
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names

    
def text_process(text):
    '''
    transform text by a series of steps, including case normalization,
    tokenize, removing stop words and lemmatizer.
    
    INPUT:
    text - (str)
        Input text
    
    OUTPUT:
    tokens - (list) 
        a list of tokens
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    define meachine learning pipeline, parameters and GridSearchCV object.
    
    INPUT:
    None
    
    OUTPUT:
    cv - (GridSearchCV Object)
         GridSearchCV Object with pipeline, parameters, scoring all defined
        
    '''
    parameter = {'nlp__ngram_range':[(1,1)],
             'nlp__max_features':(None, 400, 500, 600, 700, 800, 900, 1000, 2000, 4000 ),
             'clf__estimator__alpha':[0, 1],
            }

    pipeline_NB = Pipeline([
        ('nlp', TfidfVectorizer(tokenizer=text_process)),
        ('clf', MultiOutputClassifier(BernoulliNB()))
    ])

    cv = GridSearchCV(pipeline_NB, param_grid=parameter, scoring='f1_macro')
    
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    output model's accuracy, f1, precision, recall for all categories
    
    INPUT:
    model - (pipeline)
        model obj which needs to be evaluate
    X_test - ()
        X of test set
    Y_test - ()
        Y of test set
    category_names - (list)
        category names list 
    
    OUTPUT:
    None
    '''
    pred = model.best_estimator_.predict(X_test)
    
    res = {
        'accuracy':[],
        'f1':[],
        'precision':[], 
        'recall':[]
    }
    
    for i in range(36):
        report = classification_report(Y_test.iloc[:,i], pred[:,i], output_dict=True)
        try:
            precision = report['1']['precision']
        except:
            precision = np.nan
        try:
            recall = report['1']['recall']
        except:
            recall = np.nan
        try:
            f1 = report['1']['f1-score']
        except:
            f1 = np.nan
        accuracy = report['accuracy']

        print(f'{Y_test.columns[i]:25}\
        accuracy:{accuracy:7.4f} f1:{f1:7.4f} precision:{precision:7.4f} recall:{recall:7.4f}')
        
        res['precision'].append(precision)
        res['recall'].append(recall)
        res['f1'].append(f1)
        res['accuracy'].append(accuracy)
        
    mean_acc = np.array(res['accuracy']).mean()
    mean_f1 = np.array(res['f1'])[~np.isnan(res['f1'])].mean()
    mean_precision = np.array(res['precision'])[~np.isnan(res['precision'])].mean()
    mean_recall = np.array(res['recall'])[~np.isnan(res['recall'])].mean()
    
    print(f"{mean_acc:49.4f} {mean_f1:10.4f} {mean_precision:17.4f} {mean_recall:14.4f}")
        
def save_model(model, save_path):
    '''
    save model in a pickle file.
    
    INPUT:
    model - (pipeline object)
        model object
    save_path - (str)
        model saved file path 
    
    OUPUT:
    None
    '''
    with open(save_path, 'wb') as outfile:
        pickle.dump(model.best_estimator_, outfile)


def main():
    '''
    Train classifier for disasters response.
    
    INPUT:
    argv[1] - (str)
        file path for disasters response database
    argv[2] - (str)
        file path for disasters response classifier
    
    OUTPUT:
    None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()