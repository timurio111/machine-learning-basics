import pickle
import re
import click
import spacy
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def process(text):
    global nlp
    text = text = re.sub(r'[^\w\s]', '', text.lower())
    text = text = re.sub(r'\n', ' ', text.lower())
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop:
            tokens.append(token.lemma_)
    return ' '.join(tokens)

def preprocess_df(df):
    df = df[['rating', 'text', 'title']]
    df = df.dropna()
    df['text'] = df['text'].apply(process)
    df['title'] = df['title'].apply(process)
    df['text'] = df['title'] + '. ' + df['text']
    df = df[['rating', 'text']] 
    return df

@click.group()
def cli():
    pass

@cli.command()
@click.option('--data', type=click.Path(exists=True))
@click.option('--test', type=click.Path())
@click.option('--split', type=float)
@click.option('--model', type=click.Path())
def train(data, test, split, model):
    df = pd.read_csv(data)
    df = preprocess_df(df)

    if test:
        try:
            test_df = pd.read_csv()
        except:
            click.echo("Invalid test path")
            sys.exit(1)

        test_df = preprocess_df(test_df)

        X_train = df['text']
        y_train = df['rating']

        X_test = test_df['text']
        y_test = test_df['rating']
    
    elif split:
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['rating'], test_size=split)

    else:
        click.echo("Invalid test path and split size")
        sys.exit(1)
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    pickle.dump((vectorizer, clf), open('model.pkl', 'wb'))


@click.command()
@click.option('--model', required=True, type=str)
@click.option('--data', required=True, type=str)
def predict(model, data):
    vectorizer, _model = pickle.load(open(model, 'rb'))
    try:
        _data = pd.read_csv(data)
    except:
        click.echo("Invalid data path")
        sys.exit(1)
    _data = preprocess_df(_data)
    X_pred = vectorizer.transform(_data['text'])
    preds = _model.predict(X_pred)
    print(*preds)

cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()