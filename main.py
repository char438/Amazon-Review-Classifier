import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

def main():
    new_input = input("Give a review you'd like to be predicted. \n")

    df = pd.DataFrame({
        'Text': [new_input] 
    })


    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def polarity_scores_roberta(example):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg' : scores[0],
            'roberta_neu' : scores[1],
            'roberta_pos' : scores[2]
        }
        return scores_dict


    res = {}
    try:
        text = df['Text'][0]
        roberta_result = polarity_scores_roberta(text)
        res[0] = roberta_result
    except RuntimeError:
        print(f'Try input a shorter review!')


    results_df = pd.DataFrame(res).T

    #Trained Model
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv('input/Results.csv')

    X = df.drop('Score',axis='columns')
    y = df.Score

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    model = RandomForestClassifier(n_estimators= 10)
    model.fit(X_train, y_train)

    new_prediction = model.predict(results_df)
    print("The predicted score is:", new_prediction[0])

    loop = input("Would you like another prediction? Y/N \n")
    if loop == 'Y':
        main()

if __name__ == '__main__':
    main()
    