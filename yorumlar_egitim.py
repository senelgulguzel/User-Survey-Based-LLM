import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def main():
    #
    df = pd.read_csv("combine.csv")  

    
    X = df['Body']  
    y = df['Comments']  

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    
    print(classification_report(y_test, y_pred))

    
    new_comments = ["Bu yorum askerlik hakkında", "Bu yorum ev işleri hakkında", "Bu yorum öğrencilik hakkında"] #yorumların özetlenmesi ve kullanıcılara bir kimlik yaratılması amacıyla etiketleme yapmak
    predictions = model.predict(new_comments)
    for comment, prediction in zip(new_comments, predictions):
        print(f"Comment: {comment} -> Prediction: {prediction}")

if __name__ == "__main__":
    main()
