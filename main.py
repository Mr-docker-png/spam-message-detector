import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


# datset
data = {
    'message': [
        'Win money now',
        'Claim your free prize',
        'Hello friend how are you',
        'Lets meet tomorrow',
        'Free lottery offer',
        'Are you coming today',
        'Win cash prize now',
        'Good morning',
        'Limited time free offer',
        'Call me tonight',
        'Earn money quickly',
        'See you tomorrow'
    ],

    'label': [
        'spam',
        'spam',
        'ham',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
}

df=pd.DataFrame(data)

X=df['message']
y=df['label']

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model=MultinomialNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print("Model Accuracy: ",accuracy)

# from user input
user_message=input("\nEnter a Message: ")
message_vector=vectorizer.transform([user_message])

prediction=model.predict(message_vector)

if prediction[0] =='spam':
    print("\nThis message is SPAM")
else:
    print("\n This meesage is Not SPAM")