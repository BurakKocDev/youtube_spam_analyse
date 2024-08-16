import pandas as pd  
import numpy as np 


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier


from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('Youtube01-Psy.csv')
print(df.sample(5))

print(df.columns)

# Özellik Seçimi
data = df[['CONTENT', 'CLASS']].copy()
print(data.sample(5)) 

data['CLASS'] = data.CLASS.map({
    0 : 'Not Spam',
    1 : 'Spam Comment'
})

print(data.sample(5))

X = np.array(data['CONTENT']) # giriş
y = np.array(data['CLASS'])   # çıkış


cv = CountVectorizer()
X = cv.fit_transform(X).toarray()

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=0.2, random_state=42)


def TestModel(model, modelName):
    model.fit(X_Train, y_Train)
    modelScore = model.score(X_Test, y_Test)

    print(f"{modelName} Model Score: {modelScore : .3f}")




def TryExample(model = None, string : str = None, modelList : list = None):
    
    sample = (data['CONTENT'].sample(1)).iloc[0]
    print("=> Sample:", sample)
    rest = cv.transform([sample]).toarray()

    if (model != None):
        predict = model.predict(rest)
        print(f">> Prediction: {predict}")


    elif(modelList != None):
        for mdl in modelList:
            mdlName = str(mdl).split('()')[0]
            predict = mdl.predict(rest)
            print(f">> {mdlName} Prediction: {predict}")
            
            
            
            
modelBernoulli = BernoulliNB()
modelGaussian = GaussianNB()
modelMultinomial = MultinomialNB()

modelSVM = SVC()
modelRandomForest = RandomForestClassifier()
modelDecisionTree = DecisionTreeClassifier()
modelLogisticRegression = LogisticRegression()
modelKNN = KNeighborsClassifier()


TestModel(modelBernoulli, "BernoulliNB")
TestModel(modelGaussian, "GaussianNB")
TestModel(modelMultinomial, "MultinomialNB")



TestModel(modelSVM, "SVC")
TestModel(modelRandomForest, "RandomForest")
TestModel(modelDecisionTree, "DecisionTree")
TestModel(modelLogisticRegression, "LinearRegression")
TestModel(modelKNN, "KNN")

models = [modelBernoulli, modelGaussian, modelMultinomial, modelSVM, modelRandomForest, modelDecisionTree, modelLogisticRegression, modelKNN]     


TryExample(modelList=models)

# İllede Karar Tabanlı Sistem Kullanıyorsam, eğer bir özelliğim varsa LogisticRegression, çoklu özelliğim yani sütunum var ise SVM burada mantıklı olacaktır
# Basit bir veri seti, az bir veri seti, bu yüzden LR mantıklı olacaktır
       