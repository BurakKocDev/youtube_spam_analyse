English Summary:
This Python script performs text classification on YouTube comments to detect spam using various machine learning models. The dataset is preprocessed by selecting relevant features and converting text data into numerical form using CountVectorizer. It maps the class labels "0" and "1" to "Not Spam" and "Spam Comment", respectively.

Several models are trained, including Naive Bayes classifiers (BernoulliNB, GaussianNB, MultinomialNB), Support Vector Classifier (SVC), RandomForestClassifier, DecisionTreeClassifier, Logistic Regression, and K-Nearest Neighbors (KNN). The accuracy score for each model is computed and printed. Additionally, the script provides a function to test each model with a sample from the dataset and prints the prediction for spam or not spam.

Türkçe Özet:
Bu Python betiği, YouTube yorumları üzerinde spam tespiti yapmak için metin sınıflandırması gerçekleştirir. Veri kümesi, ilgili özellikler seçilerek ve metin verileri CountVectorizer kullanılarak sayısal forma dönüştürülür. Sınıf etiketleri "0" ve "1" sırasıyla "Not Spam" ve "Spam Comment" olarak yeniden adlandırılır.

Birden fazla model eğitilir: Naive Bayes sınıflandırıcılar (BernoulliNB, GaussianNB, MultinomialNB), Destek Vektör Sınıflandırıcı (SVC), RandomForestClassifier, DecisionTreeClassifier, Lojistik Regresyon ve K-En Yakın Komşu (KNN). Her modelin doğruluk skoru hesaplanır ve yazdırılır. Ayrıca, her modelin bir veri örneği ile test edilmesini sağlayan ve spam olup olmadığını tahmin eden bir fonksiyon da içerir.






