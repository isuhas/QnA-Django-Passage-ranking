from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render_to_response

# Create your views here.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords


#-----------------Data loading--------------------------------------
df = pd.read_excel("C:/Users/Suhas/Desktop/QnA.xlsx")
#--------------------------------------------------------------------

passages = df["Passages"].tolist()
printable = set(string.printable)

#-----------------Data Cleaning--------------------------------------
def remove_non_english(x):
    return "".join(list(filter(lambda x: x in printable, x)))
	
#df["Passages"] = df["Passages"].apply(remove_non_english)
df["Passages"] = df["Passages"].apply(lambda x:"".join([i for i in x if i not in string.punctuation]))

def whitespaces(x):
    w = re.sub(' +', ' ',x)
    w = w.strip()
    return w

df["Passages"] = df["Passages"].apply(lambda x: whitespaces(x))

stop = stopwords.words('english')
df["Passages"] = [x.lower() for x in df["Passages"]]
df["Passages"] = df["Passages"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df["Passages"].head()

data = df["Passages"].tolist()

def home(request):
	return HttpResponse("Welcome!!!")

#-----------------Prediction--------------------------------------
def predict(request):
	# Get values from browser
	q = request.GET.get('q')
	try:
		tfidf_vectorizer = TfidfVectorizer()
		tfidf_matrix = tfidf_vectorizer.fit_transform(data)
		query = tfidf_vectorizer.transform([str(q)])
		print(cosine_similarity(query, tfidf_matrix))
		score = cosine_similarity(query, tfidf_matrix)
		n = 5 # top 5
		score_ind = np.argsort(score[0].tolist())[::-1][:n] # getting the index of passage match (Highest to lowest)
		l = []
		for i in score_ind:
			l.append(passages[i])
			#l.append("-"*100)
		li = [*range(5)]
		combo = zip(li,l)
		dd = dict(combo)
	except Exception as e:
		print(e)
	#return HttpResponse(l)
	return JsonResponse(dd[0], safe = False)
	#return JsonResponse(dd, safe = False)
