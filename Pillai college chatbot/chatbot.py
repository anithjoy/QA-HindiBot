
#Meet Robo: your friend

#import necessary libraries
import io
import random
from re import I
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
from hindi_stemmer import hi_stem
# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

stop_words = set(line.strip() for line in open('final_stopwords.txt', encoding='utf8', errors ='ignore'))
#Reading in the corpus
with open('pillaicollegehindi.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()
    
#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
def hi_stem1(tokens):
	    if tokens not in stop_words:
	        token = ''
	        a = []
	        for token in tokens.split(" "):
	            a.append(hi_stem(token))
	        return(a)
	    else :
	        a = []
	        a.append(hi_stem(tokens))
	        return(a)

def get_processed_text(document):
	return hi_stem1(document)
#lemmer = WordNetLemmatizer()
#def LemTokens(tokens):
#    return [lemmer.lemmatize(token) for token in tokens]
#remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
#def LemNormalize(text):
#    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("नमस्कार")
GREETING_RESPONSES = ["""नमस्कार |
	आप कैसे है ?
	में कैसे आपकी सहायता कर सकता हूँ |
	"""]
QUESTION1 = ["आप कैसे है ?"]
ANSWERS = ["""नमस्कार |
	आप कैसे है ?
	में कैसे आपकी सहायता कर सकता हूँ |
	"""]
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    if sentence == "किन विषयों को पढ़ाया जाता है":
        return("कंप्यूटर इंजीनियरिंग, इलेक्ट्रॉनिक्स इंजीनियरिंग, मैकेनिकल इंजीनियरिंग सिखाया जाता है")
    if sentence == "कॉलेज में उपलब्ध सुविधाएं":
        return("जिम, खेल, कंप्यूटर लैब, इलेक्ट्रॉनिक लैब, लाइब्रेरी जैसी सुविधाएं उपलब्ध हैं")
    if sentence == "कॉलेज कहाँ है":
        return("कॉलेज नए पनवेल में स्थित है")


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=get_processed_text, stop_words=stop_words)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"मैं क्षमाप्रार्थी हूँ पर मैं आपके इस सवाल का उत्तर देने में असमर्थ हूँ ।"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("ROBO: नमस्कार आप कैसे है ? में कैसे आपकी सहायता कर सकता हूँ")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))
            else:
                print("ROBO: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: अलविदा")    
        
        

