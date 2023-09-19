#10-07-2023
import re
import string

import nltk
import numpy as np
import requests
import spacy
from apiclient.discovery import build
# from pyngrok import ngrok
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from gensim.models import KeyedVectors
from keras.models import load_model
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#load and download the model,nlp and stopwords list
model = load_model("p.h5")
nltk.download("punkt")
wv = KeyedVectors.load('place.kv')
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

#initialize api key for search engine,punctuations for preprocessing and a case to differentiate url from text
punctuations = string.punctuation
api_key = "AIzaSyD9d_pO0NacO6mx5Cx66DIZoWkhmuXhJpo"
resource = build("customsearch", "v1", developerKey=api_key).cse()
HTML_PARSE_STRING = "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"

def spacy_tokenizer(sentence):
  doc = nlp(sentence)
  mytokens = [word.lemma_.lower().strip() for word in doc]
  mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
  return mytokens

#this function takes the input
def sent_vec(sent):
  vector_size = wv.vector_size
  wv_res = np.zeros(vector_size)
  # print(wv_res)
  ctr = 1
  for w in sent:
    if w in wv:
      ctr += 1
      wv_res += wv[w]
  wv_res = wv_res / ctr
  return wv_res

#for input 1(headline or text) use search engine to get similar articles
def getSimilarArticles(input_text):
  #1st change
  input_text=spacy_tokenizer(input_text)
  #tokenize the input text and then place them as words in the input
  print(input_text)
  result = resource.list(q=input_text, cx='91e1f252db90748a5').execute()  # q is the query
  links = []
  headline = []
  for item in result['items']:
    print(item['title'], item['link'])
    links.append(item['link'])
  for i in range(len(links)):
    result = requests.get(links[i])
    doc = BeautifulSoup(result.text, "html.parser")
    i = doc.find("h1")
    if (i != None):
      headline.append(i.text)

    title = []
    for i in range(len(headline)):
        headline[i] = ' '.join(headline[i].split())
        # print(headline[i])
        title.append(headline[i])

    parts = []
    for i in range(len(title)):
        r = word_tokenize(title[i])
        # tokenizer = RegexpTokenizer(r'\w+')
        r = (" ").join(r)
        parts.append(r)

    vectorizer = TfidfVectorizer()
    l = vectorizer.fit_transform(parts)

    cs = linear_kernel(l[0:1], l).flatten()
    # cs = np.sort(cs)[::-1]
    x = max(cs)
    r = (cs[0])
    print("r:", r)
    result = requests.get(links[np.where(cs == r)[0][0]])
    print("Result inside", result)
    return result.text, links[np.where(cs == r)[0][0]]

#return urls
def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+')

    # Find all matches of URLs in the text
    urls = re.findall(url_pattern, text)
    print("Printing URLs: ", urls)
    return urls




app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

regex = "(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
regex1="(https?://\S+)"

@app.route("/", methods=["POST"])
def output():
    if 'input1' in request.form:
        input1 = request.form['input1']
        url = re.search(r'(https?://\S+)', input1)
        print(url)
        if  re.search(r'(https?://\S+)', input1):
            # Only URL is present
            #2 cases
            url = re.search(r'(https?://\S+)', input1)
            print(url)
            if(len(url[0])==input1):
                is_similar_search = False
                similar_article_url = "NA"
                return render_template("index.html",
                                       content="Not Possible",
                                       ip_text=input1,
                                       is_similar_search=is_similar_search,
                                       similar_search_link=similar_article_url)
            else:
                print(url[0])
                result = url[0]
                is_similar_search = False
                similar_article_url = "NA"
                inp_text=input1

        #headline
        else:
            result, similar_article_url = getSimilarArticles(input1)
            print("Similar article url" + similar_article_url)
            is_similar_search = True
            inp_text = input1

    if 'input2' in request.form:
        input2 = request.form['input2']
        if(re.search(HTML_PARSE_STRING,input2)==input2):
            result=input2
            is_similar_search=False
            similar_article_url="NA"
            inp_text = input2
        else:
            #not possible,not a text section
            return render_template("index.html",
                                   content="Not Possible",
                                   ip_text=input2,
                                   is_similar_search="Not Possible",
                                   similar_search_link="None")


    # #if 'input3' in request.form:
    # else:
    #     return render_template("index.html",
    #                            content="Not Possible",
    #                            ip_text=input1,
    #                            is_similar_search="",
    #                            similar_search_link="None")



    print("Result", result)
    response = requests.get(result)
    html_content = response.content
    doc = BeautifulSoup(html_content, "html.parser")
    #doc = BeautifulSoup(result, "html.parser")
    div_contents = doc.find_all('div')
    text_full=""
    for div_content in div_contents:
        nested_tags = div_content.find_all("p", recursive=False)
        for tag in nested_tags:
            print(tag.text.strip())
            text_full=tag.text.strip()
    print(text_full)
    a = spacy_tokenizer(' '.join(text_full))
    a = sent_vec(a)
    a = a.reshape((1, 300))
    t = model.predict(a)
    if np.round(t) == 1:
        answer = "True News"
    else:
       answer = "False News"
    if similar_article_url == "":
        similar_article_url = "NAA"
    print("Similar article url" + similar_article_url)
    print("Result:", result)
    return render_template("index.html",
                                content=answer,
                               ip_text=inp_text,
                               is_similar_search=is_similar_search,
                               similar_search_link=similar_article_url)



if __name__ == '__main__':
    app.run(debug=True)

