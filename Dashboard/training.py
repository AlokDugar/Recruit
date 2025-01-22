from nltk.corpus import stopwords
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
import PyPDF2
import spacy

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()  
    return text


def train(fnames):
    docs = []

    for i in range(len(fnames)):
        pdf_path = './' + fnames[i]
        print(f"Processing file: {pdf_path}")
        texts = extract_text_from_pdf(pdf_path)
        tx = " ".join(texts.split('\n'))
        docs.append(tx)

    print(docs[len(docs) - 1])

    nlp_model_annotation = spacy.load('./Dashboard/nlpdesc_model', exclude=['tokenizer'])
    nlp_CV_annotation = spacy.load('./Dashboard/nlpdesc_CV_model', exclude=['tokenizer'])

    documents = []

    for index, d in enumerate(docs):
        if index == len(docs) - 1:
            document = nlp_model_annotation(d)
        else:
            document = nlp_CV_annotation(d)

        text = "".join(ent.text for ent in document.ents)

        if len(text) < 100:
            documents.append(d)
        else:
            documents.append(text)

    wordnet = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)

    modified_arr = [[wordnet.lemmatize(i.lower()) for i in tokenize(d.translate(table)) if i.lower() not in stop_words] for d in documents]

    skip = []
    for d in modified_arr:
        for i in range(len(d)):
            synonyms = []
            for syn in wn.synsets(d[i]):
                for l in syn.lemmas():
                    synonyms.append(l.name())

            for doc in modified_arr:
                for j in range(len(doc)):
                    if doc[j] not in skip:
                        if doc[j] != d[i] and doc[j] in synonyms:
                            doc[j] = d[i]
                            if doc[j] not in skip:
                                skip.append(doc[j])

    modified_doc = [' '.join(i) for i in modified_arr]
    tf_idf = TfidfVectorizer().fit_transform(modified_doc)

    similarity = []
    length = len(documents) - 1
    for i in range(length + 1):
        cosine = cosine_similarity(tf_idf[length], tf_idf[i])
        similarity.append(cosine)
        print(cosine)

    return similarity


def train_desc(fnames):
    docs = []

    for i in range(len(fnames)):
        pdf_path = './' + fnames[i]
        print(f"Processing file: {pdf_path}")
        texts = extract_text_from_pdf(pdf_path)
        tx = " ".join(texts.split('\n'))
        docs.append(tx)

    nlp_model_annotation = spacy.load('./Dashboard/nlpdesc_model')
    nlp_CV_annotation = spacy.load('./Dashboard/nlpdesc_CV_model')

    documents = []

    for index, d in enumerate(docs):
        if index == len(docs) - 1:
            document = nlp_CV_annotation(d)
        else:
            document = nlp_model_annotation(d)

        text = "".join(ent.text for ent in document.ents)

        if len(text) < 100:
            documents.append(d)
        else:
            documents.append(text)

    wordnet = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)

    modified_arr = [[wordnet.lemmatize(i.lower()) for i in tokenize(d.translate(table)) if i.lower() not in stop_words] for d in documents]

    skip = []
    for d in modified_arr:
        for i in range(len(d)):
            synonyms = []
            for syn in wn.synsets(d[i]):
                for l in syn.lemmas():
                    synonyms.append(l.name())

            for doc in modified_arr:
                for j in range(len(doc)):
                    if doc[j] not in skip:
                        if doc[j] != d[i] and doc[j] in synonyms:
                            doc[j] = d[i]
                            if doc[j] not in skip:
                                skip.append(doc[j])

    modified_doc = [' '.join(i) for i in modified_arr]
    tf_idf = TfidfVectorizer().fit_transform(modified_doc)

    similarity = []
    length = len(documents) - 1
    for i in range(length + 1):
        cosine = cosine_similarity(tf_idf[length], tf_idf[i])
        similarity.append(cosine)
        print(cosine)

    return similarity
