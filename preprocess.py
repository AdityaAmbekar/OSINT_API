# import spacy
import re
# import en_core_web_sm
# from spacy.lang.en.stop_words import STOP_WORDS
# import sys
# import warnings
from ftfy import fix_text

# Function to clean the word of any punctuation or special characters
def cleanPunc(sentence): 
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/|-]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

# Function to produce pure alphabet based text
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def prepro(input):
    cleaned_input = cleanPunc(input)
    input_text = keepAlpha(cleaned_input)
    text = input_text.lower().split()
    input = [fix_text(str(i)) for i in text]
    return input