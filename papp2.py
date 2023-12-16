import spacy
import pytextrank
import streamlit as st
from PIL import Image

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

image = Image.open('mitaoe-logo.jpg')
st.image(image)
st.header('Hello!! from - Text Summarizers - MITAOE')
inp = st.text_input('Paste text')
example_text = str(inp)

doc=nlp(example_text)

for sent in doc._.textrank.summary(limit_sentences=1):
  print(sent)

phrases_and_ranks=[(phrase.chunks[0],phrase.rank) for phrase in doc._.phrases]
phrases_and_ranks[:10]
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name="google/pegasus-xsum"

from transformers import PegasusTokenizer


pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model=PegasusForConditionalGeneration.from_pretrained(model_name)
tokens = pegasus_tokenizer(example_text,truncation=True,padding="longest",return_tensors="pt")
st.text(tokens)