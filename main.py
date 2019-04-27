import pandas as pd
from nltk.tokenize import word_tokenize
import gensim

data = pd.read_json("sarcasm_headlines_dataset.json", lines=True)

data = data[["headline", "is_sarcastic"]]
data["headline"] = data["headline"].apply(lambda text: gensim.utils.simple_preprocess(text))

print(data.head())
print(data.info())

model = gensim.models.Word2Vec(
	data["headline"],
	size=150,
	window=10,
	min_count=2,
	workers=6
)
model.train(data["headline"], total_examples=len(data["headline"], epochs=10))
w1 = "praise"
model.wv.most_similar(positive=w1)
