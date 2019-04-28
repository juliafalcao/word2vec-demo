import pandas as pd
from nltk.tokenize import word_tokenize
import gensim
import sys


def main():
	# Verificando se alguma palavra foi informada para verificacao de similaridade
	if len(sys.argv) < 2:
		print("Informe uma palavra como argumento para verificar as palavras mais similares")
		sys.exit(0)

	print('#Aplicando o word2vec no dataset, por favor aguarde')

	# Carregando dataset de treinamento
	data = pd.read_json("sarcasm_headlines_dataset.json", lines=True)

	# Entrada = titulo artigo | Saida: o artigo é sarcastico ?
	data = data[["headline", "is_sarcastic"]]

	# Lendo todas os titulos de artigos
	data["headline"] = data["headline"].apply(lambda text: gensim.utils.simple_preprocess(text))

	# Aplicando o word2vec
	model = gensim.models.Word2Vec(
		data["headline"],
		size=700,
		window=4,
		min_count=2,
		workers=6
	)

	model.train(data["headline"], total_examples=len(data["headline"]), epochs=30)
	palavra = sys.argv[1]

	print('#Palavras mais parecidas com ' + palavra)
	print(model.wv.most_similar(positive=palavra,topn=8))


if __name__ == "__main__":
    main()

