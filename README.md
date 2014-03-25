# takahe

takahe is a multi-sentence compression module. Given a set of redundant sentences, a word-graph is constructed by iteratively adding sentences to it. The best compression is obtained by finding the shortest path in the word graph. The original algorithm was published and described in:

* Katja Filippova, Multi-Sentence Compression: Finding Shortest Paths in Word Graphs, *Proceedings of the 23rd International Conference on Computational Linguistics (Coling 2010)*, pages 322-330, 2010.

A keyphrase-based reranking method can be applied to generate more informative compressions. The reranking method is described in:

* Florian Boudin and Emmanuel Morin, Keyphrase Extraction for N-best Reranking in Multi-Sentence Compression, *Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2013)*, 2013.


## Dependancies

As of today, takahe is built for Python 2.

You may need to install the following libraries :

- [networkx](http://networkx.github.io/) (installation guide is available [here](http://networkx.github.io/documentation/latest/install.html))
- [graphviz](http://www.graphviz.org/) and graphviz-dev
- [pygraphviz](http://pygraphviz.github.io/documentation/latest/install.html)



## Example
A typical usage of this module is:
    
	import takahe
        
	# Create a word graph from the set of sentences with parameters :
	# - minimal number of words in the compression : 6
	# - language of the input sentences : en (english)
	# - POS tag for punctuation marks : PUNCT
	compresser = takahe.word_graph( sentences, 
								    nb_words = 6, 
		                            lang = 'en', 
		                            punct_tag = "PUNCT" )

	# Get the 50 best paths
	candidates = compresser.get_compression(50)

	# 1. Rerank compressions by path length (Filippova's method)
	for cummulative_score, path in candidates:

		# Normalize path score by path length
		normalized_score = cummulative_score / len(path)

		# Print normalized score and compression
		print round(normalized_score, 3), ' '.join([u[0] for u in path])

	# Write the word graph in the dot format
	compresser.write_dot('test.dot')

	# 2. Rerank compressions by keyphrases (Boudin and Morin's method)
	reranker = takahe.keyphrase_reranker( sentences,  
										  candidates, 
										  lang = 'en' )

	reranked_candidates = reranker.rerank_nbest_compressions()

	# Loop over the best reranked candidates
	for score, path in reranked_candidates:
		
		# Print the best reranked candidates
		print round(score, 3), ' '.join([u[0] for u in path])
