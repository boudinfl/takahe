# takahe

takahe is a multi-sentence compression module. Given a set of redundant sentences, a word-graph is constructed by iteratively adding sentences to it. The best compression is obtained by finding the shortest path in the word graph. The original algorithm was published and described in:

* Katja Filippova, Multi-Sentence Compression: Finding Shortest Paths in Word Graphs, *Proceedings of the 23rd International Conference on Computational Linguistics (Coling 2010)*, pages 322-330, 2010.
    
A typical usage of this module is:
    
	import takahe
        
	# A list of tokenized and POS-tagged sentences
	sentences = ['Hillary/NNP Clinton/NNP wanted/VBD to/stop visit/VB ...']
        
	# Create a word graph from the set of sentences
	msc = takahe.graph_fusion(sentences, 6, 'en', "PUNCT")

	# Get the 50 best paths
	candidates = msc.get_compression(50)

	# Rerank compressions by path length
	for cummulative_score, path in best_paths:

		# Compute the normalized score
		normalized_score = cummulative_score / len(path)

		# Print normalized score and compression
		print round(normalized_score, 3), ' '.join([u[0] for u in path])