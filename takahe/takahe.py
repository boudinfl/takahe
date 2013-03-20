#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
:Name:
    takahe

:Authors:
    Florian Boudin (florian.boudin@univ-nantes.fr)

:Version:
    0.33

:Date:
    Feb. 2013

:Description:
    takahe is a multi-sentence compression module. Given a set of redundant 
    sentences, a word-graph is constructed by iteratively adding sentences to 
    it. The best compression is obtained by finding the shortest path in the
    word graph. The original algorithm was published and described in:

        Katja Filippova, Multi-Sentence Compression: Finding Shortest Paths
        in Word Graphs, *Proceedings of the 23rd International Conference on 
        Computational Linguistics (Coling 2010)*, pages 322-330, 2010.

:History:
    - 0.33 (Feb. 2013), bug fixes and better code documentation
    - 0.32 (Jun. 2012), Punctuation marks are now considered within the graph, 
      compressions are then punctuated
    - 0.31 (Nov. 2011), modified context function (uses the left and right 
      contexts), improved docstring documentation, bug fixes
    - 0.3 (Oct. 2011), improved K-shortest paths algorithm including verb/size 
      constraints and ordered lists for performance
    - 0.2 (Dec. 2010), removed dependencies from nltk (i.e. POS-tagging, 
      tokenization and stopwords removal)
    - 0.1 (Nov. 2010), first version

:Dependencies:
    - *networkx* for the graph construction (v1.2+)

:Usage:
    A typical usage of this module is::
    
        import takahe
        
        # A list of tokenized and POS-tagged sentences
        sentences = ['Hillary/NNP Clinton/NNP wanted/VBD to/stop visit/VB ...']
        
        # Create a word graph from the set of sentences with parameters :
        # - minimal number of words in the compression : 6
        # - language of the input sentences : en (english)
        # - POS tag for punctuation marks : PUNCT
        compresser = takahe.word_graph(sentences, 6, 'en', "PUNCT")

        # Get the 50 best paths
        candidates = compresser.get_compression(5)

        # Rerank compressions by path length
        for cummulative_score, path in candidates:

            # Normalize path score by path length
            normalized_score = cummulative_score / len(path)

            # Print normalized score and compression
            print round(normalized_score, 3), ' '.join([u[0] for u in path])

        # Write the word graph in the dot format
        compresser.write_dot('test.dot')

:Misc:
    The Takahe is a flightless bird indigenous to New Zealand. It was thought to
    be extinct after the last four known specimens were taken in 1898. However, 
    after a carefully planned search effort the bird was rediscovered by on 
    November 20, 1948. (Wikipedia, http://en.wikipedia.org/wiki/takahe)  
"""

import math
import codecs
import os
import re
import sys
import bisect
import networkx as nx
#import matplotlib.pyplot as plt

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# [ Class word_graph
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
class word_graph:
    """
    The word_graph class constructs a word graph from the set of sentences given
    as input. The set of sentences is a list of strings, sentences are tokenized
    and words are POS-tagged (e.g. ``"Saturn/NNP is/VBZ the/DT sixth/JJ 
    planet/NN from/IN the/DT Sun/NNP in/IN the/DT Solar/NNP System/NNP"``). 
    Three optional parameters can be specified. The first parameter is the 
    minimal number of words for the best compression (default value is 8). The 
    second parameter is the language and is used for selecting the correct 
    stopwords list (default is "en" for english, stopword lists are localized in 
    /resources/ directory). The third parameter is the punctuation mark tag 
    used during graph construction (default is PUNCT).
    """

    #-T-----------------------------------------------------------------------T-
    def __init__(self, sentence_list, nb_words=8, lang="en", punct_tag="PUNCT"):

        self.sentence = sentence_list
        """ A list of sentences provided by the user. """

        self.length = len(sentence_list)
        """ The number of sentences given for fusion. """
        
        self.nb_words = nb_words
        """ The minimal number of words in the compression. """

        self.resources = os.path.dirname(__file__) + '/resources/'
        """ The path of the resources folder. """

        self.stopword_path = self.resources+'stopwords.'+lang+'.dat'
        """ The path of the stopword list, e.g. stopwords.[lang].dat. """

        self.stopwords = self.load_stopwords(self.stopword_path)
        """ The set of stopwords loaded from stopwords.[lang].dat. """

        self.punct_tag = punct_tag
        """ The stopword tag used in the graph. """

        self.graph = nx.DiGraph()
        """ The directed graph used for fusion. """
    
        self.start = '-start-'
        """ The start token in the graph. """

        self.stop = '-end-'
        """ The end token in the graph. """

        self.sep = '/-/'
        """ The separator used between a word and its POS in the graph. """
        
        self.term_freq = {}
        """ The frequency of a given term. """
        
        self.verbs = set(['VB', 'VBD', 'VBP', 'VBZ', 'VH', 'VHD', 'VHP', 'VBZ', 
        'VV', 'VVD', 'VVP', 'VVZ', 'V'])
        """
        The list of verb POS tags required in the compression. At least *one* 
        verb must occur in the candidate compressions.
        """

        # 1. Pre-process the sentences
        self.pre_process_sentences()

        # 2. Compute term statistics
        self.compute_statistics()

        # 3. Build the word graph
        self.build_graph()
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def pre_process_sentences(self):
        """
        Pre-process the list of sentences given as input. Split sentences using 
        whitespaces and convert each sentence to a list of (word, POS) tuples.
        """

        for i in range(self.length):
        
            # Normalise extra white spaces
            self.sentence[i] = re.sub(' +', ' ', self.sentence[i])
            
            # Tokenize the current sentence in word/POS
            sentence = self.sentence[i].split(' ')

            # Creating an empty container for the cleaned up sentence
            container = [(self.start, self.start)]

            # Looping over the words
            for w in sentence:
                
                # Splitting word, POS
                m = re.match("^(.+)/(.+)$", w)
                
                # Extract the word information
                token, POS = m.group(1), m.group(2)

                # Add the token/POS to the sentence container
                container.append((token.lower(), POS))
                    
            # Add the stop token at the end of the container
            container.append((self.stop, self.stop))

            # Recopy the container into the current sentence
            self.sentence[i] = container
    #-B-----------------------------------------------------------------------B-
    
    
    #-T-----------------------------------------------------------------------T-
    def build_graph(self):
        """
        Constructs a directed word graph from the list of input sentences. Each
        sentence is iteratively added to the directed graph according to the 
        following algorithm:

        - Word mapping/creation is done in four steps:

            1. non-stopwords for which no candidate exists in the graph or for 
               which an unambiguous mapping is possible or which occur more than
               once in the sentence

            2. non-stopwords for which there are either several possible
               candidates in the graph

            3. stopwords

            4. punctuation marks

        For the last three groups of words where mapping is ambiguous we check 
        the immediate context (the preceding and following words in the sentence 
        and the neighboring nodes in the graph) and select the candidate which 
        has larger overlap in the context, or the one with a greater frequency 
        (i.e. the one which has more words mapped onto it). Stopwords are mapped 
        only if there is some overlap in non-stopwords neighbors, otherwise a 
        new node is created. Punctuation marks are mapped only if the preceding 
        and following words in the sentence and the neighboring nodes are the
        same.

        - Edges are then computed and added between mapped words.
        
        Each node in the graph is represented as a tuple ('word/POS', id) and 
        possesses an info list containing (sentence_id, position_in_sentence)
        tuples.
        """     

        # Iteratively add each sentence in the graph ---------------------------
        for i in range(self.length):

            # Compute the sentence length
            sentence_len = len(self.sentence[i])

            # Create the mapping container
            mapping = [0] * sentence_len

            #-------------------------------------------------------------------
            # 1. non-stopwords for which no candidate exists in the graph or for 
            #    which an unambiguous mapping is possible or which occur more 
            #    than once in the sentence.
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue
            
                # Create the node identifier
                node = token.lower() + self.sep + POS

                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)] )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)

                # If there is only one matching node in the graph (id is 0)
                elif k == 1:

                    # Get the sentences id of this node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, 0)]['info']:
                        ids.append(sid)
                    
                    # Update the node in the graph if not same sentence
                    if not i in ids:
                        self.graph.node[(node, 0)]['info'].append((i, j))
                        mapping[j] = (node, 0)

                    # Else Create new node for redundant word
                    else:
                        self.graph.add_node((node, 1), info=[(i, j)])
                        mapping[j] = (node, 1)

            #-------------------------------------------------------------------
            # 2. non-stopwords for which there are either several possible
            #    candidates in the graph.
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]
                
                # If stopword or punctuation mark, continues
                if token in self.stopwords or re.search('(?u)^\W$', token):
                    continue

                # If word is not already mapped to a node
                if mapping[j] == 0:

                    # Create the node identifier
                    node = token.lower() + self.sep + POS
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS
                    
                    # Find the number of ambiguous nodes in the graph
                    k = self.ambiguous_nodes(node)

                    # Search for the ambiguous node with the larger overlap in
                    # context or the greater frequency.
                    ambinode_overlap = []
                    ambinode_frequency = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context(node, l, 'left')
                        r_context = self.get_directed_context(node, l, 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)

                        # Add the frequency of the ambiguous node
                        ambinode_frequency.append(
                            len( self.graph.node[(node, l)]['info'] )
                        )
                
                    # Search for the best candidate while avoiding a loop
                    found = False
                    selected = 0
                    while not found:
                    
                        # Select the ambiguous node
                        selected = self.max_index(ambinode_overlap)
                        if ambinode_overlap[selected] == 0:
                            selected = self.max_index(ambinode_frequency)
                        
                        # Get the sentences id of this node
                        ids = []
                        for sid, p in self.graph.node[(node, selected)]['info']:
                            ids.append(sid)
                        
                        # Test if there is no loop
                        if i not in ids:
                            found = True
                            break
            
                        # Remove the candidate from the lists
                        else:
                            del ambinode_overlap[selected]
                            del ambinode_frequency[selected]
                            
                        # Avoid endless loops
                        if len(ambinode_overlap) == 0:
                            break
                    
                    # Update the node in the graph if not same sentence
                    if found:
                        self.graph.node[(node, selected)]['info'].append((i, j))
                        mapping[j] = (node, selected)

                    # Else create new node for redundant word
                    else:
                        self.graph.add_node((node, k), info=[(i, j)])
                        mapping[j] = (node, k)
            
            #-------------------------------------------------------------------
            # 3. map the stopwords to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* stopword, continues
                if not token in self.stopwords :
                    continue

                # Create the node identifier
                node = token.lower() + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)] )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes, the
                        # boolean indicates to consider only non stopwords
                        l_context = self.get_directed_context(node, l, 'left',\
                                    True)
                        r_context = self.get_directed_context(node, l, 'right',\
                                    True)
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 0:
                    # if i not in ids and \
                    # (ambinode_overlap[selected] > 1 and POS==self.punct_tag) or\
                    # (ambinode_overlap[selected] > 0 and POS!=self.punct_tag) :

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        mapping[j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k) , info=[(i, j)])

                        # Mark the word as mapped to k
                        mapping[j] = (node, k)

            #-------------------------------------------------------------------
            # 4. lasty map the punctuation marks to the nodes
            #-------------------------------------------------------------------
            for j in range(sentence_len):

                # Get the word and tag
                token, POS = self.sentence[i][j]

                # If *NOT* punctuation mark, continues
                if not re.search('(?u)^\W$', token):
                    continue

                # Create the node identifier
                node = token.lower() + self.sep + POS
                    
                # Find the number of ambiguous nodes in the graph
                k = self.ambiguous_nodes(node)

                # If there is no node in the graph, create one with id = 0
                if k == 0:

                    # Add the node in the graph
                    self.graph.add_node( (node, 0), info=[(i, j)] )

                    # Mark the word as mapped to k
                    mapping[j] = (node, 0)
   
                # Else find the node with overlap in context or create one
                else:
                    
                    # Create the neighboring nodes identifiers
                    prev_token, prev_POS = self.sentence[i][j-1]
                    next_token, next_POS = self.sentence[i][j+1]
                    prev_node = prev_token.lower() + self.sep + prev_POS
                    next_node = next_token.lower() + self.sep + next_POS

                    ambinode_overlap = []
            
                    # For each ambiguous node
                    for l in range(k):

                        # Get the immediate context words of the nodes
                        l_context = self.get_directed_context(node, l, 'left')
                        r_context = self.get_directed_context(node, l, 'right')
                        
                        # Compute the (directed) context sum
                        val = l_context.count(prev_node) 
                        val += r_context.count(next_node)

                        # Add the count of the overlapping words
                        ambinode_overlap.append(val)
                    
                    # Get best overlap candidate
                    selected = self.max_index(ambinode_overlap)
                
                    # Get the sentences id of the best candidate node
                    ids = []
                    for sid, pos_s in self.graph.node[(node, selected)]['info']:
                        ids.append(sid)

                    # Update the node in the graph if not same sentence and 
                    # there is at least one overlap in context
                    if i not in ids and ambinode_overlap[selected] > 1:

                        # Update the node in the graph
                        self.graph.node[(node, selected)]['info'].append((i, j))

                        # Mark the word as mapped to k
                        mapping[j] = (node, selected)

                    # Else create a new node
                    else:
                        # Add the node in the graph
                        self.graph.add_node( (node, k) , info=[(i, j)])

                        # Mark the word as mapped to k
                        mapping[j] = (node, k)

            #-------------------------------------------------------------------
            # 4. Connects the mapped words with directed edges
            #-------------------------------------------------------------------
            for j in range(1, len(mapping)):
                self.graph.add_edge(mapping[j-1], mapping[j])

        # Assigns a weight to each node in the graph ---------------------------
        for node1, node2 in self.graph.edges_iter():
            edge_weight = self.get_edge_weight(node1, node2)
            self.graph.add_edge(node1, node2, weight=edge_weight)
    #-B-----------------------------------------------------------------------B-

 
    #-T-----------------------------------------------------------------------T-
    def ambiguous_nodes(self, node):
        """
        Takes a node in parameter and returns the number of possible candidate 
        (ambiguous) nodes in the graph.
        """
        k = 0
        while(self.graph.has_node((node, k))):
            k += 1
        return k
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_directed_context(self, node, k, dir='all', non_pos=False):
        """
        Returns the directed context of a given node, i.e. a list of word/POS of
        the left or right neighboring nodes in the graph. The function takes 
        four parameters :

        - node is the word/POS tuple
        - k is the node identifier used when multiple nodes refer to the same 
          word/POS (e.g. k=0 for (the/DET, 0), k=1 for (the/DET, 1), etc.)
        - dir is the parameter that controls the directed context calculation, 
          it can be set to left, right or all (default)
        - non_pos is a boolean allowing to remove stopwords from the context 
          (default is false)
        """

        # Define the context containers
        l_context = []
        r_context = []

        # For all the sentence/position tuples
        for sid, off in self.graph.node[(node, k)]['info']:
            
            prev = self.sentence[sid][off-1][0].lower() + self.sep +\
                   self.sentence[sid][off-1][1]
                   
            next = self.sentence[sid][off+1][0].lower() + self.sep +\
                   self.sentence[sid][off+1][1]
                   
            if non_pos:
                if self.sentence[sid][off-1][0] not in self.stopwords:
                    l_context.append(prev)
                if self.sentence[sid][off+1][0] not in self.stopwords:
                    r_context.append(next)
            else:
                l_context.append(prev)
                r_context.append(next)

        # Returns the left (previous) context
        if dir == 'left':
            return l_context
        # Returns the right (next) context
        elif dir == 'right':
            return r_context
        # Returns the whole context
        else:
            l_context.extend(r_context)
            return l_context
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def get_edge_weight(self, node1, node2):
        """
        Compute the weight of an edge *e* between nodes *node1* and *node2*. It 
        is computed as e_ij = (A / B) / C with:
        
        - A = freq(i) + freq(j), 
        - B = Sum (s in S) 1 / diff(s, i, j)
        - C = freq(i) * freq(j)
        
        A node is a tuple of ('word/POS', unique_id).
        """

        # Get the list of (sentence_id, pos_in_sentence) for node1
        info1 = self.graph.node[node1]['info']
        
        # Get the list of (sentence_id, pos_in_sentence) for node2
        info2 = self.graph.node[node2]['info']
        
        # Get the frequency of node1 in the graph
        # freq1 = self.graph.degree(node1)
        freq1 = len(info1)
        
        # Get the frequency of node2 in cluster
        # freq2 = self.graph.degree(node2)
        freq2 = len(info2)

        # Initializing the diff function list container
        diff = []

        # For each sentence of the cluster (for s in S)
        for s in range(self.length):
        
            # Compute diff(s, i, j) which is calculated as
            # pos(s, i) - pos(s, j) if pos(s, i) < pos(s, j)
            # O otherwise
    
            # Get the positions of i and j in s, named pos(s, i) and pos(s, j)
            # As a word can appear at multiple positions in a sentence, a list
            # of positions is used
            pos_i_in_s = []
            pos_j_in_s = []
            
            # For each (sentence_id, pos_in_sentence) of node1
            for sentence_id, pos_in_sentence in info1:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_i_in_s.append(pos_in_sentence)
            
            # For each (sentence_id, pos_in_sentence) of node2
            for sentence_id, pos_in_sentence in info2:
                # If the sentence_id is s
                if sentence_id == s:
                    # Add the position in s
                    pos_j_in_s.append(pos_in_sentence)
                    
            # Container for all the diff(s, i, j) for i and j
            all_diff_pos_i_j = []
            
            # Loop over all the i, j couples
            for x in range(len(pos_i_in_s)):
                for y in range(len(pos_j_in_s)):
                    diff_i_j = pos_i_in_s[x] - pos_j_in_s[y]
                    # Test if word i appears *BEFORE* word j in s
                    if diff_i_j < 0:
                        all_diff_pos_i_j.append(-1.0*diff_i_j)
                        
            # Add the mininum distance to diff (i.e. in case of multiple 
            # occurrencies of i or/and j in sentence s), 0 otherwise.
            if len(all_diff_pos_i_j) > 0:
                diff.append(1.0/min(all_diff_pos_i_j))
            else:
                diff.append(0.0)
                
        weight1 = freq1
        weight2 = freq2

        return ( (freq1 + freq1) / sum(diff) ) / (weight1 * weight2)
    #-B-----------------------------------------------------------------------B-
   
   
    #-T-----------------------------------------------------------------------T-
    def k_shortest_paths(self, start, end, k=10):
        """
        Simple implementation of a k-shortest paths algorithms. Takes three
        parameters: the starting node, the ending node and the number of 
        shortest paths desired. Returns a list of k tuples (path, weight).
        """

        # Initialize the list of shortest paths
        kshortestpaths = []

        # Initializing the label container 
        orderedX = []
        orderedX.append((0, start, 0))
        
        # Initializing the path container
        paths = {}
        paths[(0, start, 0)] = [start]
        
        # Initialize the visited container
        visited = {}
        visited[start] = 0

        # Initialize the sentence container that will be used to remove 
        # duplicate sentences passing throught different nodes
        sentence_container = {}
    
        # While the number of shortest paths isn't reached or all paths explored
        while len(kshortestpaths) < k and len(orderedX) > 0:
        
            # Searching for the shortest distance in orderedX
            shortest = orderedX.pop(0)
            shortestpath = paths[shortest]
            
            # Removing the shortest node from X and paths
            del paths[shortest]
    
            # Iterating over the accessible nodes
            for node in self.graph.neighbors(shortest[1]):
            
                # To avoid loops
                if node in shortestpath:
                    continue
            
                # Compute the weight to node
                w = shortest[0] + self.graph[shortest[1]][node]['weight']
            
                # If found the end, adds to k-shortest paths 
                if node == end:

                    #-T-------------------------------------------------------T-
                    # --- Constraints on the shortest paths

                    # 1. Check if path contains at least one werb
                    # 2. Check the length of the shortest path, without 
                    #    considering punctuation marks and starting node (-1 in
                    #    the range loop, because nodes are reversed)
                    # 3. Check the paired parentheses and quotation marks
                    # 4. Check if sentence is not redundant

                    nb_verbs = 0
                    length = 0
                    paired_parentheses = 0
                    quotation_mark_number = 0
                    raw_sentence = ''

                    for i in range(len(shortestpath) - 1):
                        word, tag = shortestpath[i][0].split(self.sep)
                        # 1.
                        if tag in self.verbs:
                            nb_verbs += 1
                        # 2.
                        if not re.search('(?u)^\W$', word):
                            length += 1
                        # 3.
                        else:
                            if word == '(':
                                paired_parentheses -= 1
                            elif word == ')':
                                paired_parentheses += 1
                            elif word == '"':
                                quotation_mark_number += 1
                        # 4.
                        raw_sentence += word + ' '
                    
                    # Remove extra space from sentence
                    raw_sentence = raw_sentence.strip()

                    if nb_verbs > 1 and length >= self.nb_words and \
                        paired_parentheses == 0 and \
                        (quotation_mark_number%2) == 0 \
                        and not sentence_container.has_key(raw_sentence):
                        path = [node]
                        path.extend(shortestpath)
                        path.reverse()
                        weight = float(w) #/ float(length)
                        kshortestpaths.append((path, weight))
                        sentence_container[raw_sentence] = 1

                    #-B-------------------------------------------------------B-

                else:
            
                    # test if node has already been visited
                    if visited.has_key(node):
                        visited[node] += 1
                    else:
                        visited[node] = 0
                    id = visited[node]

                    # Add the node to orderedX
                    bisect.insort(orderedX, (w, node, id))
                    
                    # Add the node to paths
                    paths[(w, node, id)] = [node]
                    paths[(w, node, id)].extend(shortestpath)
    
        # Returns the list of shortest paths
        return kshortestpaths
    #-B-----------------------------------------------------------------------B-
    
    #-T-----------------------------------------------------------------------T-
    def get_compression(self, nb_candidates=50):
        """
        Searches all possible paths from **start** to **end** in the word graph,
        removes paths containing no verb or shorter than *n* words. Returns an
        ordered list (smaller first) of nb (default value is 50) (cummulative 
        score, path) tuples. The score is not normalized with the sentence 
        length.
        """

        # Search for the k-shortest paths in the graph
        self.paths = self.k_shortest_paths((self.start+self.sep+self.start, 0),
                                           (self.stop+self.sep+self.stop, 0),
                                            nb_candidates)

        # Initialize the fusion container
        fusions = []
        
        # Test if there are some paths
        if len(self.paths) > 0:
        
            # For nb candidates 
            for i in range(min(nb_candidates, len(self.paths))):
                nodes = self.paths[i][0]
                sentence = []
                
                for j in range(1, len(nodes)-1):
                    word, tag = nodes[j][0].split(self.sep)
                    sentence.append((word, tag))

                bisect.insort(fusions, (self.paths[i][1], sentence))

        return fusions
    #-B-----------------------------------------------------------------------B-

    #-T-----------------------------------------------------------------------T-
    def max_index(self, l):
        """ Returns the index of the maximum value of a given list. """

        ll = len(l)
        if ll < 0:
            return None
        elif ll == 1:
            return 0
        max_val = l[0]
        max_ind = 0
        for z in range(1, ll):
            if l[z] > max_val:
                max_val = l[z]
                max_ind = z
        return max_ind
    #-B-----------------------------------------------------------------------B-


    #-T-----------------------------------------------------------------------T-
    def compute_statistics(self):
        """
        This function iterates over the cluster's sentences and computes the
        following statistics about each word:
        
        - term frequency (self.term_freq)
        """

        # Structure for containing the list of sentences in which a term occurs
        terms = {}

        # Loop over the sentences
        for i in range(self.length):
        
            # For each tuple (token, POS) of sentence i
            for token, POS in self.sentence[i]:
            
                # generate the word/POS token
                node = token.lower() + self.sep + POS
                
                # Add the token to the terms list
                if not terms.has_key(node):
                    terms[node] = [i]
                else:
                    terms[node].append(i)

        # Loop over the terms
        for w in terms:

            # Compute the term frequency
            self.term_freq[w] = len(terms[w])
    #-B-----------------------------------------------------------------------B-

    #-T-----------------------------------------------------------------------T-
    def load_stopwords(self, path):
        """
        This function loads a stopword list from the *path* file and returns a 
        set of words. Lines begining by '#' are ignored.
        """

        # Set of stopwords
        stopwords = set([])

        # For each line in the file
        for line in codecs.open(path, 'r', 'utf-8'):
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip().lower())

        # Return the set of stopwords
        return stopwords
    #-B-----------------------------------------------------------------------B-
    
    #-T-----------------------------------------------------------------------T-
    def write_dot(self, dotfile):
        """ Outputs the word graph in dot format in the specified file. """
        nx.write_dot(self.graph, dotfile)
    #-B-----------------------------------------------------------------------B-

#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ] Ending word_graph class
#~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~