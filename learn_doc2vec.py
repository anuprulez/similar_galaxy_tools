"""
Learn similarity among documents using neural networks (Doc2Vec)
"""

import numpy as np
import operator
import gensim
import json
from gensim.models.doc2vec import TaggedDocument
from random import shuffle


class Learn_Doc2Vec_Similarity:

    @classmethod
    def __init__( self, doc_tokens ):
        self.doc_tokens = doc_tokens

    @classmethod
    def _tag_document( self ):
        """
        Get tagged documents
        """
        tagged_documents = []
        tools_list = list()
        tool_counter = 0
        for tool in self.doc_tokens:
            if tool not in tools_list:
                tools_list.append( tool )
            tokens = self.doc_tokens[ tool ]
            tokens = [ token for ( token, score ) in tokens ]
            td = TaggedDocument( gensim.utils.to_unicode(' '.join( tokens ) ).split(), [ tool_counter ] )
            tagged_documents.append( td )
            tool_counter += 1
        return tagged_documents, tools_list

    @classmethod
    def _find_document_similarity( self, tagged_documents, tools_list, src, window=15, n_dim=200, training_epochs=10, iterations=800 ):
        """
        Find the similarity among documents by training a neural network (Doc2Vec)
        """
        len_tools = len( tools_list )
        model = gensim.models.Doc2Vec( tagged_documents, dm=0, size=n_dim, negative=2, min_count=1, iter=iterations, window=window, alpha=1e-2, min_alpha=1e-4, dbow_words=0, sample=1e-5, seed=2 )
        for epoch in range( training_epochs ):
            print ( 'Training epoch %d of %d' % ( epoch + 1, training_epochs ) )
            shuffle( tagged_documents )
            model.train( tagged_documents, total_examples=model.corpus_count, epochs=model.iter )
        tools_similarity = list()
        doc_vectors = list()
        for index in range( len_tools ):
            similarity = model.docvecs.most_similar( index, topn=len_tools )
            sim_scores = [ ( int( item_id ), score ) for ( item_id, score ) in similarity ]
            sim_scores = sorted( sim_scores, key=operator.itemgetter( ( 0 ) ), reverse=False )
            # set 1.0 as similarity score with itself
            sim_scores.insert( index, ( index, 1.0 ) )
            # set the similarity values less than the median to 0 to
            # match up with the sudden drop of values to 0 in input/output similarity values
            sim_scores = [ score if score >= 0.0 else 0.0 for ( item_id, score ) in sim_scores ]
            tools_similarity.append( sim_scores )
            doc_vectors.append( [ str( x ) for x in model.docvecs[ index ] ] )
        with open( "data/doc_vecs_" + src + ".json", "w" ) as doc_vecs:
            doc_vecs.write( json.dumps( doc_vectors ) )
        return tools_similarity

    @classmethod
    def learn_doc_similarity( self, src, window, n_dim ):
        """
        Learn similarity among documents using neural network (doc2vec)
        """
        print( "Computing similarity..." )
        tagged_doc, tools_list = self._tag_document()
        tools_similarity = self._find_document_similarity( tagged_doc, tools_list, src, window, n_dim )
        return tools_similarity, tools_list
