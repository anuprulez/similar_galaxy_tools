"""
Predict similarity among tools using BM25 score for each token
"""

import os
import numpy as np
import pandas as pd
import operator
import json
from math import *

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

import utils


class PredictToolSimilarity:

    def __init__( self ):
        self.file_path = '/data/all_tools.csv'

    @classmethod
    def read_file( self ):
        """
        Read the description of all tools
        """
        os.chdir( os.path.dirname( os.path.abspath( __file__ ) ) + '/data' )
        return pd.read_csv( 'all_tools.csv' )

    @classmethod
    def extract_tokens( self, file, tokens_source ):
        """
        Extract tokens from the description of all tools
        """
        tools_tokens_source = dict()
        for source in tokens_source:
            tools_tokens = list()
            for item in file.iterrows():
                row = item[ 1 ]
                tokens = self.get_tokens_from_source( row, source )
                tools_tokens.append( tokens )
            tools_tokens_source[ source ] = tools_tokens
        return tools_tokens_source

    @classmethod
    def get_tokens_from_source( self, row, source ):
        tokens = ''
        if source == 'input_output':
            tokens = utils._restore_space( utils._get_text( row, "inputs" ) ) + ' '
            tokens += utils._restore_space( utils._get_text( row, "outputs" ) )
        elif source == 'name_desc':
            tokens = utils._restore_space( utils._get_text( row, "name" ) ) + ' '
            tokens += utils._restore_space( utils._get_text( row, "description" ) )
        elif source == 'edam_help':
            tokens = utils._get_text( row, "help" ) + ' '
            tokens += utils._get_text( row, "edam_topics" )
        return utils._remove_special_chars( tokens.lower() )

    @classmethod
    def refine_tokens( self, tokens ):
        """
        Refine the set of tokens by removing words like 'to', 'with'
        """
        k = 2
        b = 0
        refined_tokens_sources = dict()
        for source in tokens:
            refined_tokens = list()
            files = list()
            inverted_frequency = dict()
            file_id = -1
            total_file_length = 0
            for item in tokens[ source ]:
                file_id += 1
                file_tokens = item.split(" ")
                total_file_length += len( file_tokens )
                term_frequency = dict()
                for token in file_tokens:
                    if token is not '':
                        file_ids = list()
                        if token not in inverted_frequency:
                            file_ids.append( file_id )
                        else:
                            file_ids = inverted_frequency[ token ]
                            if file_id not in file_ids:
                                file_ids.append( file_id )
                        inverted_frequency[ token ] = file_ids
                        # for term frequency
                        if token not in term_frequency:
                            term_frequency[ token ] = 1
                        else:
                            term_frequency[ token ] += 1
                files.append( term_frequency )
            # parameters
            N = len(files)
            average_file_length = float( total_file_length ) / N
        
            # find BM25 score for each token of each tool. It helps to determine
            # how important each word is with respect to the tool and other tools
            for item in files:
                for token in item:
                    file_length = len( item )
                    tf = item[ token ]
                    idf = np.log2( N / len( inverted_frequency[ token ] ) )
                    alpha = ( 1 - b ) + ( float( b * file_length ) / average_file_length )
                    tf_star = tf * float( ( k + 1 ) ) / ( k * alpha + tf )
                    tf_idf = tf_star * idf
                    item[ token ] = tf_idf

            # filter tokens based on the BM25 scores. Not all tokens are important
            for item in files:
                sorted_x = sorted( item.items(), key=operator.itemgetter( 1 ), reverse=True )
                scores = [ score for (token, score) in sorted_x ]
                mean_score = np.mean( scores )
                sigma = np.sqrt( np.var( scores ) )
                selected_tokens = [ ( token, score ) for ( token, score ) in sorted_x if not utils._check_number( token ) ]
                selected_tokens_sorted = sorted( selected_tokens, key=operator.itemgetter( 1 ), reverse=True )
                refined_tokens.append( selected_tokens_sorted )
            tokens_file_name = 'tokens_' + source + '.txt'
            with open( tokens_file_name, 'w' ) as file:
                file.write( json.dumps( refined_tokens ) )
                file.close()
            refined_tokens_sources[ source ] = refined_tokens
        return refined_tokens_sources

    def create_document_tokens_matrix( self, documents_tokens ):
        """
        Create document tokens matrix
        """
        document_tokens_matrix_sources = dict()
        for source in documents_tokens:
            # create a unique list of all words
            all_tokens = list()
            doc_tokens = documents_tokens[ source ]
            for file_item in range( 0, len( doc_tokens ) ):
                for word_score in doc_tokens[ file_item ]:
                    word = word_score[ 0 ]
                    if word not in all_tokens:
                        all_tokens.append( word )

            document_tokens_matrix = np.zeros( ( len( doc_tokens ), len( all_tokens ) ) )
            for file_index, file_item in enumerate( doc_tokens ):
                for word_score in file_item:
                    word_index = [ token_index for token_index, token in enumerate( all_tokens ) if token == word_score[ 0 ] ][ 0 ]
                    document_tokens_matrix[ file_index ][ word_index ] = word_score[ 1 ]
            document_tokens_matrix_sources[ source ] = document_tokens_matrix
        return document_tokens_matrix_sources

    @classmethod
    def find_tools_cos_distance_matrix( self, document_token_matrix_sources ):
        """
        Find similarity distance using Cosine distance among tools
        """
        similarity_matrix_sources = dict()
        mat_size = 0
        for source in document_token_matrix_sources:
            similarity_matrix_sources[ source ] = cosine_similarity( document_token_matrix_sources[ source ] )
            mat_size = len( similarity_matrix_sources[ source ] )
        return similarity_matrix_sources, mat_size

    @classmethod
    def assign_similarity_importance( self, similarity_matrix_sources, mat_size ):
        #importance_factor = [ input_output_factor, name_desc_factor, edam_help_factor ]
        importance_factor = [ 0.6, 0.3, 0.1 ]
        counter = 0
        similarity_matrix = np.zeros( ( mat_size, mat_size ) )
        for scores_source in similarity_matrix_sources:
            similarity_matrix += importance_factor[ counter ] * similarity_matrix_sources[ scores_source ]
            counter += 1
        return similarity_matrix

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe ):
        """
        Get similar tools for each tool
        """
        count_items = dataframe.count()[ 0 ]
        similarity_threshold = 0
        similarity = list()
        for i, rowi in dataframe.iterrows():
            file_similarity = dict()
            scores = list()
            for j, rowj in dataframe.iterrows():
                score = round( similarity_matrix[ i ][ j ], 2 )
                if score > similarity_threshold:
                    record = {
                        "name_description": rowj[ "name" ] + " " + ( utils._get_text( rowj, "description" ) ),
                        "id": rowj[ "id" ],
                        "input_types": utils._get_text( rowj, "inputs" ),
                        "output_types": utils._get_text( rowj, "outputs" ),
                        "what_it_does": utils._get_text( rowj, "help" ),
                        "edam_text": utils._get_text( rowj, "edam_topics" ),
                        "score": score
                    }
                    scores.append( record )
            file_similarity[ "scores" ] = scores
            file_similarity[ "id" ] = rowi[ "id" ]
            similarity.append( file_similarity )

        with open( 'similarity_matrix.json','w' ) as file:
            file.write( json.dumps( similarity ) )
            file.close()


if __name__ == "__main__":
    tool_similarity = PredictToolSimilarity()
    dataframe = tool_similarity.read_file()
    print "Read tool files"

    tokens = tool_similarity.extract_tokens( dataframe, [ 'input_output', 'name_desc', 'edam_help' ] )
    print "Extracted tokens"

    refined_tokens = tool_similarity.refine_tokens( tokens )
    print "Refined tokens"

    document_tokens_matrix = tool_similarity.create_document_tokens_matrix( refined_tokens )
    print "Created document term matrix"

    print "Computing distance..."
    #tools_distance_matrix = tool_similarity.find_tools_eu_distance_matrix( document_tokens_matrix )
    
    tools_distance_matrix, mat_size = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix )
    print "Computed distance"

    print "Assign importance to similarity matrix..."
    similarity_matrix = tool_similarity.assign_similarity_importance( tools_distance_matrix, mat_size )

    print "Writing results to a JSON file..."
    tool_similarity.associate_similarity( similarity_matrix, dataframe )
    print "Listed the similar tools in a JSON file"

    print "Program finished"
