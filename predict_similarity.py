"""
Predict similarity among tools using BM25 score for each token
"""

import os
import re
import numpy as np
import pandas as pd
import operator
import json
import sys

from sklearn.feature_extraction.text import CountVectorizer
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
    def extract_tokens( self, file ):
        """
        Extract tokens from the description of all tools
        """
        tools_tokens = list()
        for item in file.iterrows():
            row = item[ 1 ]
            description = utils._get_text( row, "description" )
            name = utils._get_text( row, "name" )
            help = utils._get_text( row, "help" )

            inputs = utils._restore_space( utils._get_text( row, "inputs" ) )
            outputs = utils._restore_space( utils._get_text( row, "outputs" ) )
            tokens = ''
            # append all tokens
            tokens += inputs + " "
            tokens += outputs + " "
            tokens += name + " "
            tokens += description + " "
            #tokens +=help
            tools_tokens.append( utils._remove_special_chars( tokens ) )
        return tools_tokens

    @classmethod
    def refine_tokens( self, tokens ):
        """
        Refine the set of tokens by removing words like 'to', 'with'
        """
        refined_tokens = list()
        files = list()
        inverted_frequency = dict()
        file_id = -1
        total_file_length = 0
        for item in tokens:
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
        k = 2
        b = 0.5

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
            sorted_x = sorted( item.items(), key=operator.itemgetter(1), reverse=True )
            scores = [ score for (token, score) in sorted_x ]
            mean_score = np.mean( scores )
            sigma = np.sqrt( np.var( scores ) )
            selected_tokens = [ ( token, score ) for ( token, score ) in sorted_x if not utils._check_number( token ) and score >= utils._check_quality_scores( mean_score, sigma, score ) ]
            refined_tokens.append( selected_tokens )
        return refined_tokens

    def create_document_tokens_matrix( self, documents_tokens ):
        """
        Create document tokens matrix
        """
        # create a unique list of all words
        all_tokens = list()
        for file_item in range( 0, len( documents_tokens ) ):
            for word_score in documents_tokens[ file_item ]:
                word = word_score[ 0 ]
                if word not in all_tokens:
                    all_tokens.append( word )

        document_tokens_matrix = np.zeros( ( len( documents_tokens ), len( all_tokens ) ) )
        for file_index, file_item in enumerate( documents_tokens ):
            for word_score in file_item:
                word_index = [ token_index for token_index, token in enumerate( all_tokens ) if token == word_score[ 0 ] ][ 0 ]
                document_tokens_matrix[ file_index ][ word_index ] = word_score[ 1 ]
        document_token_matrix.to_csv( 'dtmatrix.csv' )
        return document_tokens_matrix

    @classmethod
    def find_tools_eu_distance_matrix( self, document_token_matrix ):
        """
        Find similarity distance using Euclidean distance among tools
        """
        similarity_matrix = euclidean_distances( document_token_matrix )
        return similarity_matrix

    @classmethod
    def find_tools_cos_distance_matrix( self, document_token_matrix ):
        """
        Find similarity distance using Cosine distance among tools
        """
        similarity_matrix = cosine_similarity( document_token_matrix )
        #utils._plot_heatmap( similarity_matrix )
        return similarity_matrix

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe ):
        """
        Get similar tools for each tool
        """
        count_items = dataframe.count()[ 0 ]
        similarity_threshold = 0.1
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
    tokens = tool_similarity.extract_tokens( dataframe )
    print "Extracted tokens"
    refined_tokens = tool_similarity.refine_tokens( tokens )
    print "Refined tokens"
    document_tokens_matrix = tool_similarity.create_document_tokens_matrix( refined_tokens )
    print "Created document term matrix"
    #tools_distance_matrix = tool_similarity.find_tools_eu_distance_matrix( document_tokens_matrix )
    print "Computing cosine distance..."
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix )
    print "Computed cosine distance"
    tool_similarity.associate_similarity( tools_distance_matrix, dataframe )
    print "Listed the similar tools in a JSON file"
    print "Program finished"
