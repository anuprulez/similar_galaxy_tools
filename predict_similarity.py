"""
Predict similarity among tools by analyzing the attributes and 
finding proximity using Best Match (BM25) and Gradient Descent algorithms
"""
import sys
import os
import numpy as np
import pandas as pd
import operator
import json
import time
from math import *
from nltk.stem import *

import utils
import gradientdescent


class PredictToolSimilarity:

    @classmethod
    def __init__( self ):
        self.tools_show = 50
        self.uniform_prior = 1 / 3.

    @classmethod
    def read_file( self, file_path ):
        """
        Read the description of all tools
        """
        return pd.read_csv( file_path )

    @classmethod
    def extract_tokens( self, file, tokens_source ):
        """
        Extract tokens from the description of all tools
        """
        tools_tokens_source = dict()
        for source in tokens_source:
            tools_tokens = dict()
            for row in file.iterrows():
                tokens = self.get_tokens_from_source( row[ 1 ], source )
                tools_tokens[ row[ 1 ][ "id" ] ] = tokens
            tools_tokens_source[ source ] = tools_tokens
        return tools_tokens_source

    @classmethod
    def get_tokens_from_source( self, row, source ):
        """
        Fetch tokens from different sources namely input and output files, names and desc of tools and 
        further help and EDAM sources
        """
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
        k = 1.75
        b = 0.75
        port_stemmer = PorterStemmer()
        refined_tokens_sources = dict()
        for source in tokens:
            refined_tokens = dict()
            files = dict()
            inverted_frequency = dict()
            file_id = -1
            total_file_length = 0
            for item in tokens[ source ]:
                file_id += 1
                file_tokens = tokens[ source ][ item ].split(" ")
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
                files[ item ] = term_frequency
            # parameters
            N = len(files)
            average_file_length = float( total_file_length ) / N

            # find BM25 score for each token of each tool. It helps to determine
            # how important each word is with respect to the tool and other tools
            for item in files:
                for token in files[ item ]:
                    file_item = files[ item ]
                    file_length = len( file_item )
                    tf = file_item[ token ]
                    idf = np.log2( N / len( inverted_frequency[ token ] ) )
                    alpha = ( 1 - b ) + ( float( b * file_length ) / average_file_length )
                    tf_star = tf * float( ( k + 1 ) ) / ( k * alpha + tf )
                    tf_idf = tf_star * idf
                    file_item[ token ] = tf_idf

            # filter tokens based on the BM25 scores. Not all tokens are important
            for item in files:
                file_item = files[ item ]
                sorted_x = sorted( file_item.items(), key=operator.itemgetter( 1 ), reverse=True )
                scores = [ score for ( token, score ) in sorted_x ]
                selected_tokens = [ ( port_stemmer.stem( token ), score ) for ( token, score ) in sorted_x if not utils._check_number( token ) and len( token ) > 2 ]
                selected_tokens_sorted = sorted( selected_tokens, key=operator.itemgetter( 1 ), reverse=True )
                refined_tokens[ item ] = selected_tokens_sorted
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
        tools_list = list()
        for source in documents_tokens:
            # create a unique list of all words
            all_tokens = list()
            doc_tokens = documents_tokens[ source ]
            for tool_item in doc_tokens:
                if tool_item not in tools_list:
                    tools_list.append( tool_item )
                for word_score in doc_tokens[ tool_item ]:
                    word = word_score[ 0 ]
                    if word not in all_tokens:
                        all_tokens.append( word )

            # create tools x tokens matrix containing respective frequency or relevance score for each term
            document_tokens_matrix= np.zeros( ( len( tools_list ), len( all_tokens ) ) )
            counter = 0
            for tool_item in doc_tokens:
                for word_score in doc_tokens[ tool_item ]:
                    word_index = [ token_index for token_index, token in enumerate( all_tokens ) if token == word_score[ 0 ] ][ 0 ]
                    # we take of score 1 if we need exact word matching for input and output file types.
                    # otherwise we take ranked scores for each token
                    document_tokens_matrix[ counter ][ word_index ] = 1 if source == 'input_output' else word_score[ 1 ]
                counter += 1
            document_tokens_matrix_sources[ source ] = document_tokens_matrix
        return document_tokens_matrix_sources, tools_list

    @classmethod
    def find_tools_cos_distance_matrix( self, document_token_matrix_sources, tools_list ):
        """
        Find similarity distance using Cosine distance among tools
        """
        mat_size = len( tools_list )
        similarity_matrix_sources = dict()
        for source in document_token_matrix_sources:
            sim_mat = document_token_matrix_sources[ source ]
            sim_scores = np.zeros( ( mat_size, mat_size ) )
            for index_x, item_x in enumerate( sim_mat ):
                for index_y, item_y in enumerate( sim_mat ):
                    # assign similarity score for a pair of tool using cosine angle bectween their vectors
                    sim_scores[ index_x ][ index_y ] = utils._angle( item_x, item_y )
            similarity_matrix_sources[ source ] = sim_scores
        return similarity_matrix_sources

    @classmethod
    def assign_similarity_importance( self, similarity_matrix_sources, tools_list, optimal_weights ):
        """
        Assign importance to the similarity scores coming for different sources
        """
        similarity_matrix_original = list()
        similarity_matrix_learned = list()
        all_tools = len( tools_list )
        
        for tool_index, tool in enumerate( tools_list ):
            sim_mat_original = np.zeros( all_tools )
            sim_mat_tool_learned = np.zeros( all_tools )
            for source in similarity_matrix_sources:
                # add up the similarity scores from each source weighted by a uniform prior
                sim_mat_original += self.uniform_prior * similarity_matrix_sources[ source ][ tool_index ]
                # add up the similarity scores from each source weighted by importance factors learned by machine leanring algorithm
                sim_mat_tool_learned += optimal_weights[ tools_list[ tool_index ] ][ source ] * similarity_matrix_sources[ source ][ tool_index ]
            similarity_matrix_original.append( sim_mat_original )
            similarity_matrix_learned.append( sim_mat_tool_learned )
            
        return similarity_matrix_original, similarity_matrix_learned

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe, tools_list ):
        """
        Get similar tools for each tool
        """
        tools_info = dict()
        for j, rowj in dataframe.iterrows():
            tools_info[ rowj[ "id" ] ] = rowj

        similarity_threshold = 0
        similarity = list()
        for index, item in enumerate( similarity_matrix ):
            tool_similarity = dict()
            scores = list()
            root_tool = {}
            tool_id = tools_list[ index ]
            for tool_index, tool_item in enumerate( tools_list ):
                rowj = tools_info[ tool_item ]
                score = round( item[ tool_index ], 2 )
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
                    if rowj[ "id" ] == tool_id:
                        root_tool = record
                    else:
                        scores.append( record )

            tool_similarity[ "root_tool" ] = root_tool
            sorted_scores = sorted( scores, key = operator.itemgetter( "score" ), reverse = True )
            # don't take all the tools predicted, just TOP something
            tool_similarity[ "similar_tools" ] = sorted_scores[ :self.tools_show ]
            similarity.append( tool_similarity )

        with open( 'similarity_matrix.json','w' ) as file:
            file.write( json.dumps( similarity ) )
            file.close()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print( "Usage: python predict_similarity.py <file_path> <number_of_iterations>" )
        exit( 1 )

    start_time = time.time()
    np.seterr( all = 'ignore' )
    tool_similarity = PredictToolSimilarity()
    dataframe = tool_similarity.read_file( sys.argv[ 1 ] )
    print "Read tool files"

    tokens = tool_similarity.extract_tokens( dataframe, [ 'input_output', 'name_desc', 'edam_help' ] )
    print "Extracted tokens"

    refined_tokens = tool_similarity.refine_tokens( tokens )
    print "Refined tokens"

    document_tokens_matrix, files_list = tool_similarity.create_document_tokens_matrix( refined_tokens )
    print "Created tools tokens matrix"

    print "Computing distance..."
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix, files_list )
    print "Computed distance"

    print "Learning optimal weights..."
    gd = gradientdescent.GradientDescentOptimizer( int( sys.argv[ 2 ] ) )
    optimal_weights, cost_tools, iterations = gd.gradient_descent( tools_distance_matrix, files_list )
    print "Optimal weights found"

    print "Assign importance to tools similarity matrix..."
    similarity_matrix_original, similarity_matrix_learned = tool_similarity.assign_similarity_importance( tools_distance_matrix, files_list, optimal_weights )
    
    print "Plotting the changes of costs during iterations..."
    utils._plot_tools_cost( cost_tools, iterations )

    print "Plots for learning..."
    utils._plots_original_learned_matrix( similarity_matrix_original, similarity_matrix_learned, files_list )

    print "Writing results to a JSON file..."
    tool_similarity.associate_similarity( similarity_matrix_learned, dataframe, files_list )
    
    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )
