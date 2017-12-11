"""
Predict similarity among tools by analyzing the attributes and 
finding proximity using Best Match (BM25) algorithms
"""
import os
import numpy as np
import pandas as pd
import operator
import json
from math import *

import utils
import gradientdescent

class PredictToolSimilarity:

    @classmethod
    def __init__( self ):
        self.file_path = '/data/all_tools.csv'
        self.tools_show = 80

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
                scores = [ score for (token, score) in sorted_x ]
                mean_score = np.mean( scores )
                sigma = np.sqrt( np.var( scores ) )
                selected_tokens = [ ( token, score ) for ( token, score ) in sorted_x if not utils._check_number( token ) ]
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
                    sim_scores[ index_x ][ index_y ] = utils._angle( item_x, item_y )
            similarity_matrix_sources[ source ] = sim_scores
        return similarity_matrix_sources

    @classmethod
    def assign_similarity_importance( self, similarity_matrix_sources, tools_list, optimal_weights ):
        """
        Assign importance to the similarity scores coming for different sources
        """
        all_tools = len( tools_list )
        similarity_matrix = np.zeros( ( all_tools, all_tools ) )
        for source in similarity_matrix_sources:
            #similarity_matrix += np.dot( similarity_matrix_sources[ source ], optimal_weights[ source ].transpose() )
            similarity_matrix += optimal_weights[ source ] * similarity_matrix_sources[ source ]
        return similarity_matrix

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
            sorted_scores = sorted( scores[ :80 ], key=operator.itemgetter( "score" ), reverse=True )
            tool_similarity[ "root_tool" ] = root_tool
            tool_similarity[ "similar_tools" ] = sorted_scores
            similarity.append( tool_similarity )
            #print "Finished tool %d" % index

        with open( 'similarity_matrix.json','w' ) as file:
            file.write( json.dumps( similarity ) )
            file.close()


if __name__ == "__main__":
    np.seterr( all='ignore' )
    tool_similarity = PredictToolSimilarity()
    gd = gradientdescent.GradientDescentOptimizer()
    dataframe = tool_similarity.read_file()
    print "Read tool files"

    tokens = tool_similarity.extract_tokens( dataframe, [ 'input_output', 'name_desc', 'edam_help' ] )
    print "Extracted tokens"

    refined_tokens = tool_similarity.refine_tokens( tokens )
    print "Refined tokens"

    document_tokens_matrix, files_list = tool_similarity.create_document_tokens_matrix( refined_tokens )
    print "Created document term matrix"

    print "Computing distance..."
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix, files_list )
    print "Computed distance"

    print "Learning optimal weights..."
    optimal_weights, cost_tools, iterations = gd.gradient_descent( tools_distance_matrix, files_list )
    print "Optimal weights found..."

    print "Assign importance to similarity matrix..."
    similarity_matrix = tool_similarity.assign_similarity_importance( tools_distance_matrix, files_list, optimal_weights )

    print "Writing results to a JSON file..."
    tool_similarity.associate_similarity( similarity_matrix, dataframe, files_list )
    print "Listed the similar tools in a JSON file"

    print "Plotting the changes of costs during iterations..."
    utils._plot_tools_cost( cost_tools, iterations )

    print "Program finished"
