"""
Compute similarity among tools by analyzing the attributes and
finding proximity using BM25, Doc2Vec and Gradient Descent algorithms
"""

import sys
import os
import numpy as np
import operator
import json
import time

import utils
import extract_tokens
import learn_doc2vec
import gradientdescent


class ComputeToolSimilarity:

    @classmethod
    def __init__( self, tools_data_path ):
        self.data_source = [ 'input_output', 'name_desc_edam', 'help_text' ]
        self.tools_data_path = tools_data_path
        self.tools_show = 20

    def create_io_tokens_matrix( self, doc_tokens ):
        """
        Create document tokens matrix for input/output tokens
        """
        tools_list = list()
        counter = 0
        all_tokens = list()
        for tool_item in doc_tokens:
            if tool_item not in tools_list:
                tools_list.append( tool_item )
            for word_score in doc_tokens[ tool_item ]:
                word = word_score[ 0 ]
                if word not in all_tokens:
                    all_tokens.append( word )
        # create tools x tokens matrix containing respective frequency or relevance score for each term
        document_tokens_matrix = np.zeros( ( len( tools_list ), len( all_tokens ) ) )
        for tool_item in doc_tokens:
            for word_score in doc_tokens[ tool_item ]:
                word_index = [ token_index for token_index, token in enumerate( all_tokens ) if token == word_score[ 0 ] ][ 0 ]
                document_tokens_matrix[ counter ][ word_index ] = 1.0 # set flag at a word's index
            counter += 1
        return document_tokens_matrix

    @classmethod
    def find_io_similarity( self, input_output_tokens_matrix, tools_list ):
        """
        Find similarity distance between vectors for input/output tokens
        """
        own_similarity_score = 1.0
        mat_size = len( tools_list )
        sim_scores = np.zeros( [ mat_size, mat_size ] )
        sim_mat = input_output_tokens_matrix
        for index_x, item_x in enumerate( sim_mat ):
            tool_scores = sim_scores[ index_x ]
            for index_y, item_y in enumerate( sim_mat ):
                # compute similarity scores between two vectors
                tool_scores[ index_y ] = own_similarity_score if index_x == index_y else utils._jaccard_score( item_x, item_y )
        return sim_scores

    @classmethod
    def convert_similarity_as_list( self, similarity_matrix_sources, all_tools ):
        """
        Convert the similarity scores to list
        """
        all_tools_len = len( all_tools )
        similarity_matrix_similarity_sources = dict()
        for source in similarity_matrix_sources:
            similarity_matrix_dist = np.zeros( [ all_tools_len, all_tools_len] )
            similarity_matrix_src = similarity_matrix_sources[ source ]
            for index in range( all_tools_len ):
                similarity_matrix_dist[ index ][ : ] = similarity_matrix_src[ index ]
            similarity_matrix_similarity_sources[ source ] = similarity_matrix_dist
        return similarity_matrix_similarity_sources

    @classmethod
    def assign_similarity_importance( self, similarity_matrix_sources, tools_list, optimal_weights ):
        """
        Assign importance to the similarity scores coming for different sources
        """
        similarity_matrix_learned = list()
        all_tools = len( tools_list )
        similarity_tools = dict()
        similarity_scores_path = "data/similarity_scores_sources_optimal.json"
        for tool_index, tool in enumerate( tools_list ):
            tool_name = tools_list[ tool_index ] 
            similarity_tools[ tools_list[ tool_index ] ] = dict()
            sim_mat_tool_learned = np.zeros( all_tools )
            for source in similarity_matrix_sources:
                optimal_weight_source = optimal_weights[ tools_list[ tool_index ] ][ source ]
                tool_source_scores = similarity_matrix_sources[ source ][ tool_index ]
                similarity_tools[ tool_name ][ source ] = [ item for item in tool_source_scores ]
                # add up the similarity scores from each source weighted by importance factors learned by machine leanring algorithm
                sim_mat_tool_learned += optimal_weight_source * tool_source_scores
            similarity_tools[ tool_name ][ "optimal" ] = [ item for item in sim_mat_tool_learned ]
            similarity_matrix_learned.append( sim_mat_tool_learned )
        with open( similarity_scores_path, 'w' ) as file:
            file.write( json.dumps( similarity_tools ) )
        return similarity_matrix_learned

    @classmethod
    def compute_mutual_correlation( self, scores ):
        """
        Compute mutual correlation between sources
        """
        io_scores = list()
        nd_scores = list()
        ht_scores = list()
        for index, item in enumerate( scores ):
            io_scores.append( item[ "input_output_score" ] )
            nd_scores.append( item[ "name_desc_edam_score" ] )
            ht_scores.append( item[ "help_text_score" ] )
        io_nd_cov = np.correlate( io_scores, nd_scores, "same" )
        nd_ht_cov = np.correlate( nd_scores, ht_scores, "same" )
        io_ht_cov = np.correlate( io_scores, ht_scores, "same" )
        return io_nd_cov, nd_ht_cov, io_ht_cov

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe, tools_list, optimal_weights, cost_tools, original_matrix, learning_rates, uniform_cost_tools, gradients ):
        """
        Get similar tools for each tool
        """
        tools_info = dict()
        similarity = list()
        len_datasources = len( self.data_source )
        for j, rowj in dataframe.iterrows():
            tools_info[ rowj[ "id" ] ] = rowj
        for index, item in enumerate( similarity_matrix ):
            tool_similarity = dict()
            scores = list()
            root_tool = {}
            tool_id = tools_list[ index ]
            # row of similarity scores for a tool against all tools for all sources
            row_input_output = original_matrix[ self.data_source[ 0 ] ][ index ]
            row_name_desc = original_matrix[ self.data_source[ 1 ] ][ index ]
            row_help_text = original_matrix[ self.data_source[ 2 ] ][ index ]
            # sum the scores from multiple sources
            average_scores = [ ( x + y + z ) / len_datasources for x, y, z in zip( row_input_output, row_name_desc, row_help_text ) ]
            weighted_scores = item.tolist()
            # gradients for all the sources
            tool_gradients = gradients[ tool_id ]
            io_gradient = tool_gradients[ self.data_source[ 0 ] ]
            nd_gradient = tool_gradients[ self.data_source[ 1 ] ]
            ht_gradient = tool_gradients[ self.data_source[ 2 ] ]
            for tool_index, tool_item in enumerate( tools_list ):
                rowj = tools_info[ tool_item ]
                # optimal similarity score for a tool against a tool
                score = weighted_scores[ tool_index ]
                # similarity score with input and output file types
                input_output_score = row_input_output[ tool_index ]
                # similarity score with name, desc etc attributes
                name_desc_edam_score = row_name_desc[ tool_index ]
                help_text_score = row_help_text[ tool_index ]
                record = {
                   "name_description": rowj[ "name" ] + " " + ( utils._get_text( rowj, "description" ) ),
                   "id": rowj[ "id" ],
                   "input_types": utils._get_text( rowj, "inputs" ),
                   "output_types": utils._get_text( rowj, "outputs" ),
                   "what_it_does": utils._get_text( rowj, "help" ),
                   "edam_text": utils._get_text( rowj, "edam_topics" ),
                   "score": score,
                   "input_output_score": input_output_score,
                   "name_desc_edam_score": name_desc_edam_score,
                   "help_text_score": help_text_score
                }
                if rowj[ "id" ] == tool_id:
                    root_tool = record
                else:
                    scores.append( record )
            tool_similarity[ "root_tool" ] = root_tool
            sorted_scores = sorted( scores, key=operator.itemgetter( "score" ), reverse=True )[ : self.tools_show ]
            tool_similarity[ "similar_tools" ] = sorted_scores
            tool_similarity[ "optimal_weights" ] = optimal_weights[ tool_id ]
            tool_similarity[ "cost_iterations" ] = cost_tools[ tool_id ]
            tool_similarity[ "optimal_similar_scores" ] = weighted_scores
            tool_similarity[ "average_similar_scores" ] = average_scores
            tool_similarity[ "uniform_cost_tools" ] = uniform_cost_tools[ tool_id ]
            tool_similarity[ "combined_gradients" ] = [ np.sqrt( x ** 2 + y ** 2 + z ** 2 ) for x, y, z in zip( io_gradient, nd_gradient, ht_gradient ) ]
            tool_similarity[ "learning_rates_iterations" ] = learning_rates[ tool_id ]
            io_nd_corr, nd_ht_corr, io_ht_corr = self.compute_mutual_correlation( sorted_scores )
            tool_similarity[ "io_nd_corr" ] = io_nd_corr.tolist()
            tool_similarity[ "nd_ht_corr" ] = nd_ht_corr.tolist()
            tool_similarity[ "io_ht_corr" ] = io_ht_corr.tolist()
            similarity.append( tool_similarity )
        all_tools = dict()
        all_tools[ "list_tools" ] = tools_list
        similarity.append( all_tools )
        similarity_json = os.path.join( os.path.dirname( self.tools_data_path ) + '/' + 'similarity_matrix.json' )
        with open( similarity_json, 'w' ) as file:
            file.write( json.dumps( similarity ) )
            file.close()


if __name__ == "__main__":

    if len( sys.argv ) != 3:
        print( "Usage: python compute_similarity.py <file_path> <max_number_of_iterations>" )
        exit( 1 )

    start_time = time.time()
    np.seterr( all='ignore' )
    tool_similarity = ComputeToolSimilarity( sys.argv[ 1 ] )
    tokens = extract_tokens.ExtractTokens( sys.argv[ 1 ] )
    dataframe, refined_tokens = tokens.get_tokens( tool_similarity.data_source )
    nd_similarity = learn_doc2vec.Learn_Doc2Vec_Similarity( refined_tokens[ tool_similarity.data_source[ 1 ] ] )
    nd_similarity_matrix, tools_list = nd_similarity.learn_doc_similarity()
    ht_similarity = learn_doc2vec.Learn_Doc2Vec_Similarity( refined_tokens[ tool_similarity.data_source[ 2 ] ] )
    ht_similarity_matrix, tools_list = ht_similarity.learn_doc_similarity()

    io_doc_tokens = tool_similarity.create_io_tokens_matrix( refined_tokens[ tool_similarity.data_source[ 0 ] ] )
    io_similarity_matrix = tool_similarity.find_io_similarity( io_doc_tokens, tools_list )
    print( "Computed similarity matrices for all the sources" )

    distance_dict = dict()
    distance_dict[ tool_similarity.data_source[ 1 ] ] = nd_similarity_matrix
    distance_dict[ tool_similarity.data_source[ 2 ] ] = ht_similarity_matrix
    distance_dict[ tool_similarity.data_source[ 0 ] ] = io_similarity_matrix

    print( "Converting similarity as similarity distributions..." )
    similarity_as_list = tool_similarity.convert_similarity_as_list( distance_dict, tools_list )

    print( "Learning optimal weights..." )
    gd = gradientdescent.GradientDescentOptimizer( int( sys.argv[ 2 ] ), tool_similarity.data_source )
    optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients = gd.gradient_descent( similarity_as_list, tools_list )

    print( "Assign importance to tools similarity matrix..." )
    similarity_matrix_learned = tool_similarity.assign_similarity_importance( similarity_as_list, tools_list, optimal_weights )

    print( "Writing results to a JSON file..." )
    tool_similarity.associate_similarity( similarity_matrix_learned, dataframe, tools_list, optimal_weights, cost_tools, similarity_as_list, learning_rates, uniform_cost_tools, gradients )
    end_time = time.time()
    print( "Program finished in %d seconds" % int( end_time - start_time ) )
