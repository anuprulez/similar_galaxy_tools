"""
Predict similarity among tools by analyzing the attributes and
finding proximity using Best Match (BM25) and Gradient Descent algorithms
"""
import sys
import os
import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import svd
import pandas as pd
import operator
import json
import time

import utils
import gradientdescent


class PredictToolSimilarity:

    @classmethod
    def __init__( self, tools_data_path ):
        self.data_source = [ 'input_output', 'name_desc_edam', 'help_text' ]
        self.tools_data_path = tools_data_path
        self.tools_show = 20

    @classmethod
    def read_file( self ):
        """
        Read the description of all tools
        """
        return pd.read_csv( self.tools_data_path )

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
            # remove duplicate file type individually from input and output file types and merge
            input_tokens = utils._restore_space( utils._get_text( row, "inputs" ) )
            input_tokens = utils._remove_duplicate_file_types( input_tokens )
            output_tokens = utils._restore_space( utils._get_text( row, "outputs" ) )
            output_tokens = utils._remove_duplicate_file_types( output_tokens )
            if input_tokens is not "" and output_tokens is not "":
                tokens = input_tokens + ' ' + output_tokens
            elif output_tokens is not "":
                tokens = output_tokens
            elif input_tokens is not "":
                tokens = input_tokens
        elif source == 'name_desc_edam':
            tokens = utils._restore_space( utils._get_text( row, "name" ) ) + ' '
            tokens += utils._restore_space( utils._get_text( row, "description" ) ) + ' '
            tokens += utils._get_text( row, "edam_topics" )
        elif source == 'help_text':
            tokens += utils._get_text( row, "help" )
        return utils._remove_special_chars( tokens )

    @classmethod
    def reject_outlier_tokens( self, tokens_list ):
        """
        Reject tokens which have low TF-IDF scores i.e. outliers
        """
        scores = [ score for ( token, score ) in tokens_list ]
        standard_deviation = np.std( scores )
        mean = np.mean( scores )
        shortened_tokens_list = [ ( token, score ) for ( token, score ) in tokens_list if ( abs( score - mean ) < 2 * standard_deviation ) ]
        return shortened_tokens_list

    @classmethod
    def refine_tokens( self, tokens ):
        """
        Refine the set of tokens by removing words like 'to', 'with'
        """
        k = 1.75
        b = 0.75
        stop_words_file = "stop_words.txt"
        all_stopwords = list()
        refined_tokens_sources = dict()
        # collect all the stopwords
        with open( stop_words_file ) as file:
            lines = file.read()
            all_stopwords = lines.split( "\n" )
        for source in tokens:
            refined_tokens = dict()
            files = dict()
            inverted_frequency = dict()
            file_id = -1
            total_file_length = 0
            for item in tokens[ source ]:
                file_id += 1
                file_tokens = tokens[ source ][ item ].split(" ")
                if source in "name_desc_edam" or source in "help_text":
                    file_tokens = utils._clean_tokens( file_tokens, all_stopwords )
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
            N = len( files )
            average_file_length = float( total_file_length ) / N
            # find BM25 score for each token of each tool. It helps to determine
            # how important each word is with respect to the tool and other tools
            for item in files:
                file_item = files[ item ]
                file_length = len( file_item )
                for token in file_item:
                    tf = file_item[ token ]
                    # normalize the term freq of token for each document
                    tf = float( tf ) / file_length
                    idf = np.log2( N / len( inverted_frequency[ token ] ) )
                    alpha = ( 1 - b ) + ( float( b * file_length ) / average_file_length )
                    tf_star = tf * float( ( k + 1 ) ) / ( k * alpha + tf )
                    tf_idf = tf_star * idf
                    file_item[ token ] = tf_idf
            # filter tokens based on the BM25 scores and stop words. Not all tokens are important
            for item in files:
                file_tokens = files[ item ]
                tokens_scores = [ ( token, score ) for ( token, score ) in file_tokens.items() ]
                if len( tokens_scores ) > 0:
                    tokens_scores = self.reject_outlier_tokens( tokens_scores )
                sorted_tokens = sorted( tokens_scores, key=operator.itemgetter( 1 ), reverse=True )
                refined_tokens[ item ] = sorted_tokens
            tokens_file_name = 'tokens_' + source + '.txt'
            token_file_path = os.path.join( os.path.dirname( self.tools_data_path ) + '/' + tokens_file_name )
            with open( token_file_path, 'w' ) as file:
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
            document_tokens_matrix = np.zeros( ( len( tools_list ), len( all_tokens ) ) )
            counter = 0
            for tool_item in doc_tokens:
                for word_score in doc_tokens[ tool_item ]:
                    word_index = [ token_index for token_index, token in enumerate( all_tokens ) if token == word_score[ 0 ] ][ 0 ]
                    document_tokens_matrix[ counter ][ word_index ] = word_score[ 1 ]
                counter += 1
            document_tokens_matrix_sources[ source ] = document_tokens_matrix
        return document_tokens_matrix_sources, tools_list

    @classmethod
    def compute_low_rank_matrix( self, u, s, v, rank ):
        """
        Compute low rank approximation of a full rank matrix
        """
        u_approx = u[ :, :rank ]
        s_approx = s[ :rank ]
        sum_taken_percent = np.sum( s_approx ) / float( np.sum( s ) )
        s_approx = np.diag( np.array( s_approx ) )
        v_approx = v[ :rank, : ]
        return [ u_approx.dot( s_approx ).dot( v_approx ), sum_taken_percent ]

    @classmethod
    def find_optimal_low_rank_matrix( self, orig_similarity_matrix, orig_rank, u, s, v, singular_reduction ):
        """
        Find the rank which captures most of the information from the original full rank matrix
        """
        '''rank_list = list()
        sum_singular_values = list()
        for rank in range( 0, orig_rank ):
            compute_result = self.compute_low_rank_matrix( u, s, v, rank + 1 )
            rank_list.append( ( rank + 1 ) / float( orig_rank ) )
            sum_singular_values.append( compute_result[ 1 ] )
        utils._plot_singular_values_rank( rank_list, sum_singular_values )'''
        return self.compute_low_rank_matrix( u, s, v, int( singular_reduction * orig_rank ) )

    @classmethod
    def factor_matrices( self, document_tokens_matrix_sources ):
        """
        Latent semantic indexing
        """
        print "Computing lower rank representations of similarity matrices..."
        approx_similarity_matrices = dict()
        # it is number determined heuristically which helps capturing most of the
        # information in the original similarity matrices
        singular_reduction = 0.4
        for source in document_tokens_matrix_sources:
            similarity_matrix = document_tokens_matrix_sources[ source ]
            mat_rnk = matrix_rank( similarity_matrix )
            u, s, v = svd( similarity_matrix )
            # sort the singular values in descending order. Top ones most important
            s = sorted( s, reverse=True )
            compute_result = self.find_optimal_low_rank_matrix( similarity_matrix, mat_rnk, u, s, v, singular_reduction )
            approx_similarity_matrices[ source ] = compute_result[ 0 ]
        return approx_similarity_matrices

    @classmethod
    def find_tools_cos_distance_matrix( self, document_token_matrix_sources, tools_list ):
        """
        Find similarity distance using cosine distance among tools
        """
        mat_size = len( tools_list )
        similarity_matrix_sources = dict()
        for source in document_token_matrix_sources:
            print "Computing similarity scores for source %s..." % source
            sim_mat = document_token_matrix_sources[ source ]
            sim_scores = np.zeros( ( mat_size, mat_size ) )
            for index_x, item_x in enumerate( sim_mat ):
                tool_scores = sim_scores[ index_x ]
                for index_y, item_y in enumerate( sim_mat ):
                    # compute similarity scores between two vectors
                    if source == "input_output":
                        pair_score = utils._jaccard_score( item_x, item_y )
                    else:
                        pair_score = utils._cosine_angle_score( item_x, item_y )
                    tool_scores[ index_y ] = pair_score
            similarity_matrix_sources[ source ] = sim_scores
        return similarity_matrix_sources

    @classmethod
    def convert_similarity_as_list( self, similarity_matrix_sources, all_tools ):
        """
        Convert the similarity scores into similarity distributions and take only positive values
        """
        all_tools_len = len( all_tools )
        similarity_matrix_prob_dist_sources = dict()
        for source in similarity_matrix_sources:
            similarity_matrix_prob_dist = np.zeros( [ all_tools_len, all_tools_len] )
            similarity_matrix = similarity_matrix_sources[ source ]
            for index in range( all_tools_len ):
                row = similarity_matrix[ index ]
                # take only positive values
                similarity_matrix_prob_dist[ index ][ : ] = [ item if item > 0.0 else 0.0 for item in row ]
            similarity_matrix_prob_dist_sources[ source ] = similarity_matrix_prob_dist
        return similarity_matrix_prob_dist_sources

    @classmethod
    def assign_similarity_importance( self, similarity_matrix_sources, tools_list, optimal_weights ):
        """
        Assign importance to the similarity scores coming for different sources
        """
        similarity_matrix_learned = list()
        all_tools = len( tools_list )
        for tool_index, tool in enumerate( tools_list ):
            sim_mat_tool_learned = np.zeros( all_tools )
            for source in similarity_matrix_sources:
                optimal_weight_source = optimal_weights[ tools_list[ tool_index ] ][ source ]
                # add up the similarity scores from each source weighted by importance factors learned by machine leanring algorithm
                sim_mat_tool_learned += optimal_weight_source * similarity_matrix_sources[ source ][ tool_index ]
            similarity_matrix_learned.append( sim_mat_tool_learned )
        return similarity_matrix_learned

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
            # row of similarity scores for a tool against all tools
            row_input_output = original_matrix[ self.data_source[ 0 ] ][ index ]
            row_name_desc = original_matrix[ self.data_source[ 1 ] ][ index ]
            row_help_text = original_matrix[ self.data_source[ 2 ] ][ index ]
            # sum the scores from multiple sources
            average_normalized_scores = [ ( x + y + z ) / len_datasources for x, y, z in zip( row_input_output, row_name_desc, row_help_text ) ]
            optimal_normalized_scores = item.tolist()
            tool_gradients = gradients[ tool_id ]
            io_gradient = tool_gradients[ self.data_source[ 0 ] ]
            nd_gradient = tool_gradients[ self.data_source[ 1 ] ]
            ht_gradient = tool_gradients[ self.data_source[ 2 ] ]
            for tool_index, tool_item in enumerate( tools_list ):
                rowj = tools_info[ tool_item ]
                # optimal similarity score for a tool against a tool
                score = optimal_normalized_scores[ tool_index ]
                # similarity score with input and output file types
                input_output_score = row_input_output[ tool_index ]
                # similarity score with name, desc etc attributes
                name_desc_edam_score = row_name_desc[ tool_index ]
                # similarity score with help text
                helptext_score = row_help_text[ tool_index ]
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
                   "help_text_score": helptext_score
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
            tool_similarity[ "optimal_similar_scores" ] = optimal_normalized_scores
            tool_similarity[ "average_similar_scores" ] = average_normalized_scores
            tool_similarity[ "uniform_cost_tools" ] = uniform_cost_tools[ tool_id ]
            tool_similarity[ "combined_gradients" ] = [ np.sqrt( x ** 2 + y ** 2 + z ** 2 ) for x, y, z in zip( io_gradient, nd_gradient, ht_gradient ) ]
            similarity.append( tool_similarity )
        all_tools = dict()
        all_tools[ "list_tools" ] = tools_list
        similarity.append( all_tools )
        similarity_json = os.path.join( os.path.dirname( self.tools_data_path ) + '/' + 'similarity_matrix.json' )
        with open( similarity_json, 'w' ) as file:
            file.write( json.dumps( similarity ) )
            file.close()


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print( "Usage: python predict_similarity.py <file_path> <max_number_of_iterations>" )
        exit( 1 )

    start_time = time.time()
    np.seterr( all='ignore' )
    tool_similarity = PredictToolSimilarity( sys.argv[ 1 ] )
    dataframe = tool_similarity.read_file()
    print "Read tool files"

    tokens = tool_similarity.extract_tokens( dataframe, tool_similarity.data_source )
    print "Extracted tokens"

    refined_tokens = tool_similarity.refine_tokens( tokens )
    print "Refined tokens"

    document_tokens_matrix, files_list = tool_similarity.create_document_tokens_matrix( refined_tokens )
    print "Created tools tokens matrix"

    document_tokens_matrix_factored = tool_similarity.factor_matrices( document_tokens_matrix )
    print "Matrices factored"

    print "Computing similarity..."
    start_time_similarity_comp = time.time()
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix_factored, files_list )
    end_time_similarity_comp = time.time()
    print "Computed similarity in %d seconds" % int( end_time_similarity_comp - start_time_similarity_comp )

    print "Converting similarity as similarity distributions..."
    similarity_as_list = tool_similarity.convert_similarity_as_list( tools_distance_matrix, files_list )

    print "Learning optimal weights..."
    gd = gradientdescent.GradientDescentOptimizer( int( sys.argv[ 2 ] ) )
    optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients = gd.gradient_descent( similarity_as_list, files_list )

    print "Assign importance to tools similarity matrix..."
    similarity_matrix_learned = tool_similarity.assign_similarity_importance( similarity_as_list, files_list, optimal_weights )

    print "Writing results to a JSON file..."
    tool_similarity.associate_similarity( similarity_matrix_learned, dataframe, files_list, optimal_weights, cost_tools, similarity_as_list, learning_rates, uniform_cost_tools, gradients )
    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )
