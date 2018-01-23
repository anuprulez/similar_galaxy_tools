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

import utils


class PredictToolSimilarity:

    @classmethod
    def __init__( self, tools_data_path ):
        self.data_source = [ 'input_output_name_desc_edam_help' ]
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
        input_tokens = utils._restore_space( utils._get_text( row, "inputs" ) )
        input_tokens = utils._remove_duplicate_file_types( input_tokens )
        output_tokens = utils._restore_space( utils._get_text( row, "outputs" ) )
        output_tokens = utils._remove_duplicate_file_types( output_tokens )
        if input_tokens is not "" and output_tokens is not "":
            tokens = input_tokens + ' ' + output_tokens + ' '
        elif output_tokens is not "":
            tokens = output_tokens + ' '
        elif input_tokens is not "":
            tokens = input_tokens + ' '
        tokens += utils._restore_space( utils._get_text( row, "name" ) ) + ' '
        tokens += utils._restore_space( utils._get_text( row, "description" ) ) + ' '
        tokens += utils._get_text( row, "help" ) + ' '
        tokens += utils._get_text( row, "edam_topics" )
        return utils._remove_special_chars( tokens )

    @classmethod
    def refine_tokens( self, tokens ):
        """
        Refine the set of tokens by removing words like 'to', 'with'
        """
        k = 1.75
        b = 0.75
        refined_tokens_sources = dict()
        stop_words_file = "stop_words.txt"

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

            # filter tokens based on the BM25 scores. Not all tokens are important
            for item in files:
                file_tokens = files[ item ]
                tokens_scores = [ ( token, score ) for ( token, score ) in file_tokens.items() ]
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
            document_tokens_matrix= np.zeros( ( len( tools_list ), len( all_tokens ) ) )
            counter = 0
            for tool_item in doc_tokens:
                for word_score in doc_tokens[ tool_item ]:
                    word_index = [ token_index for token_index, token in enumerate( all_tokens ) if token == word_score[ 0 ] ][ 0 ]
                    document_tokens_matrix[ counter ][ word_index ] = word_score[ 1 ]
                counter += 1
            document_tokens_matrix_sources[ source ] = document_tokens_matrix
        return document_tokens_matrix_sources, tools_list

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
                    tool_scores[ index_y ] = utils._cosine_angle_score( item_x, item_y )
            similarity_matrix_sources[ source ] = sim_scores
        return similarity_matrix_sources

    @classmethod
    def convert_prob_distributions( self, similarity_matrix_sources, all_tools ):
        """
        Convert the similarity scores into log probability distributions
        """
        correct_sum = 1
        all_tools_len = len( all_tools )
        similarity_matrix_prob_dist_sources = dict()
        for source in similarity_matrix_sources:
            similarity_matrix_prob_dist = np.zeros( [ all_tools_len, all_tools_len] )
            similarity_matrix = similarity_matrix_sources[ source ]
            for index in range( all_tools_len ):
                row = similarity_matrix[ index ]
                row_sum = np.sum( row )
                row_sum = row_sum if row_sum > 0 else correct_sum
                prob_dist = [ float( item_similarity ) / row_sum for item_similarity in row ]
                similarity_matrix_prob_dist[ index ][ : ] = prob_dist
            similarity_matrix_prob_dist_sources[ source ] = similarity_matrix_prob_dist
        return similarity_matrix_prob_dist_sources

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe, tools_list ):
        """
        Get similar tools for each tool
        """
        tools_info = dict()
        similarity = list()
        for j, rowj in dataframe.iterrows():
            tools_info[ rowj[ "id" ] ] = rowj

        for index, item in enumerate( tools_list ):
            tool_similarity = dict()
            prob_similarity_scores = list()
            root_tool = {}
            tool_id = tools_list[ index ]
            # row of probability scores for a tool against all tools
            row_source_prob = similarity_matrix[ self.data_source[ 0 ] ][ index ]
            # find the mean probability scores to fill in the zero probability values
            mean_io_prob = np.mean( row_source_prob )
            row_source_updated_prob = [ item if item > 0 else mean_io_prob for item in row_source_prob ]

            for tool_index, tool_item in enumerate( tools_list ):
                rowj = tools_info[ tool_item ]
                record = {
                   "name_description": rowj[ "name" ] + " " + ( utils._get_text( rowj, "description" ) ),
                   "id": rowj[ "id" ],
                   "input_types": utils._get_text( rowj, "inputs" ),
                   "output_types": utils._get_text( rowj, "outputs" ),
                   "what_it_does": utils._get_text( rowj, "help" ),
                   "edam_text": utils._get_text( rowj, "edam_topics" ),
                   "score": row_source_updated_prob[ tool_index ],
                   "source_prob": row_source_updated_prob[ tool_index ],
                }
                if rowj[ "id" ] == tool_id:
                    root_tool = record
                else:
                    prob_similarity_scores.append( record )

            tool_similarity[ "root_tool" ] = root_tool
            sorted_prob_similarity_scores = sorted( prob_similarity_scores, key=operator.itemgetter("score"), reverse=True )[:self.tools_show ]         
            tool_similarity[ "source_prob_similar_tools" ] = sorted_prob_similarity_scores
            tool_similarity[ "source_prob_dist" ] = row_source_prob.tolist()
            similarity.append( tool_similarity )
            print "Tool index: %d finished" % index
        print "Writing results to a JSON file..."
        all_tools = dict()
        all_tools[ "list_tools" ] = tools_list
        similarity.append( all_tools )
        similarity_json = os.path.join( os.path.dirname( self.tools_data_path ) + '/' + 'similarity_matrix.json' )
        with open( similarity_json, 'w' ) as file:
            file.write( json.dumps( similarity ) )
            file.close()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print( "Usage: python predict_similarity.py <file_path>" )
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

    print "Computing similarity..."
    start_time_similarity_comp = time.time()
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix, files_list )
    end_time_similarity_comp = time.time()
    print "Computed similarity in %d seconds" % int( end_time_similarity_comp - start_time_similarity_comp )

    print "Converting similarities to probability distributions..."
    similarity_matrix_prob_dist_sources = tool_similarity.convert_prob_distributions( tools_distance_matrix, files_list )

    print "Computing probability scores..."
    tool_similarity.associate_similarity( similarity_matrix_prob_dist_sources, dataframe, files_list )

    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )
