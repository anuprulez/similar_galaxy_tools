"""
Predict similarity among tools by analyzing the attributes and
finding proximity using Best Match (BM25) and Gradient Descent algorithms
"""
import sys
import os
import re
import numpy as np
import pandas as pd
import operator
import json
import time
import gensim
from gensim.models.doc2vec import TaggedDocument

import utils
import gradientdescent


class PredictToolSimilarity:

    @classmethod
    def __init__( self, tools_data_path ):
        self.data_source = [ 'input_output', 'name_desc_edam_help' ]
        self.tools_data_path = tools_data_path
        self.tools_show = 40

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
        elif source == 'name_desc_edam_help':
            tokens = utils._restore_space( utils._get_text( row, "name" ) ) + ' '
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
                if source in "name_desc_edam_help":
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
                tokens_scores = [ token for ( token, score ) in file_tokens.items() ]
                #sorted_tokens = sorted( tokens_scores, key=operator.itemgetter( 1 ), reverse=True )
                refined_tokens[ item ] = tokens_scores
            tokens_file_name = 'tokens_' + source + '.txt'
            token_file_path = os.path.join( os.path.dirname( self.tools_data_path ) + '/' + tokens_file_name )
            with open( token_file_path, 'w' ) as file:
                file.write( json.dumps( refined_tokens ) )
                file.close()
            refined_tokens_sources[ source ] = refined_tokens
        return refined_tokens_sources

    def tag_document( self, tokens_sources ):
        """
        Get tagged documents
        """
        tagged_doc = []
        tools_list = list()
        for source in tokens_sources:
            doc_tokens = tokens_sources[ source ]
            if source == "name_desc_edam_help":
                tool_counter = 0
                for tool in doc_tokens:
                    if tool not in tools_list:
                        tools_list.append( tool )
                    tokens = doc_tokens[ tool ]
                    td = TaggedDocument( gensim.utils.to_unicode(' '.join( tokens )).split(), [ tool_counter ] )
                    tagged_doc.append( td )
                    tool_counter += 1
        return tagged_doc, tools_list

    def find_document_similarity( self, tagged_documents ):
        """
        Find the similarity among documents by training a neural network (Doc2Vec)
        """
        training_epochs = 20
        model = gensim.models.Doc2Vec( tagged_documents, dm = 0, alpha=0.05, size= 100, min_alpha=0.025, min_count=0 )
        for epoch in range( training_epochs ):
            if epoch % 20 == 0:
                print ( 'Training epoch %s' % epoch )
            model.train( tagged_documents, total_examples=model.corpus_count, epochs=model.iter )
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        tools_similarity_dict = dict()
        tools_similarity = list()
        for index in range( len( tagged_documents ) ):
            similarity = model.docvecs.most_similar( index )
            sum_scores = np.sum( [ score for ( item, score ) in similarity ] )
            sum_scores = 1.0 if sum_scores == 0 else float( sum_scores )
            sim_scores = [ ( int( item_id ), score / sum_scores ) for ( item_id, score ) in similarity ]
            sim_scores = sorted( sim_scores, key=operator.itemgetter( ( 0 ) ), reverse=False )
            tools_similarity.append( sim_scores )
        return tools_similarity

    '''def create_document_tokens_matrix( self, documents_tokens ):
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
                    if source == "input_output":
                        pair_score = utils._jaccard_score( item_x, item_y )
                    else:
                        pair_score = utils._cosine_angle_score( item_x, item_y )
                    tool_scores[ index_y ] = pair_score
            similarity_matrix_sources[ source ] = sim_scores
        return similarity_matrix_sources

    @classmethod
    def convert_prob_distributions( self, similarity_matrix_sources, all_tools ):
        """
        Convert the similarity scores into probability distributions
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
        return similarity_matrix_sources, similarity_matrix_learned'''

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe, tools_list ):
        """
        Get similar tools for each tool
        """
        tools_info = dict()
        similarity_threshold = 0
        similarity = list()
        for j, rowj in dataframe.iterrows():
            tools_info[ rowj[ "id" ] ] = rowj
            
        for index, item in enumerate( similarity_matrix ):
            tool_similarity = dict()
            scores = list()
            root_tool = tools_list[ index ]
            root_row = tools_info[ root_tool ]
            # row of similarity scores for a tool against all tools
            #optimal_normalized_scores = item
            for similar_tool in item:
                similar_tool_id = similar_tool[ 0 ]
                rowj = tools_info[ tools_list[ similar_tool_id ] ]
                score = similar_tool[ 1 ]
                # take similar tools found using Gradient Descent + BM25
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
            root_record = {
                   "name_description": root_row[ "name" ] + " " + ( utils._get_text( root_row, "description" ) ),
                   "id": root_row[ "id" ],
                   "input_types": utils._get_text( root_row, "inputs" ),
                   "output_types": utils._get_text( root_row, "outputs" ),
                   "what_it_does": utils._get_text( root_row, "help" ),
                   "edam_text": utils._get_text( root_row, "edam_topics" ),
                   "score": 1
            }
            scores.append( root_record )
            tool_similarity[ "root_tool" ] = root_tool
            sorted_scores = sorted( scores, key=operator.itemgetter( "score" ), reverse=True )
            # don't take all the tools predicted, just TOP something
            tool_similarity[ "similar_tools" ] = sorted_scores
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

    similarity_matrix, files_list = tool_similarity.tag_document( refined_tokens )
    print "Created tools similarity matrix"

    tagged_doc, tools_list = tool_similarity.tag_document( refined_tokens )
    print "Created tools similarity matrix"

    learned_simiarity_matrix = tool_similarity.find_document_similarity( tagged_doc )
    print "Learned similarity using neural networks"

    print "Writing results to a JSON file..."
    tool_similarity.associate_similarity( learned_simiarity_matrix, dataframe, tools_list )
    #end_time = time.time()
    #print "Program finished in %d seconds" % int( end_time - start_time )

    '''print "Computing similarity..."
    start_time_similarity_comp = time.time()
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix, files_list )
    end_time_similarity_comp = time.time()
    print "Computed similarity in %d seconds" % int( end_time_similarity_comp - start_time_similarity_comp )

    print "Converting similarities to probability distributions..."
    similarity_matrix_prob_dist_sources = tool_similarity.convert_prob_distributions( tools_distance_matrix, files_list )

    print "Learning optimal weights..."
    gd = gradientdescent.GradientDescentOptimizer( int( sys.argv[ 2 ] ) )
    optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients = gd.gradient_descent( similarity_matrix_prob_dist_sources, tools_distance_matrix, files_list )

    print "Assign importance to tools similarity matrix..."
    similarity_matrix_original, similarity_matrix_learned = tool_similarity.assign_similarity_importance( similarity_matrix_prob_dist_sources, files_list, optimal_weights )

    print "Writing results to a JSON file..."
    tool_similarity.associate_similarity( similarity_matrix_learned, dataframe, files_list, optimal_weights, cost_tools, similarity_matrix_original, learning_rates, uniform_cost_tools, gradients )
    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )'''
