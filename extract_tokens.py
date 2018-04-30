"""
Extract useful tokens from multiple attributes of Galaxy tools
"""

import os
import numpy as np
import pandas as pd
import operator
import json

import utils


class ExtractTokens:

    @classmethod
    def __init__( self, tools_data_path ):
        self.tools_data_path = tools_data_path

    @classmethod
    def _read_file( self ):
        """
        Read the description of all tools
        """
        return pd.read_csv( self.tools_data_path )

    @classmethod
    def _extract_tokens( self, file, tokens_source ):
        """
        Extract tokens from the description of all tools
        """
        tools_tokens_source = dict()
        for source in tokens_source:
            tools_tokens = dict()
            for row in file.iterrows():
                tokens = self._get_tokens_from_source( row[ 1 ], source )
                tools_tokens[ row[ 1 ][ "id" ] ] = tokens
            tools_tokens_source[ source ] = tools_tokens
        return tools_tokens_source

    @classmethod
    def _get_tokens_from_source( self, row, source ):
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
        elif source == "help_text":
            tokens = utils._get_text( row, "help" )
        return utils._remove_special_chars( tokens )

    @classmethod
    def _refine_tokens( self, tokens ):
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
                sorted_tokens = sorted( tokens_scores, key=operator.itemgetter( 1 ), reverse=True )
                refined_tokens[ item ] = sorted_tokens
            tokens_file_name = 'tokens_' + source + '.txt'
            token_file_path = os.path.join( os.path.dirname( self.tools_data_path ) + '/' + tokens_file_name )
            with open( token_file_path, 'w' ) as file:
                file.write( json.dumps( refined_tokens ) )
                file.close()
            refined_tokens_sources[ source ] = refined_tokens
        return refined_tokens_sources

    @classmethod
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
                    if source == "input_output":
                        document_tokens_matrix[ counter ][ word_index ] = 1.0
                    else:
                        document_tokens_matrix[ counter ][ word_index ] = word_score[ 1 ]
                counter += 1
            document_tokens_matrix_sources[ source ] = document_tokens_matrix
        return document_tokens_matrix_sources, tools_list

    @classmethod
    def get_tokens( self, data_source ):
        """
        Get refined tokens
        """
        print( "Extracting tokens..." )
        dataframe = self._read_file()
        tokens = self._extract_tokens( dataframe, data_source )
        documents_tokens_matrix, tools_list = self.create_document_tokens_matrix( self._refine_tokens( tokens ) )
        return dataframe, documents_tokens_matrix, tools_list 
