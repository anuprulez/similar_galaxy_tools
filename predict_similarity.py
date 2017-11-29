import os
import re
import numpy as np
import pandas as pd
import operator

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity


class PredictToolSimilarity:

    def __init__( self ):
        self.file_path = '/data/all_tools.csv'

    @classmethod
    def read_file( self ):
        """
        Read the description of all tools
        """
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname + '/data')
        return pd.read_csv( 'all_tools.csv' )

    @classmethod
    def extract_tokens( self, file ):
        """
        Extract tokens from the description of all tools
        """
        tools_tokens = list()
        for item in file.iterrows():
            row = item[ 1 ]
            # float type is empty
            description = "" if type(row[ "description" ]) is float else str( row[ "description" ] )
            inputs = "" if type(row[ "inputs" ]) is float else str( row[ "inputs" ] )
            name = "" if type(row[ "name" ]) is float else str( row[ "name" ] )
            help = "" if type(row[ "help" ]) is float else str( row[ "help" ] )
            inputs = inputs.split( "," )
            inputs = " ".join( inputs )
            # append all tokens
            tokens = description + " " + inputs + " " + name + " " + help
            #tokens = inputs + " " + name
            # remove special characters from tokens
            tokens = re.sub( '[^a-zA-Z0-9\n\.]', ' ', tokens )
            tools_tokens.append( tokens )
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
        k = 1.75
        b = 0.75

        # find BM25 score for each token of each tool. It helps to determine
        # how important each word is with respect to the tool and other tools
        for item in files:
            for token in item:
                file_length = len( item )
                tf = item[ token ]
                idf = np.log2( N / len( inverted_frequency[ token ] ) )
                tf_star = tf * float( ( k + 1 ) ) / ( k * ( 1 - b + float( b * file_length ) / average_file_length ) + tf )
                tf_idf = tf_star * idf
                item[ token ] = tf_idf

        # filter tokens based on the BM25 scores. Not all tokens are important
        for item in files:
            sorted_x = sorted( item.items(), key=operator.itemgetter(1), reverse=True )
            scores = [ score for (token, score) in sorted_x ]
            mean_score = np.mean( scores )
            selected_tokens = [ token for (token, score) in sorted_x if score >= mean_score ]
            selected_tokens = " ".join( selected_tokens )
            refined_tokens.append( selected_tokens )
        return refined_tokens

    @classmethod
    def create_document_tokens_matrix( self, tokens ):
        """
        Create document tokens matrix
        """
        token_vector = CountVectorizer( analyzer='word' )
        all_tokens = token_vector.fit_transform( tokens )
        document_token_matrix = pd.DataFrame( all_tokens.toarray(), columns=token_vector.get_feature_names() )
        document_token_matrix.to_csv( 'dtmatrix.csv' )
        return document_token_matrix

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
        return similarity_matrix

    @classmethod
    def associate_similarity( self, similarity_matrix, dataframe ):
        """
        Get similar tools for each tool
        """
        count_items = dataframe.count()[ 0 ]
        how_many_similar_tools = 5
        similarity = list()
        for i, rowi in dataframe.iterrows():
            file_similarity = dict()
            scores = list()
            for j, rowj in dataframe.iterrows():
                record = dict()
                record[ "name_description" ] = rowj[ "name" ] + " " + ( "" if type( rowj[ "description" ] ) is float else str( rowj[ "description" ] ) )
                record[ "id" ] = rowj[ "id" ]
                record[ "input_types" ] = "" if type( rowj[ "inputs" ] ) is float else str( rowj[ "inputs" ] )
                record[ "what_it_does" ] = "" if type( rowj[ "help" ] ) is float else str( rowj[ "help" ] )
                record[ "score" ] = round( similarity_matrix[ i ][ j ], 2 )
                scores.append( record )
            file_similarity[ "scores" ] = scores
            file_similarity[ "id" ] = rowi[ "id" ]
            similarity.append( file_similarity )

        similar_tools_result = ''
        for file_index in range( count_items ):
            analyze_file = similarity[ file_index ]
            sorted_tools = sorted( analyze_file[ "scores" ], key=operator.itemgetter( "score" ), reverse=True )
            for index, item in enumerate( sorted_tools[ :how_many_similar_tools ] ):
                if index == 0:
                    similar_tools_result += "Tool: '\n'"
                    similar_tools_result += str( item ) + '\n\n'
                    similar_tools_result += "Similar tools: '\n'"
                else:
                    similar_tools_result += str( item ) + '\n'
            similar_tools_result += "----------------------------------------------------------------------------------------------- '\n\n'"

        with open( 'tool_results.txt','w' ) as file:
            file.write( similar_tools_result )
            file.close()

   
if __name__ == "__main__":
    tool_similarity = PredictToolSimilarity()
    dataframe = tool_similarity.read_file()
    tokens = tool_similarity.extract_tokens( dataframe )
    refined_tokens = tool_similarity.refine_tokens( tokens )
    document_tokens_matrix = tool_similarity.create_document_tokens_matrix( refined_tokens )
    tools_distance_matrix = tool_similarity.find_tools_eu_distance_matrix( document_tokens_matrix )
    print "Computing cosine distance..."
    tools_distance_matrix = tool_similarity.find_tools_cos_distance_matrix( document_tokens_matrix )
    tool_similarity.associate_similarity( tools_distance_matrix, dataframe )

