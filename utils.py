import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.stem import *

port_stemmer = PorterStemmer()
token_category_list = [ 'NNS', 'NN', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP' ]

def _get_text( row, attr ):
    return "" if type( row[ attr ] ) is float else str( row[ attr ] )

def _remove_special_chars( text ):
    return re.sub( '[^a-zA-Z0-9]', ' ', text )

def _clean_tokens( text_list ):
    # discard numbers and one letter words
    tokens = [ item.lower() for item in text_list if len( item ) > 1 and not _check_number( item ) ]
    # differentiate words based on their types as nouns, verbs etc
    tokens = nltk.pos_tag( tokens )
    # accept words that fall in the category mentioned (verbs, nouns)
    tokens = [ port_stemmer.stem( item ) for ( item, tag ) in tokens if tag in token_category_list ]
    return tokens

def _restore_space( text ):
    return " ".join( text.split( "," ) )

def _format_dict_string( dictionary ):
    string = ''
    for item in dictionary:
        string += '%s: %s' % ( item, dictionary[ item ] ) + '\n'
    return string

def _check_number( item ):
    try:
        int( item )
        return True
    except Exception as exception:
        return False

def _angle( vector1, vector2 ):
    """
    Get value of the cosine angle between two vectors
    """
    # if either of the vectors is zero, then cosine of the angle is also 0 
    # which means vectors are dissimilar 
    vec1_length = np.sqrt( np.dot( vector1, vector1 ) )
    vec2_length = np.sqrt( np.dot( vector2, vector2 ) )

    if vec1_length == 0 or vec2_length == 0:
        return 0
    else:  
        return np.dot( vector1, vector2 ) / ( vec1_length * vec2_length )

def _jaccard_score( vector1, vector2 ):
    """
    Get jaccard score for two vectors
    """
    intersection = [ 1 for a, b in zip( vector1, vector2 ) if a > 0 and b > 0 ]
    union = [ 1 for a, b in zip( vector1, vector2 ) if a > 0 or b > 0]
    union_len = len( union )

    if union_len == 0:
        return 0
    else:
        return float( len( intersection ) ) / union_len

def _plot_heatmap( similarity_matrix ):
    sns.heatmap(similarity_matrix, annot=True, fmt="g", cmap='viridis')
    plt.show()

def _plot_tools_cost( cost_tools, iterations ):
    """
    Generate plot for varying cost in iterations
    """
    mean_tool_cost = np.mean( cost_tools, 0 )
    x_axis = [ x for x in range( iterations ) ]
    plt.plot( x_axis, mean_tool_cost )
    plt.grid(color='k', linestyle='-', linewidth=0.25)
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Learned cost' )
    plt.show()

def _plot_learning_rate( learning_rates, iterations ):
    plt.plot( [ x for x in range( iterations ) ], learning_rates )
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Learning rates' )
    plt.grid( color='k', linestyle='-', linewidth=0.25 )
    plt.show()

def _plots_original_learned_matrix( matrix_original, matrix_learned, files_list ):
    """
    Generate plots for original, learned and difference of costs
    """
    
    all_tools = len( files_list )
    x_axis = [ x for x in range( all_tools ) ]

    for source in matrix_original:
        original_cost = list()
        learned_cost = list()
        diff_cost = list()
        # compute mean similarity score for each tool
        for index, item in enumerate( matrix_learned ):
            mean_learned = np.mean( item )
            mean_original = np.mean( matrix_original[ source ][ index ] )
            diff_cost.append( mean_learned - mean_original )
            learned_cost.append( mean_learned )
            original_cost.append( mean_original )

        # generate plots with their respective legends
        l_cost, = plt.plot( x_axis, learned_cost, 'b.', label = 'Learned cost' )
        o_cost, = plt.plot( x_axis, original_cost, 'r.', label = 'Original cost with ' + source )
        d_cost, = plt.plot( x_axis, diff_cost, 'g.', label = 'Learned - Original' )
        h_line, = plt.plot( x_axis, [ 0 for x in range( all_tools ) ], 'k.', label = 'Cost = 0' )

        plt.legend( handles = [ l_cost, o_cost, d_cost, h_line ] )
        plt.grid(color='k', linestyle='-', linewidth=0.15)
        plt.xlabel( 'Tools' )
        plt.ylabel( 'Cost with source ' + source )
        plt.show()
