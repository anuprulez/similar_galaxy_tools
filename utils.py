import re
import numpy as np
import nltk
from nltk.stem import PorterStemmer

nltk.download('averaged_perceptron_tagger')
port_stemmer = PorterStemmer()
token_category_list = [ 'NNS', 'NN', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP' ]


def _get_text( row, attr ):
    return "" if type( row[ attr ] ) is float else str( row[ attr ] )


def _remove_special_chars( text ):
    return re.sub( '[^a-zA-Z0-9]', ' ', text )


def _remove_duplicate_file_types( tokens ):
    tokens_list = tokens.split( " " )
    unique_tokens = ''
    for item in tokens_list:
        if item not in unique_tokens and item is not "":
            unique_tokens = item if unique_tokens == "" else unique_tokens + " " + item
    return unique_tokens


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
    except Exception:
        return False


def l1_norm( vector ):
    norm = np.sum( vector )
    return 0 if norm == 0 else vector / norm


def _euclidean_distance( x, y ):
    x_norm = l1_norm( x )
    y_norm = l1_norm( y )
    return np.sqrt( np.sum( ( x_norm - y_norm ) ** 2 ) )


def _cosine_angle_score( vector1, vector2 ):
    """
    Get value of the cosine angle between two vectors
    """
    # if either of the vectors is zero, then similarity is also 0
    # which means the vectors cannot be compared
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
    dot_product = np.dot( vector1, vector2 )
    jaccard_denominator = np.dot( vector1, vector1 ) + np.dot( vector2, vector2 ) - dot_product
    if jaccard_denominator == 0:
        return 0
    else:
        return dot_product / float( jaccard_denominator )
