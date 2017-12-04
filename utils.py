import re
import matplotlib.pyplot as plt
import seaborn as sns

def _get_text( row, attr ):
    return "" if type( row[ attr ] ) is float else str( row[ attr ] )

def _remove_special_chars( text ):
    return re.sub( '[^a-zA-Z0-9\n\.]', ' ', text )

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

def _check_quality_scores( mean, sigma, score ):
    return ( score >= mean - 2 * sigma )

def _plot_heatmap( similarity_matrix ):
    sns.heatmap(similarity_matrix, annot=True, fmt="g", cmap='viridis')
    plt.show()
    #sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
    '''plt.imshow(similarity_matrix, interpolation='nearest')
    plt.matshow(similarity_matrix, cmap='hot')
    plt.show()
    for item in range(0, 5):
        print similarity_matrix[item][0:5]'''
    
