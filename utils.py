import re
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *


# Methods for cleaning/ formatting data from xml files of tools
def _get_text( row, attr ):
    return "" if type( row[ attr ] ) is float else str( row[ attr ] )

def _remove_special_chars( text ):
    return re.sub( '[^a-zA-Z0-9]', ' ', text )

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

# Get cosine of angle between two vectors
def _angle( vector1, vector2 ):
  if linalg.norm( vector1 ) == 0 or linalg.norm( vector2 ) == 0:
      return 0
  else:  
      return dot( vector1, vector2 ) / ( linalg.norm( vector1 ) * linalg.norm( vector2 ) )

# Generate plots
def _plot_heatmap( similarity_matrix ):
    sns.heatmap(similarity_matrix, annot=True, fmt="g", cmap='viridis')
    plt.show()

def _plot_tools_cost( cost_tools, iterations ):
    plt.plot( [ x for x in range( iterations ) ], cost_tools )
    plt.xlabel( 'Number of iterations' )
    plt.ylabel( 'Cost' )
    plt.show()


