import re
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *

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

def _plot_heatmap( similarity_matrix ):
    sns.heatmap(similarity_matrix, annot=True, fmt="g", cmap='viridis')
    plt.show()

def _angle(u, v):
  if linalg.norm(u) == 0 or linalg.norm(v) == 0:
      return 2e-10
  else:  
      return dot(u,v)/linalg.norm(u)/linalg.norm(v)
