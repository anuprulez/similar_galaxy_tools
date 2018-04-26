import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json


def read_files( file_path ):
    with open( file_path, 'r' ) as similarity_data:
        return json.loads( similarity_data.read() )


def plot_doc_tokens_mat( source ):
    # plot documents tokens matrix  
    doc_token_mat = read_files( "data/similarity_source_orig.json" )
    doc_token_mat = doc_token_mat[ source ]

    doc_token_mat_low = read_files( "data/similarity_source_low_rank.json" )
    doc_token_mat_low = doc_token_mat_low[ source ]

    plot_correlation( doc_token_mat, "Original documents tokens matrix", len( doc_token_mat ) )

    plot_correlation( doc_token_mat_low, "Low rank documents tokens matrix", len( doc_token_mat_low ) )
    '''plt.rcParams[ "font.serif" ] = "Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman"
    plt.rcParams[ "font.size" ] = 22
    plt.plot( io_tokens )
    plt.plot( nd_tokens )
    plt.plot( ht_tokens )
    plt.ylabel( 'Number of tokens' )
    plt.xlabel( 'Number of tools' )
    plt.title( 'Distribution of number of tokens for all the sources' )
    plt.legend( [ "Input and output", "Name and description", "Help text" ] )
    plt.grid( True )
    plt.show()'''


def plot_correlation( matrix, title, mat_size ):
    # plot correlation matrix
    font = { 'family' : 'sans serif', 'size': 16 }
    plt.rc( 'font', **font )
    #plt.matshow( matrix, cmap=plt.cm.Oranges )
    plt.imshow( matrix, cmap=plt.cm.Oranges )
    x_pos = np.arange( mat_size )
    y_pos = np.arange( mat_size )
    colorbar()
    plt.title( title )
    plt.show()

# sources: 'name_desc_edam', 'help_text'
plot_doc_tokens_mat( "input_output" )
plot_doc_tokens_mat( "name_desc_edam" )
plot_doc_tokens_mat( "help_text" )
