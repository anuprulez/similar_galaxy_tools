import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json


def read_tokens( file_path ):
    with open( file_path, 'r' ) as similarity_data:
        return json.loads( similarity_data.read() )


def plot_tokens_size():
    # plot nubmer of tokens for each tool for all sources
    io_tokens = list()
    nd_tokens = list()
    ht_tokens = list()
    
    io_tools_tokens = read_tokens( "data/tokens_input_output.txt" )
    nd_tools_tokens = read_tokens( "data/tokens_name_desc_edam.txt" )
    ht_tools_tokens = read_tokens( "data/tokens_help_text.txt" )
    for index, item in enumerate( io_tools_tokens ):
        io_tokens.append( len( io_tools_tokens[ item ] ) )
        nd_tokens.append( len( nd_tools_tokens[ item ] ) )
        ht_tokens.append( len( ht_tools_tokens[ item ] ) )

    #print np.sum( io_tokens )
    #print np.sum( nd_tokens )
    #print np.sum( ht_tokens )
    plt.rcParams[ "font.serif" ] = "Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman"
    plt.rcParams[ "font.size" ] = 22
    plt.plot( io_tokens )
    plt.plot( nd_tokens )
    plt.plot( ht_tokens )
    plt.ylabel( 'Number of tokens' )
    plt.xlabel( 'Number of tools' )
    plt.title( 'Distribution of number of tokens for all the sources' )
    plt.legend( [ "Input and output", "Name and description", "Help text" ] )
    plt.grid( True )
    plt.show()


def plot_correlation( matrix, title, mat_size ):
    # plot correlation matrix
    font = { 'family' : 'sans serif', 'size': 16 }
    plt.rc( 'font', **font )
    plt.matshow( matrix, cmap=plt.cm.Oranges, vmin=0, vmax=1 )
    #groups = [ "LiRe", "LoRe", "BeRe", "LDA" ]
    x_pos = np.arange( mat_size )
    #plt.xticks( x_pos,groups)
    y_pos = np.arange( mat_size )
    #plt.yticks(y_pos,groups)
    colorbar()
    plt.title( title )
    plt.show()


def extract_correlation( file_path ):
    # extract correlation matrices from multiple sources
    # 'input_output', 'name_desc_edam', 'help_text'
    with open( file_path, 'r' ) as similarity_data:
        sim_data = json.loads( similarity_data.read() )
    mat_size = len( sim_data )
    sim_score_ht = np.zeros( [ mat_size, mat_size ] )
    sim_score_nd = np.zeros( [ mat_size, mat_size ] )
    sim_score_io = np.zeros( [ mat_size, mat_size ] )
    sim_score_op = np.zeros( [ mat_size, mat_size ] )
    tools = list()
    for index, item in enumerate( sim_data ):
        tools.append( item )
        sources_sim = sim_data[ item ]
        sim_score_ht[ index ][ : ] = sources_sim[ "help_text" ]
        sim_score_nd[ index ][ : ] = sources_sim[ "name_desc_edam" ]
        sim_score_io[ index ][ : ] = sources_sim[ "input_output" ]
        sim_score_op[ index ][ : ] = sources_sim[ "optimal" ]
    plot_correlation( sim_score_io, "Similarity scores based on input and output file types", mat_size )
    plot_correlation( sim_score_nd, "Similarity scores based on name and description", mat_size )
    plot_correlation( sim_score_ht, "Similarity scores based on helptext", mat_size )
    plot_correlation( sim_score_op, "Optimal similarity scores", mat_size )
        

def plot_venn_diag():
    # Plot venn for Jaccard Index
    plt.subplot(211)
    venn2(subsets = (4, 4, 3), set_labels = ('Top-k predicted classes', 'Actual k classes'))
    plt.title( "Top-k accuracy" )
    font = { 'family' : 'sans serif', 'size': 16 }
    plt.rc('font', **font)
    plt.show()


#extract_correlation( "data/similarity_scores_sources_optimal.json" )
plot_tokens_size()
