import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json


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


extract_correlation( "data/similarity_scores_sources_optimal.json" )
