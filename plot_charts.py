import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json

# Font close to Times New Roman
# https://mondaybynoon.com/linux-font-equivalents-to-popular-web-typefaces/

FONT_SIZE = 26
plt.rcParams["font.family"] = "FreeSerif"
plt.rc('text', usetex=True)
plt.rcParams[ 'text.latex.preamble' ]=[r"\usepackage{amsmath}"]
plt.rcParams[ "font.size" ] = FONT_SIZE
colors_dict = dict()
colors_dict[ "help_text" ] = 'C2'
colors_dict[ "name_desc_edam" ] = 'C1'
colors_dict[ "input_output" ] = 'C0'

def read_files( file_path ):
    with open( file_path, 'r' ) as similarity_data:
        return json.loads( similarity_data.read() )


def plot_doc_tokens_mat( file_path_io, file_path_nd, file_path_ht ):
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    sub_titles = [ "Input \& output", "Name \& description", "Help text" ]
    io_tools_tokens = read_files( file_path_io )
    nd_tools_tokens = read_files( file_path_nd )
    ht_tools_tokens = read_files( file_path_ht ) 
    NEW_FONT_SIZE = FONT_SIZE - 2
    io_tokens = list()
    for item in io_tools_tokens:
        a = list()
        for x in item:
            a.append( float( x ) )
        io_tokens.append( a )

    nd_tokens = list()
    for item in nd_tools_tokens:
        a = list()
        for x1 in item:
            a.append( float( x1 ) )
        nd_tokens.append( a )

    ht_tokens = list()
    for item in ht_tools_tokens:
        a = list()
        for x2 in item:
            a.append( float( x2 ) )
        ht_tokens.append( a )

    for row, axis in enumerate( axes ):
        #heatmap1 = axes[ 0 ].imshow( io_tokens, cmap=plt.cm.Reds )
        heatmap2 = axes[ 0 ].imshow( nd_tokens, cmap=plt.cm.Reds )
        heatmap3 = axes[ 1 ].imshow( ht_tokens, cmap=plt.cm.Reds )

        #axes[ 0 ].set_title( sub_titles[ 0 ], fontsize = NEW_FONT_SIZE )
        axes[ 0 ].set_title( sub_titles[ 1 ], fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_title( sub_titles[ 2 ], fontsize = NEW_FONT_SIZE )
        
        for tick in axes[ 0 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axes[ 1 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        '''for tick in axes[ 2 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )'''
        
        for tick in axes[ 0 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axes[ 1 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        '''for tick in axes[ 2 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )'''
        
        #axes[ 0 ].set_xlabel( "Tokens", fontsize = NEW_FONT_SIZE )
        axes[ 0 ].set_xlabel( "N-dimensions", fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_xlabel( "M-dimensions", fontsize = NEW_FONT_SIZE )
 
        axes[ 0 ].set_ylabel( "Documents", fontsize = NEW_FONT_SIZE )
    plt.suptitle( "Documents-tokens multi-dimensional vectors" )
    fig.subplots_adjust( right = 0.75 )
    cbar_ax = fig.add_axes( [ 0.8, 0.15, 0.02, 0.7 ] )
    fig.colorbar( heatmap3, cax=cbar_ax )
    plt.show()


def plot_lr_drop( file_path ):
    data_0_1 = read_files( file_path )
    max_len = 0
    lr_drop = list()
    for item in data_0_1:
        for gd_iter in data_0_1[ item ]:
            lr_steps = len( gd_iter )
            if len( gd_iter ) > max_len:
                max_len = len( gd_iter )
                lr_drop = gd_iter
    plt.plot( lr_drop, marker="*" )
    plt.ylabel( 'Gradient descent learning rate' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Learning rates using backtracking line search' )
    plt.grid( True )
    plt.show()


def plot_correlation( similarity_matrices, title ):
    # plot correlation matrix
    NEW_FONT_SIZE = 22
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    sources = [ "input_output", 'name_desc_edam', 'help_text', "optimal" ]
    titles_fullrank = [ "Input \& output", "Name \& description", "Help text", "Optimal" ]
    row_lst = [ [ 0, 1 ], [ 2, 3 ] ]
    for row, axis in enumerate( axes ):
        mat1 = similarity_matrices[ row_lst[ row ][ 0 ] ]
        mat2 = similarity_matrices[ row_lst[ row ][ 1 ] ]
        heatmap = axis[ 0 ].imshow( mat1, cmap=plt.cm.Reds ) 
        heatmap = axis[ 1 ].imshow( mat2, cmap=plt.cm.Reds ) 
        axis[ 0 ].set_title( titles_fullrank[ row_lst[ row ][ 0 ] ], fontsize = NEW_FONT_SIZE )
        axis[ 1 ].set_title( titles_fullrank[ row_lst[ row ][ 1 ] ], fontsize = NEW_FONT_SIZE )
        
        for tick in axis[ 0 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axis[ 1 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        
        for tick in axis[ 0 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axis[ 1 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        
        if row == 1:
            axis[ 0 ].set_xlabel( "Tools", fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( "Tools", fontsize = NEW_FONT_SIZE )
            
        axis[ 0 ].set_ylabel( "Tools", fontsize = NEW_FONT_SIZE )
        axis[ 1 ].set_ylabel( "Tools", fontsize = NEW_FONT_SIZE )

    fig.subplots_adjust( right = 0.75 )
    cbar_ax = fig.add_axes( [ 0.8, 0.15, 0.02, 0.7 ] )
    fig.colorbar( heatmap, cax=cbar_ax )
    plt.suptitle( title )
    plt.show()


def extract_correlation( file_path, title ):
    # extract correlation matrices from multiple sources
    sim_data = read_files( file_path )
    tools_list = read_files( "data/tools_list.json" )
    mat_size = len( sim_data )
    sim_score_ht = np.zeros( [ mat_size, mat_size ] )
    sim_score_nd = np.zeros( [ mat_size, mat_size ] )
    sim_score_io = np.zeros( [ mat_size, mat_size ] )
    sim_score_op = np.zeros( [ mat_size, mat_size ] )
    similarity_matrices = list()
    for index, tool_name in enumerate( tools_list ):
        sources_sim = sim_data[ tool_name ]
        sim_score_ht[ index ] = sources_sim[ "help_text" ]
        sim_score_nd[ index ] = sources_sim[ "name_desc_edam" ]
        sim_score_io[ index ] = sources_sim[ "input_output" ]
        sim_score_op[ index ] = sources_sim[ "optimal" ]
    similarity_matrices.append( sim_score_io )
    similarity_matrices.append( sim_score_nd )
    similarity_matrices.append( sim_score_ht )
    similarity_matrices.append( sim_score_op )
    plot_correlation( similarity_matrices, title )


def plot_weights_distribution( file_path, title ):
    # plot weights distribution
    NEW_FONT_SIZE = FONT_SIZE
    weights_tools = read_files( file_path )
    weights_io = list()
    weights_nd = list()
    weights_ht = list()
    for item in weights_tools:
        wts = weights_tools[ item ]
        weights_io.append( wts[ "input_output" ] )
        weights_nd.append( wts[ "name_desc_edam" ] )
        weights_ht.append( wts[ "help_text" ] )

    fig, axes = plt.subplots( nrows=1, ncols=3 )
    sources = [ "input_output", 'name_desc_edam', 'help_text', "optimal" ]
    sub_titles = [ "Input \& output", "Name \& description", "Help text", "Optimal" ]
    for row, axis in enumerate( axes ):
        axes[ 0 ].plot( weights_io, color = colors_dict[ "input_output" ] )
        axes[ 1 ].plot( weights_nd, color = colors_dict[ "name_desc_edam" ] )
        axes[ 2 ].plot( weights_ht, color = colors_dict[ "help_text" ] )

        axes[ 0 ].set_title( sub_titles[ 0 ], fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_title( sub_titles[ 1 ], fontsize = NEW_FONT_SIZE )
        axes[ 2 ].set_title( sub_titles[ 2 ], fontsize = NEW_FONT_SIZE )
        
        for tick in axes[ 0 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axes[ 1 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axes[ 2 ].xaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        
        for tick in axes[ 0 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axes[ 1 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        for tick in axes[ 2 ].yaxis.get_major_ticks():
            tick.label.set_fontsize( NEW_FONT_SIZE )
        
        axes[ 0 ].set_xlabel( "Tools", fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_xlabel( "Tools", fontsize = NEW_FONT_SIZE )
        axes[ 2 ].set_xlabel( "Tools", fontsize = NEW_FONT_SIZE )   
        axes[ 0 ].set_ylabel( "Weights", fontsize = NEW_FONT_SIZE )
    plt.suptitle( title )
    plt.show()
        


'''plot_weights_distribution( "data/0.05/optimal_weights.json", "Distribution of weights (5\% of full-rank)" )
'''
#plot_doc_tokens_mat( "data/doc_vecs_io.json", "data/doc_vecs_nd.json", "data/doc_vecs_ht.json" )
extract_correlation( "data/similarity_scores_sources_optimal.json", "Similarity" )
#plot_lr_drop( "data/0.05/learning_rates.json" )
#plot_average_cost_low_rank()
#plot_tokens_size()
#plot_singular_values()
#plot_rank_singular_variation()
#plot_doc_tokens_mat( "data/0.1/similarity_source_orig.json" )
#plot_doc_tokens_mat_low_rank( "data/0.1/similarity_source_low_rank.json" )
#plot_rank_singular_variation()
#plot_frobenius_error()'''
