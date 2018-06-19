import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json
import random

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
    fig, axes = plt.subplots( nrows=1, ncols=3 )
    sub_titles = [ "Input \& output (a)", "Name \& description (b)", "Help text (c)" ]
    io_tools_tokens = read_files( file_path_io )
    nd_tools_tokens = read_files( file_path_nd )
    ht_tools_tokens = read_files( file_path_ht ) 
    NEW_FONT_SIZE = FONT_SIZE
    io_tokens = list()
    for item in io_tools_tokens:
        a = list()
        for x in item:
            a.append( float( x ) )
        io_tokens.append( a )

    for i in range( len( io_tokens ) ):
        for j in range( len( io_tokens[ i ] ) ):
            if io_tokens[ i ][ j ] <= 0:
                io_tokens[ i ][ j ] = np.nan
            else:
                io_tokens[ i ][ j ] = io_tokens[ i ][ j ]

    nd_tokens = list()
    for item in nd_tools_tokens:
        a = list()
        for x1 in item:
            a.append( float( x1 ) )
        nd_tokens.append( a )

    for i in range( len( nd_tokens ) ):
        for j in range( len( nd_tokens[ i ] ) ):
            if nd_tokens[ i ][ j ] <= 0:
                nd_tokens[ i ][ j ] = np.nan
            else:
                nd_tokens[ i ][ j ] = nd_tokens[ i ][ j ]

    ht_tokens = list()
    for item in ht_tools_tokens:
        a = list()
        for x2 in item:
            a.append( float( x2 ) )
        ht_tokens.append( a )

    for i in range( len( ht_tokens ) ):
        for j in range( len( ht_tokens[ i ] ) ):
            if ht_tokens[ i ][ j ] <= 0:
                ht_tokens[ i ][ j ] = np.nan
            else:
                ht_tokens[ i ][ j ] = ht_tokens[ i ][ j ]
    random.shuffle( io_tokens )
    for col, axis in enumerate( axes ):
        heatmap1 = axes[ 0 ].imshow( io_tokens, cmap=plt.cm.coolwarm )
        heatmap2 = axes[ 1 ].imshow( nd_tokens, cmap=plt.cm.coolwarm )
        heatmap3 = axes[ 2 ].imshow( ht_tokens, cmap=plt.cm.coolwarm )

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

        axes[ 0 ].set_xlabel( "Tokens", fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_xlabel( "100-dimensions", fontsize = NEW_FONT_SIZE )
        axes[ 2 ].set_xlabel( "200-dimensions", fontsize = NEW_FONT_SIZE )
 
        axes[ 0 ].set_ylabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_ylabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
        axes[ 2 ].set_ylabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
    plt.suptitle( "Document-token and paragraph matrices" )
    fig.subplots_adjust( right = 0.75 )
    cbar_ax = fig.add_axes( [ 0.8, 0.15, 0.02, 0.7 ] )
    fig.colorbar( heatmap3, cax=cbar_ax )
    plt.show()


def plot_lr_drop( file_path ):
    tools_lr = read_files( file_path )
    max_len = 0
    lr_drop = list()
    for item in tools_lr:
        lr_drop = tools_lr[ item ]
        break
    plt.plot( lr_drop )
    plt.ylabel( 'Gradient descent learning rate' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Learning rates computed using time-based decay' )
    plt.grid( True )
    plt.show()


def plot_correlation( similarity_matrices, title ):
    # plot correlation matrix
    NEW_FONT_SIZE = 22
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    sources = [ "input_output", 'name_desc_edam', 'help_text', "optimal" ]
    titles_fullrank = [ "Input \& output (a)", "Name \& description (b)", "Help text (c)", "Weighted average (d)" ]
    row_lst = [ [ 0, 1 ], [ 2, 3 ] ]
    for row, axis in enumerate( axes ):
        mat1 = similarity_matrices[ row_lst[ row ][ 0 ] ]
        mat2 = similarity_matrices[ row_lst[ row ][ 1 ] ]
        for i in range( len( mat2 ) ):
            for j in range( len( mat2[ i ] ) ):
                if mat2[ i ][ j ] <= 0:
                    mat2[ i ][ j ] = np.nan
                else:
                    mat2[ i ][ j ] = mat2[ i ][ j ]

        for i in range( len( mat1 ) ):
            for j in range( len( mat1[ i ] ) ):
                if mat1[ i ][ j ] <= 0:
                    mat1[ i ][ j ] = np.nan
                else:
                    mat1[ i ][ j ] = mat1[ i ][ j ]

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
            axis[ 0 ].set_xlabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
            
        axis[ 0 ].set_ylabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
        axis[ 1 ].set_ylabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )

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
    sub_titles = [ "Input \& output (a)", "Name \& description (b)", "Help text (c)", "Weighted average (d)" ]
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
        
        axes[ 0 ].set_xlabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
        axes[ 1 ].set_xlabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )
        axes[ 2 ].set_xlabel( "Tools (documents)", fontsize = NEW_FONT_SIZE )   
        axes[ 0 ].set_ylabel( "Weights", fontsize = NEW_FONT_SIZE )
    plt.suptitle( title )
    plt.show()


def plot_gradient_drop( actual_gd_file_path ):
    actual_gd = read_files( actual_gd_file_path )
    tools_len = len( actual_gd )
    iterations = 100

    actual_io = np.zeros( [ tools_len, iterations ] )
    actual_nd = np.zeros( [ tools_len, iterations ] )
    actual_ht = np.zeros( [ tools_len, iterations ] )
    cumulative_gradient = np.zeros( [ tools_len, iterations ] )

    for index_x, item_y in enumerate( actual_gd ):
        for index_y, y in enumerate( actual_gd[ item_y ] ):
            actual_io[ index_x ][ index_y ] = y[ "input_output" ]
            actual_nd[ index_x ][ index_y ] = y[ "name_desc_edam" ]
            actual_ht[ index_x ][ index_y ] = y[ "help_text" ]

    for tool_idx in range( tools_len ):
        for iteration in range( iterations ):
            cumulative_gradient[ tool_idx ][ iteration ] = np.sqrt( actual_io[ tool_idx ][ iteration ] ** 2 + actual_nd[ tool_idx ][ iteration ] ** 2 + actual_ht[ tool_idx ][ iteration ] ** 2 )

    mean_cumulative_gradient = np.mean( cumulative_gradient, axis = 0 )
    plt.plot( mean_cumulative_gradient, color='C0' )
    plt.ylabel( 'Cumulative gradient' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Cumulative gradient over iterations for all the tools attributes' )
    plt.grid( True )
    plt.show()
    
def compute_cost( similarity_data, iterations ):
    cost_iterations = np.zeros( [ len( similarity_data ) - 1, iterations ] )
    for index, tool in enumerate( similarity_data ):
        if "cost_iterations" in tool:
            cost_iterations[ index ][ : ] = tool[ "cost_iterations" ][ :iterations ]
    # compute mean cost occurred for each tool across iterations
    return np.mean( cost_iterations, axis=0 )
    

def plot_average_cost():
    max_iter = 100
    sim_mat = read_files( "data/similarity_matrix.json" )
    cost = compute_cost( sim_mat, max_iter )
    plt.plot( cost )
    plt.ylabel( 'Mean squared error' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Mean squared error over iterations for paragraph vectors approach' )
    plt.grid( True )
    plt.show()


def plot_average_optimal_scores():
    similarity_data = read_files( "data/similarity_matrix.json" )
    ave_scores = np.zeros( [ len( similarity_data ) - 1, len( similarity_data ) - 1 ] )
    opt_scores = np.zeros( [ len( similarity_data ) - 1, len( similarity_data ) - 1 ] )
    for index, tool in enumerate( similarity_data ):
        if "average_similar_scores" in tool:
            ave_scores[ index ] = tool[ "average_similar_scores" ]
        if "optimal_similar_scores" in tool:
            opt_scores[ index ] = tool[ "optimal_similar_scores" ]
    plt.plot( np.mean( opt_scores, axis = 0 ) )
    plt.plot( np.mean( ave_scores, axis = 0 ) )
    plt.ylabel( 'Weighted similarity scores' )
    plt.xlabel( 'Tools' )
    plt.title( 'Weighted similarity using uniform and optimal weights using paragraph vectors approach', fontsize=26 )
    plt.legend( [ "Weights learned using optimisation", "Uniform weights" ], loc=4 )
    plt.grid( True )
    plt.show()
    

#plot_gradient_drop( "data/actual_gd_tools.json" ) 
#plot_weights_distribution( "data/optimal_weights.json", "Distribution of weights using paragraph vectors approach" )
#plot_doc_tokens_mat( "data/doc_vecs_io.json", "data/doc_vecs_nd.json", "data/doc_vecs_ht.json" )
#extract_correlation( "data/similarity_scores_sources_optimal.json", "Similarity matrices using paragraph vectors approach" )
plot_average_cost()
plot_average_optimal_scores()
'''plot_average_cost()
plot_gradient_drop( "data/actual_gd_tools.json" ) 
       
plot_weights_distribution( "data/optimal_weights.json", "Distribution of weights using paragraph vectors approach" )
plot_doc_tokens_mat( "data/doc_vecs_io.json", "data/doc_vecs_nd.json", "data/doc_vecs_ht.json" )
extract_correlation( "data/similarity_scores_sources_optimal.json", "Similarity matrices using paragraph vectors approach" )
plot_lr_drop( "data/learning_rates.json" )'''
