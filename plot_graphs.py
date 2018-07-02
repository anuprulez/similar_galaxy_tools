import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json
import string
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


def plot_doc_tokens_mat( file_path ):
    # plot documents tokens matrix
    new_fs = FONT_SIZE
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    sources = [ 'name_desc_edam', 'help_text' ]
    titles_fullrank = [ "Name \& description (a)", "Help text (b)" ]
    for row, axis in enumerate( axes ):
        doc_token_mat = read_files( file_path )
        doc_token_mat = doc_token_mat[ sources[ row ] ]
        random.shuffle( doc_token_mat )
        for i in range( len( doc_token_mat ) ):
            for j in range( len( doc_token_mat[ i ] ) ):
                if doc_token_mat[ i ][ j ] <= 0:
                    doc_token_mat[ i ][ j ] = np.nan
                else:
                    doc_token_mat[ i ][ j ] = doc_token_mat[ i ][ j ]
        heatmap = axis.imshow( doc_token_mat, cmap=plt.cm.coolwarm ) 
        axis.set_title( titles_fullrank[ row ], fontsize = new_fs )
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        axis.set_xlabel( "Tokens", fontsize = new_fs )
        if row == 0:
            axis.set_ylabel( "Tools (documents)", fontsize = new_fs )

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.suptitle("Document-token matrices")
    plt.show()


def plot_doc_tokens_mat_low_rank( file_path ):
    # plot documents tokens matrix  
    new_fs = FONT_SIZE - 2
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    sources = [ 'name_desc_edam', 'help_text' ]
    titles_fullrank = [ "Name \& description (a)", "Help text (b)" ]
    for row, axis in enumerate( axes ):
        doc_token_mat_low = read_files( file_path )
        doc_token_mat_low = doc_token_mat_low[ sources[ row ] ]
        random.shuffle( doc_token_mat_low )
        for i in range( len( doc_token_mat_low ) ):
            for j in range( len( doc_token_mat_low[ i ] ) ):
                if doc_token_mat_low[ i ][ j ] <= 0:
                    doc_token_mat_low[ i ][ j ] = np.nan
                else:
                    doc_token_mat_low[ i ][ j ] = doc_token_mat_low[ i ][ j ]
        heatmap = axis.imshow( doc_token_mat_low, cmap=plt.cm.coolwarm )
        axis.set_title( titles_fullrank[ row ], fontsize = new_fs )
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        axis.set_xlabel( "Tokens", fontsize = new_fs )
        if row == 0:
            axis.set_ylabel( "Tools (documents)", fontsize = new_fs )

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.suptitle("Low-rank representation of document-token matrices")
    plt.show()


def plot_tokens_size():
    # plot nubmer of tokens for each tool for all sources
    io_tokens = list()
    nd_tokens = list()
    ht_tokens = list()
    NEW_FONT_SIZE = FONT_SIZE
    #fig, axes = plt.subplots( nrows=1, ncols=3 )
    sub_titles = [ "Input \& output", "Name \& description", "Help text" ]
    io_tools_tokens = read_files( "data/tokens_input_output.txt" )
    nd_tools_tokens = read_files( "data/tokens_name_desc_edam.txt" )
    ht_tools_tokens = read_files( "data/tokens_help_text.txt" )
    for index, item in enumerate( io_tools_tokens ):
        io_tokens.append( len( io_tools_tokens[ item ] ) )
        nd_tokens.append( len( nd_tools_tokens[ item ] ) )
        ht_tokens.append( len( ht_tools_tokens[ item ] ) )

    '''for row, axis in enumerate( axes ):
        axes[ 0 ].plot( io_tokens, color = colors_dict[ "input_output" ] )
        axes[ 1 ].plot( nd_tokens, color = colors_dict[ "name_desc_edam" ] )
        axes[ 2 ].plot( ht_tokens, color = colors_dict[ "help_text" ] )

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
        axes[ 0 ].set_ylabel( "Number of tokens", fontsize = NEW_FONT_SIZE )'''
    plt.plot( io_tokens )
    plt.plot( nd_tokens )
    plt.plot( ht_tokens )
    plt.legend( [ "Input \& output", "Name \& description", "Help text" ], loc=1 ) 
    plt.ylabel( "Number of tokens" )
    plt.xlabel( "Tools" )
    plt.grid( True )
    plt.title( "Distribution of tokens" )
    plt.show()


def plot_rank_eigen_variation_fraction():
    # plot how the sum of eigen values vary with the rank of the documents-tokens matrices
    help_rank = read_files( "data/help_text_vary_eigen_rank.json" )
    name_rank = read_files( "data/name_desc_edam_vary_eigen_rank.json" )
    io_rank = read_files( "data/input_output_vary_eigen_rank.json" )
    plt.plot( io_rank[ 0 ], io_rank[ 1 ], color = colors_dict[ "input_output" ] )
    plt.plot( name_rank[ 0 ], name_rank[ 1 ], color = colors_dict[ "name_desc_edam" ] )
    plt.plot( help_rank[ 0 ], help_rank[ 1 ], color = colors_dict[ "help_text" ] )
    plt.ylabel( 'Fraction of sum of singular values' )
    plt.xlabel( 'Fraction of ranks' )
    plt.title( 'Variation of singular values with ranks', fontsize=26 )
    plt.legend( [ "Input \& output", "Name \& description", "Help text" ], loc=4, fontsize=FONT_SIZE - 2 )
    plt.grid( True )
    plt.show()
  
    
def plot_rank_eigen_variation():
    # plot how the sum of eigen values vary with the rank of the documents-tokens matrices
    NEW_FONT_SIZE_TIKCS = FONT_SIZE - 10
    NEW_FONT_SIZE = FONT_SIZE - 4
    help_rank = read_files( "data/help_text_vary_eigen_rank.json" )
    io_rank = read_files( "data/input_output_vary_eigen_rank.json" )
    name_rank = read_files( "data/name_desc_edam_vary_eigen_rank.json" )
    fig, axes = plt.subplots( nrows=1, ncols=3 )
    axes[0].plot( io_rank[ 2 ], io_rank[ 3 ], color=colors_dict[ "input_output" ] )
    axes[0].set_title( "Input \& output (a)", fontsize = NEW_FONT_SIZE )
    axes[0].set_xlabel( "Matrix rank", fontsize = NEW_FONT_SIZE )
    axes[0].set_ylabel( "Sum of singular values", fontsize = NEW_FONT_SIZE )
    axes[0].grid(True)
    
    axes[1].plot( name_rank[ 2 ], name_rank[ 3 ], color=colors_dict[ "name_desc_edam" ] )
    axes[1].set_title( "Name \& description (b)", fontsize = NEW_FONT_SIZE )
    axes[1].set_xlabel( "Matrix rank", fontsize = NEW_FONT_SIZE )
    axes[1].grid(True)
    
    axes[2].plot( help_rank[ 2 ], help_rank[ 3 ], color=colors_dict[ "help_text" ] )
    axes[2].set_title( "Help text (c)", fontsize = NEW_FONT_SIZE )
    axes[2].set_xlabel( "Matrix rank", fontsize = NEW_FONT_SIZE )
    axes[2].grid(True)
    
    for tick in axes[ 0 ].xaxis.get_major_ticks():
        tick.label.set_fontsize( NEW_FONT_SIZE_TIKCS )
    for tick in axes[ 1 ].xaxis.get_major_ticks():
        tick.label.set_fontsize( NEW_FONT_SIZE_TIKCS )
    for tick in axes[ 2 ].xaxis.get_major_ticks():
        tick.label.set_fontsize( NEW_FONT_SIZE_TIKCS )
        
    for tick in axes[ 0 ].yaxis.get_major_ticks():
        tick.label.set_fontsize( NEW_FONT_SIZE_TIKCS )
    for tick in axes[ 1 ].yaxis.get_major_ticks():
        tick.label.set_fontsize( NEW_FONT_SIZE_TIKCS )
    for tick in axes[ 2 ].yaxis.get_major_ticks():
        tick.label.set_fontsize( NEW_FONT_SIZE_TIKCS )
  
    plt.suptitle("Variation of sum of singular values with ranks of the matrices")
    plt.show()

 
def plot_frobenius_error():
    error_sources = read_files( "data/frobenius_error.json" )
    for item in error_sources:
        error_src = error_sources[ item ]
        ranks = list()
        error = list()
        full_rank = len( error_src )
        for item in error_src:
            ranks.append( item[ 0 ] / float( full_rank ) )
            error.append( item[ 1 ] )
        plot( ranks, error )
    plt.ylabel( 'Frobenius norm error' )
    plt.xlabel( 'Percentage rank' )
    plt.title( 'Frobenius error variation with the percentage rank' )
    plt.legend( [ "Help text", "Name and description", "Input and output" ] )
    plt.grid( True )
    plt.show()


def plot_rank_singular_variation():
    error_sources = read_files( "data/frobenius_error.json" )
    for item in error_sources:
        error_src = error_sources[ item ]
        ranks = list()
        sum_singular_perc = list()
        full_rank = len( error_src )
        for rnk in error_src:
            ranks.append( rnk[ 0 ] / float( full_rank ) )
            sum_singular_perc.append( rnk[ 1 ] )
        plt.plot( ranks, sum_singular_perc, color=colors_dict[ item ] )
    plt.ylabel( 'Fraction of the sum of singular values' )
    plt.xlabel( 'Fraction of the ranks of documents-tokens matrices' )
    plt.title( 'Variation of ranks with singular values ' )
    plt.legend( [ "Help text", "Name and description", "Input and output" ] )
    plt.grid( True )
    plt.show()


def plot_singular_values():
    singular_values = read_files( "data/0.3/singular_values_input_output.json" )
    fig, axes = plt.subplots( nrows=1, ncols=3 )
    axes[0].plot( singular_values[ "input_output" ], color=colors_dict[ "input_output" ] )
    axes[0].set_title( "Input \& output (a)" )
    axes[0].set_xlabel( "Count of singular values" )
    axes[0].set_ylabel( "Magnitude of singular values" )
    axes[0].grid( True )
    
    axes[1].plot( singular_values[ "name_desc_edam" ], color=colors_dict[ "name_desc_edam" ] )
    axes[1].set_title( "Name \& description (b)" )
    axes[1].set_xlabel( "Count of singular values" )
    axes[1].grid( True )
    
    axes[2].plot( singular_values[ "help_text" ], color=colors_dict[ "help_text" ] )
    axes[2].set_title( "Help text (c)" )
    axes[2].set_xlabel( "Count of singular values" )
    axes[2].grid( True )
        
    plt.suptitle("Singular values")
    plt.show()
      

def compute_cost( similarity_data, iterations ):
    cost_iterations = np.zeros( [ len( similarity_data ) - 1, iterations ] )
    for index, tool in enumerate( similarity_data ):
        if "cost_iterations" in tool:
            cost_iterations[ index ][ : ] = tool[ "cost_iterations" ][ :iterations ]
    # compute mean cost occurred for each tool across iterations
    return np.mean( cost_iterations, axis=0 )
    

def plot_average_cost_low_rank():
    max_iter = 100
    data_1_0 = read_files( "data/1.0/similarity_matrix.json" )
    data_0_7 = read_files( "data/0.7/similarity_matrix.json" )
    #data_0_5 = read_files( "data/0.5/similarity_matrix.json" )
    data_0_3 = read_files( "data/0.3/similarity_matrix.json" )
    #data_0_1 = read_files( "data/0.1/similarity_matrix.json" )
    data_0_0_5 = read_files( "data/0.05/similarity_matrix.json" )

    cost_1_0 = compute_cost( data_1_0, max_iter )
    cost_0_7 = compute_cost( data_0_7, max_iter )
    #cost_0_5 = compute_cost( data_0_5, max_iter )
    cost_0_3 = compute_cost( data_0_3, max_iter )
    #cost_0_1 = compute_cost( data_0_1, max_iter )
    cost_0_0_5 = compute_cost( data_0_0_5, max_iter )
    
    plt.plot( cost_1_0 )
    plt.plot( cost_0_7 )
    #plt.plot( cost_0_5, marker='s' )
    plt.plot( cost_0_3 )
    #plt.plot( cost_0_1,marker='x' )
    plt.plot( cost_0_0_5 )
    plt.ylabel( 'Mean squared error' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Mean squared error for various low-rank document-token matrix estimations' )
    plt.legend( [ "Full-rank", "70\% of full-rank", "30\% of full-rank", "5\% of full-rank" ], loc=1 )
    plt.grid( True )
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
    sub_titles = [ "Input \& output (a)", "Name \& description (b)", "Help text (c)" ]
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

def verify_gradient( approx_gd_file_path, actual_gd_file_path ):
    approx_gd = read_files( approx_gd_file_path )
    tools_len = len( approx_gd )
    iterations = 100
    approx_io = np.zeros( [ tools_len, iterations ] )
    approx_nd = np.zeros( [ tools_len, iterations ] )
    approx_ht = np.zeros( [ tools_len, iterations ] )

    actual_gd = read_files( actual_gd_file_path )
    actual_io = np.zeros( [ tools_len, iterations ] )
    actual_nd = np.zeros( [ tools_len, iterations ] )
    actual_ht = np.zeros( [ tools_len, iterations ] )

    for index_x, item_x in enumerate( approx_gd ):
        for index_y, y in enumerate( approx_gd[ item_x ] ):
            approx_io[ index_x ][ index_y ] = y[ "input_output" ]
            approx_nd[ index_x ][ index_y ] = y[ "name_desc_edam" ]
            approx_ht[ index_x ][ index_y ] = y[ "help_text" ]

    for index_x, item_y in enumerate( actual_gd ):
        for index_y, y in enumerate( actual_gd[ item_y ] ):
            actual_io[ index_x ][ index_y ] = y[ "input_output" ]
            actual_nd[ index_x ][ index_y ] = y[ "name_desc_edam" ]
            actual_ht[ index_x ][ index_y ] = y[ "help_text" ]

    error_io = np.mean( approx_io - actual_io, axis = 0)
    error_nd = np.mean( approx_nd - actual_nd, axis = 0 )
    error_ht = np.mean( approx_ht - actual_ht, axis = 0 )
    plt.plot( error_io, color='C0' )
    plt.plot( error_nd, color='C1' )
    plt.plot( error_ht, color='C2' )
    plt.ylabel( 'Difference of gradients' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Difference of actual and approximate gradients' )
    plt.legend( [ "Input \& output", "Name \& description", "Help text" ], loc=1 )
    plt.grid( True )
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
    
def plot_average_optimal_scores( file_path, title ):
    similarity_data = read_files( file_path )
    ave_scores = np.zeros( [ len( similarity_data ) - 1, len( similarity_data ) - 1 ] )
    opt_scores = np.zeros( [ len( similarity_data ) - 1, len( similarity_data ) - 1 ] )
    for index, tool in enumerate( similarity_data ):
        if "average_similar_scores" in tool:
            ave_scores[ index ] = tool[ "average_similar_scores" ]
        if "optimal_similar_scores" in tool:
            opt_scores[ index ] = tool[ "optimal_similar_scores" ]
    plt.plot( np.mean( opt_scores, axis = 0 ) )
    plt.plot( np.mean( ave_scores, axis = 0 ) )
    plt.ylabel( 'Average of weighted similarity' )
    plt.xlabel( 'Tools' )
    plt.title( title, fontsize = 26 )
    plt.legend( [ "Weights learned using optimisation", "Uniform weights" ], loc=4 )
    plt.grid( True )
    plt.show()


def plot_lr_drop( file_path ):
    tools_lr = read_files( file_path )
    max_len = 0
    lr_drop = list()
    for item in tools_lr:
        lr_drop = tools_lr[ item ]
        break
    plt.plot( lr_drop, color='r' )
    plt.ylabel( 'Learning rate' )
    plt.xlabel( 'Iterations' )
    plt.title( 'Decay of learning rate' )
    plt.grid( True )
    plt.show()

plot_rank_eigen_variation_fraction()
#plot_singular_values()
#plot_rank_eigen_variation_fraction()
#plot_lr_drop( "data/learning_rates.json" )
#extract_correlation( "data/1.0/similarity_scores_sources_optimal.json", "Similarity matrices" )
#plot_weights_distribution( "data/1.0/optimal_weights.json", "Distribution of weights" )
#plot_average_optimal_scores( "data/1.0/similarity_matrix.json", 'Average of weighted similarity for tools' )
#extract_correlation( "data/0.05/similarity_scores_sources_optimal.json", "Similarity matrices" )
#plot_weights_distribution( "data/0.05/optimal_weights.json", "Distribution of weights" )
#plot_average_optimal_scores( "data/0.05/similarity_matrix.json", 'Average of weighted similarity for tools' )


'''extract_correlation( "data/0.05/similarity_scores_sources_optimal.json", "Similarity matrices with 5\% of full-rank of document-token matrices" )
extract_correlation( "data/0.3/similarity_scores_sources_optimal.json", "Similarity matrices with 30\% of full-rank of document-token matrices" )
extract_correlation( "data/0.7/similarity_scores_sources_optimal.json", "Similarity matrices with 70\% of full-rank of document-token matrices" )
extract_correlation( "data/1.0/similarity_scores_sources_optimal.json", "Similarity matrices with 100\% of full-rank of document-token matrices" )'''

#plot_rank_singular_variation()
#plot_rank_eigen_variation_fraction()
#plot_tokens_size()
#plot_singular_values()
'''plot_doc_tokens_mat( "data/0.05/similarity_source_orig.json" )
plot_doc_tokens_mat_low_rank( "data/0.05/similarity_source_low_rank.json" )
plot_average_optimal_scores( "data/0.05/similarity_matrix.json", 'Weighted similarity scores using uniform and optimal weights (5\% of full-rank document-token matrices)' )
plot_average_optimal_scores( "data/1.0/similarity_matrix.json", 'Weighted similarity scores using uniform and optimal weights (full-rank document-token matrices)' )
plot_rank_eigen_variation_fraction()
plot_rank_eigen_variation()'''
'''plot_singular_values()
plot_average_cost_low_rank()
#plot_gradient_drop( "data/0.05/learning_rates.json" ) 
verify_gradient( "data/0.05/actual_gd_tools.json", "data/0.05/approx_gd_tools.json" )
plot_weights_distribution( "data/0.05/optimal_weights.json", "Distribution of weights (5\% of full-rank document-token matrices)" )
#plot_weights_distribution( "data/0.1/optimal_weights.json", "Distribution of weights (10\% of full-rank document-token matrices)" )
plot_weights_distribution( "data/0.3/optimal_weights.json", "Distribution of weights (30\% of full-rank document-token matrices)" )
#plot_weights_distribution( "data/0.5/optimal_weights.json", "Distribution of weights (50\% of full-rank document-token matrices)" )
plot_weights_distribution( "data/0.7/optimal_weights.json", "Distribution of weights (70\% of full-rank document-token matrices)" )
plot_weights_distribution( "data/1.0/optimal_weights.json", "Distribution of weights (100\% of full-rank document-token matrices)" )

extract_correlation( "data/0.05/similarity_scores_sources_optimal.json", "Similarity matrices computed with 5\% of full-rank document-token matrices" )
#extract_correlation( "data/0.1/similarity_scores_sources_optimal.json", "Similarity matrices computed with 10\% of full-rank document-token matrices" )
extract_correlation( "data/0.3/similarity_scores_sources_optimal.json", "Similarity matrices computed with 30\% of full-rank document-token matrices" )
#extract_correlation( "data/0.5/similarity_scores_sources_optimal.json", "Similarity matrices computed with 50\% of full-rank document-token matrices" )
extract_correlation( "data/0.7/similarity_scores_sources_optimal.json", "Similarity matrices computed with 70\% of full-rank document-token matrices" )
extract_correlation( "data/1.0/similarity_scores_sources_optimal.json", "Similarity matrices computed with 100\% of full-rank document-token matrices" )

#plot_gradient_drop( "data/0.05/learning_rates.json" )
plot_average_cost_low_rank()
#plot_tokens_size()
#plot_singular_values()
#plot_rank_eigen_variation()
#plot_rank_singular_variation()
#plot_doc_tokens_mat( "data/0.05/similarity_source_orig.json" )
#plot_doc_tokens_mat_low_rank( "data/0.05/similarity_source_low_rank.json" )
#plot_rank_singular_variation()
#plot_frobenius_error()'''
