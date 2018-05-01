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


def plot_doc_tokens_mat():
    # plot documents tokens matrix
    new_fs = FONT_SIZE - 2
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    sources = [ 'name_desc_edam', 'help_text' ]
    titles_fullrank = [ "Name and description", "Help text" ]
    for row, axis in enumerate( axes ):
        doc_token_mat = read_files( "data/0.1/similarity_source_orig.json" )
        doc_token_mat = doc_token_mat[ sources[ row ] ]
        heatmap = axis.imshow( doc_token_mat, cmap=plt.cm.Blues ) 
        axis.set_title( titles_fullrank[ row ], fontsize = new_fs )
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        axis.set_xlabel( "Tokens", fontsize = new_fs )
        if row == 0:
            axis.set_ylabel( "Documents", fontsize = new_fs )

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.suptitle("Documents-tokens matrices for tools attributes")
    plt.show()


def plot_doc_tokens_mat_low_rank():
    # plot documents tokens matrix  
    new_fs = FONT_SIZE - 2
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    sources = [ 'name_desc_edam', 'help_text' ]
    titles_fullrank = [ "Name and description", "Help text" ]
    for row, axis in enumerate( axes ):
        doc_token_mat_low = read_files( "data/0.1/similarity_source_low_rank.json" )
        doc_token_mat_low = doc_token_mat_low[ sources[ row ] ]
        heatmap = axis.imshow( doc_token_mat_low, cmap=plt.cm.Blues )
        axis.set_title( titles_fullrank[ row ], fontsize = new_fs )
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize( new_fs )
        axis.set_xlabel( "Tokens", fontsize = new_fs )
        if row == 0:
            axis.set_ylabel( "Documents", fontsize = new_fs )

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([0.8, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.suptitle("Low rank representation of documents-tokens matrices")
    plt.show()


def plot_tokens_size():
    # plot nubmer of tokens for each tool for all sources
    io_tokens = list()
    nd_tokens = list()
    ht_tokens = list()
    
    io_tools_tokens = read_files( "data/tokens_input_output.txt" )
    nd_tools_tokens = read_files( "data/tokens_name_desc_edam.txt" )
    ht_tools_tokens = read_files( "data/tokens_help_text.txt" )
    for index, item in enumerate( io_tools_tokens ):
        io_tokens.append( len( io_tools_tokens[ item ] ) )
        nd_tokens.append( len( nd_tools_tokens[ item ] ) )
        ht_tokens.append( len( ht_tools_tokens[ item ] ) )
    plt.plot( io_tokens )
    plt.plot( nd_tokens )
    plt.plot( ht_tokens )
    plt.ylabel( 'Number of tokens' )
    plt.xlabel( 'Number of tools' )
    plt.title( 'Distribution of tokens for the attributes of tools' )
    plt.legend( [ "Input and output", "Name and description", "Help text" ] )
    plt.grid( True )
    plt.show()


def plot_rank_eigen_variation():
    # plot how the sum of eigen values vary with the rank of the documents-tokens matrices
    help_rank = read_files( "data/help_text_vary_eigen_rank.json" )
    name_rank = read_files( "data/name_desc_edam_vary_eigen_rank.json" )
    io_rank = read_files( "data/input_output_vary_eigen_rank.json" )
    plt.plot( io_rank[ 0 ], io_rank[ 1 ] )
    plt.plot( name_rank[ 0 ], name_rank[ 1 ] )
    plt.plot( help_rank[ 0 ], help_rank[ 1 ] )
    plt.ylabel( 'Percentage of the sum of eigen values' )
    plt.xlabel( 'Percentage of rank of the matrix' )
    plt.title( 'Documents-tokens matrix rank variation with the sum of eigen values' )
    plt.legend( [ "Input and output", "Name and description", "Help text" ] )
    plt.grid( True )
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
            sum_singular_perc.append( rnk[ 2 ] )
        plt.plot( ranks, sum_singular_perc, color=colors_dict[ item ] )
    plt.ylabel( 'Fraction of the sum of singular values' )
    plt.xlabel( 'Ranks of the documents-tokens matrices in percentage' )
    plt.title( 'Variation of sum of singular values with ranks' )
    plt.legend( [ "Help text", "Name and description", "Input and output" ] )
    plt.grid( True )
    plt.show()


def plot_singular_values():
    plt.rcParams[ "font.size" ] = FONT_SIZE - 2
    singular_values = read_files( "data/0.3/singular_values_input_output.json" )
    fig, axes = plt.subplots( nrows=1, ncols=3 )
    axes[0].plot( singular_values[ "input_output" ], color=colors_dict[ "input_output" ] )
    axes[0].set_title( "Input & output" )
    axes[0].set_xlabel( "Singular values count" )
    axes[0].set_ylabel( "Singular values" )
    axes[0].grid(True)
    
    axes[1].plot( singular_values[ "name_desc_edam" ], color=colors_dict[ "name_desc_edam" ] )
    axes[1].set_title( "Name & desc." )
    axes[1].set_xlabel( "Singular values count" )
    axes[1].grid(True)
    
    axes[2].plot( singular_values[ "help_text" ], color=colors_dict[ "help_text" ] )
    axes[2].set_title( "Help text" )
    axes[2].set_xlabel( "Singular values count" )
    axes[2].grid(True)
    plt.suptitle("Singular values for tools documents-tokens matrices")
    plt.show()
      

def compute_cost( similarity_data, iterations ):
    cost_iterations = np.zeros( [ len( similarity_data ) - 1, iterations ] )
    for index, tool in enumerate( similarity_data ):
        if "cost_iterations" in tool:
            cost_iterations[ index ][ : ] = tool[ "cost_iterations" ][ :iterations ]
    # compute mean cost occurred for each tool across iterations
    return np.mean( cost_iterations, axis=0 )
    

def plot_average_cost_low_rank():
    max_iter = 10
    data_1_0 = read_files( "data/1.0/similarity_matrix.json" )
    data_0_7 = read_files( "data/0.7/similarity_matrix.json" )
    data_0_5 = read_files( "data/0.5/similarity_matrix.json" )
    data_0_3 = read_files( "data/0.3/similarity_matrix.json" )
    data_0_1 = read_files( "data/0.1/similarity_matrix.json" )
    data_0_0_5 = read_files( "data/0.05/similarity_matrix.json" )

    cost_1_0 = compute_cost( data_1_0, max_iter )
    cost_0_7 = compute_cost( data_0_7, max_iter )
    cost_0_5 = compute_cost( data_0_5, max_iter )
    cost_0_3 = compute_cost( data_0_3, max_iter )
    cost_0_1 = compute_cost( data_0_1, max_iter )
    cost_0_0_5 = compute_cost( data_0_0_5, max_iter )
    
    plt.plot( cost_1_0, marker='o' )
    plt.plot( cost_0_7, marker='8' )
    plt.plot( cost_0_5, marker='s' )
    plt.plot( cost_0_3, marker='+')
    plt.plot( cost_0_1,marker='x' )
    plt.plot( cost_0_0_5,marker='<' )
    plt.ylabel( 'Mean squared error' )
    plt.xlabel( 'Gradient descent iterations' )
    plt.title( 'Mean squared error for multiple low-rank matrix estimations' )
    plt.legend( [ "Full-rank", "70\% of full-rank", "50\% of full-rank", "30\% of full-rank", "10\% of full-rank", "5\% of full-rank" ], loc=1 )
    plt.grid( True )
    plt.show()
    

def plot_lr_drop():
    data_0_1 = read_files( "data/0.1/learning_rates.json" )
    max_len = 0
    lr_drop = list()
    for item in data_0_1:
        for gd_iter in data_0_1[ item ]:
            lr_steps = len( gd_iter )
            if len( gd_iter ) > max_len:
                max_len = len( gd_iter )
                lr_drop = gd_iter
    plt.plot( lr_drop )
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
        heatmap = axis[ 0 ].imshow( mat1, cmap=plt.cm.Blues ) 
        heatmap = axis[ 1 ].imshow( mat2, cmap=plt.cm.Blues ) 
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
    with open( file_path, 'r' ) as similarity_data:
        sim_data = json.loads( similarity_data.read() )
    mat_size = len( sim_data )
    sim_score_ht = np.zeros( [ mat_size, mat_size ] )
    sim_score_nd = np.zeros( [ mat_size, mat_size ] )
    sim_score_io = np.zeros( [ mat_size, mat_size ] )
    sim_score_op = np.zeros( [ mat_size, mat_size ] )
    tools = list()
    similarity_matrices = list()
    for index, item in enumerate( sim_data ):
        tools.append( item )
        sources_sim = sim_data[ item ]
        sim_score_ht[ index ][ : ] = sources_sim[ "help_text" ]
        sim_score_nd[ index ][ : ] = sources_sim[ "name_desc_edam" ]
        sim_score_io[ index ][ : ] = sources_sim[ "input_output" ]
        sim_score_op[ index ][ : ] = sources_sim[ "optimal" ]
    
    similarity_matrices.append( sim_score_io )
    similarity_matrices.append( sim_score_nd )
    similarity_matrices.append( sim_score_ht )
    similarity_matrices.append( sim_score_op )
    plot_correlation( similarity_matrices, title )


extract_correlation( "data/0.05/similarity_scores_sources_optimal.json", "Similarity matrices computed with 5\% of full-rank" )
extract_correlation( "data/0.1/similarity_scores_sources_optimal.json", "Similarity matrices computed with 10\% of full-rank" )
extract_correlation( "data/0.3/similarity_scores_sources_optimal.json", "Similarity matrices computed with 30\% of full-rank" )
extract_correlation( "data/0.5/similarity_scores_sources_optimal.json", "Similarity matrices computed with 50\% of full-rank" )
extract_correlation( "data/0.7/similarity_scores_sources_optimal.json", "Similarity matrices computed with 70\% of full-rank" )
extract_correlation( "data/1.0/similarity_scores_sources_optimal.json", "Similarity matrices computed with 100\% of full-rank" )
#plot_lr_drop()
#plot_average_cost_low_rank()
#plot_tokens_size()
#plot_singular_values()
#plot_rank_singular_variation()
#plot_doc_tokens_mat()
#plot_doc_tokens_mat_low_rank()
#plot_rank_singular_variation()
#plot_frobenius_error()'''
