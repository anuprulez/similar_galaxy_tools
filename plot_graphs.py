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
        doc_token_mat = read_files( "data/similarity_source_orig.json" )
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
        doc_token_mat_low = read_files( "data/similarity_source_low_rank.json" )
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
        

#plot_tokens_size()
#plot_singular_values()
#plot_rank_singular_variation()
plot_doc_tokens_mat()
plot_doc_tokens_mat_low_rank()
#plot_rank_singular_variation()
#plot_frobenius_error()'''
