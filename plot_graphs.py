import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib_venn import venn2
import json


def read_files( file_path ):
    with open( file_path, 'r' ) as similarity_data:
        return json.loads( similarity_data.read() )


def plot_doc_tokens_mat():
    # plot documents tokens matrix  
    plt.rcParams[ "font.serif" ] = "Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman"
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    font = { 'family' : 'sans serif', 'size': 24 }
    plt.rc( 'font', **font )
    sources = [ 'input_output', 'name_desc_edam', 'help_text' ]
    titles_fullrank = [ "Input and output types", "Name and description", "Help text" ]
    for row, axis in enumerate( axes ):
        doc_token_mat = read_files( "data/similarity_source_orig.json" )
        doc_token_mat = doc_token_mat[ sources[ row ] ]
        doc_token_mat_low = read_files( "data/similarity_source_low_rank.json" )
        doc_token_mat_low = doc_token_mat_low[ sources[ row ] ]
        heatmap = axis.imshow( doc_token_mat, cmap=plt.cm.Blues ) 
        axis.set_title( titles_fullrank[ row ], fontsize = 22 )
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize( 22 )
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize( 22 )
        axis.set_xlabel( "Tokens", fontsize = 22 )
        axis.set_ylabel( "Documents", fontsize = 22 )

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([ 0.8, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.suptitle("Documents-tokens matrices for multiple tools attributes")
    plt.show()

def plot_doc_tokens_mat_low_rank():
    # plot documents tokens matrix  
    plt.rcParams[ "font.serif" ] = "Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman"
    fig, axes = plt.subplots( nrows=1, ncols=2 )
    font = { 'family' : 'sans serif', 'size': 24 }
    plt.rc( 'font', **font )
    sources = [ 'input_output', 'name_desc_edam' ]
    titles_fullrank = [ "Input and output types", "Name and description", "Help text" ]
    for row, axis in enumerate( axes ):
        doc_token_mat_low = read_files( "data/similarity_source_low_rank.json" )
        doc_token_mat_low = doc_token_mat_low[ sources[ row ] ]
        heatmap = axis.imshow( doc_token_mat_low, cmap=plt.cm.Blues )
        axis.set_title( titles_fullrank[ row ], fontsize = 22 )
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize( 22 )
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize( 22 )
        axis.set_xlabel( "Tokens", fontsize = 22 )
        axis.set_ylabel( "Documents", fontsize = 22 )

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([ 0.8, 0.15, 0.02, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.suptitle("Low rank representation of documents-tokens matrices")
    plt.show()


plot_doc_tokens_mat( )
#plot_doc_tokens_mat_low_rank()
