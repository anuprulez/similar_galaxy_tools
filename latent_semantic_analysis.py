import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import svd
import json

import utils


class LatentSemanticIndexing:

    @classmethod
    def __init__( self, rank_reduce ):
        self.rank_reduction = rank_reduce

    @classmethod
    def _compute_low_rank_matrix( self, u, s, v, rank ):
        """
        Compute low rank approximation of a full rank matrix
        """
        u_approx = u[ :, :rank ]
        s_approx = s[ :rank ]
        sum_taken_percent = np.sum( s_approx ) / float( np.sum( s ) )
        s_approx = np.diag( np.array( s_approx ) )
        v_approx = v[ :rank, : ]
        return [ u_approx.dot( s_approx ).dot( v_approx ), sum_taken_percent ]

    @classmethod
    def _find_optimal_low_rank_matrix( self, orig_similarity_matrix, orig_rank, u, s, v, source ):
        """
        Find the rank which captures most of the information from the original full rank matrix
        """
        '''rank_list = list()
        sum_singular_values = list()
        vary_rank_eigen = list()
        for rank in range( 1, orig_rank ):
            compute_result = self._compute_low_rank_matrix( u, s, v, rank )
            rank_list.append( rank / float( orig_rank ) )
            sum_singular_values.append( compute_result[ 1 ] )
        vary_rank_eigen.append( rank_list )
        vary_rank_eigen.append( sum_singular_values )
        with open( "data/" + source + "_vary_eigen_rank.json", "w" ) as vary_rank:
            vary_rank.write( json.dumps( vary_rank_eigen ) )
        #utils._plot_singular_values_rank( rank_list, sum_singular_values, source )'''
        return self._compute_low_rank_matrix( u, s, v, int( self.rank_reduction * orig_rank ) )

    @classmethod
    def factor_matrices( self, document_tokens_matrix_sources ):
        """
        Latent semantic indexing
        """
        print "Computing lower rank representations of documents tokens matrices..."
        approx_similarity_matrices = dict()
        # it is number determined heuristically which helps capturing most of the
        # information in the original similarity matrices
        for source in document_tokens_matrix_sources:
            similarity_matrix = document_tokens_matrix_sources[ source ]
            mat_rnk = matrix_rank( similarity_matrix )
            u, s, v = svd( similarity_matrix )
            # sort the singular values in descending order. Top ones most important
            s = sorted( s, reverse=True )
            compute_result = self._find_optimal_low_rank_matrix( similarity_matrix, mat_rnk, u, s, v, source )
            approx_similarity_matrices[ source ] = compute_result[ 0 ]
        return approx_similarity_matrices
