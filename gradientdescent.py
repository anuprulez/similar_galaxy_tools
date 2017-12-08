"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""

import os
import numpy as np
import operator


class GradientDescentOptimizer:

    @classmethod
    def __init__( self ):
        self.number_iterations = 10000
        self.learning_rate = 0.05

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        num_all_tools = len( tools_list )
        tools_optimal_weights = list()
        for tool_index in range( num_all_tools ):
            random_importance_weights = np.random.random_sample( ( 3 ) )
            ideal_tool_score = np.ones( num_all_tools )
            for iteration in range( self.number_iterations ):
                hypothesis_similarity_matrix = np.zeros( ( num_all_tools ) )
                averate_tool_scores = np.zeros( ( num_all_tools ) )
                for source_index, source in enumerate( [ 'input_output', 'name_desc', 'edam_help' ] ):
                    averate_tool_scores += similarity_matrix[ source ][ tool_index ]
                    # TODO normalize the random weight scores for the sources?
                    hypothesis_similarity_matrix += random_importance_weights[ source_index ] * similarity_matrix[ source ][ tool_index ]
                #assign same tool score when the tool is same
                ideal_tool_score[ tool_index ] = hypothesis_similarity_matrix[ tool_index ]
                loss = hypothesis_similarity_matrix - ideal_tool_score
                cost = np.sum( loss ** 2 ) / ( 2 * num_all_tools )
                print "Iteration: %d and cost: %f" % ( iteration, cost )
                gradient = np.dot( averate_tool_scores, loss.transpose() ) / num_all_tools
                random_importance_weights = random_importance_weights - np.repeat( self.learning_rate * gradient,  len( random_importance_weights ) )
            norm_weights = [ item / np.sum( random_importance_weights ) for item in random_importance_weights ]
            tools_optimal_weights.append( norm_weights )
        return tools_optimal_weights
