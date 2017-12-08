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
        self.number_iterations = 1
        self.learning_rate = 0.0005
        self.sources = [ 'input_output', 'name_desc', 'edam_help' ]

    @classmethod
    def get_random_weights( self ):
        weights = dict()
        for item in self.sources:
            weights[ item ] = np.random.random_sample( 1 )
        return weights

    @classmethod
    def update_weights( self, weights, gradient ):
        for source in weights:
            weights[ source ] = weights[ source ] - gradient[ source ]
        return weights

    @classmethod
    def normalize_weights( self, weights ):
        sum_weights = 0
        for source in weights:
            sum_weights += weights[ source ]
        for source in weights:
            weights[ source ] = weights[ source ] / sum_weights
        return weights

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        for tool_index in range( num_all_tools ):
            random_importance_weights = self.get_random_weights()
            ideal_tool_score = np.ones( num_all_tools )
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                for source in similarity_matrix:
                    #print (similarity_matrix[ source ] == np.eye(similarity_matrix[ source ].shape[0])).all()
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    hypothesis_similarity_matrix = random_importance_weights[ source ] * tools_score_source
                    ideal_tool_score[ tool_index ] = hypothesis_similarity_matrix[ tool_index ]
                    loss = hypothesis_similarity_matrix - ideal_tool_score
                    #cost = np.sum( loss ** 2 ) / ( 2 * num_all_tools )
                    #print "Iteration: %d and cost: %f" % ( iteration, cost )
                    gradient = np.dot( tools_score_source, loss.transpose() ) / num_all_tools
                    sources_gradient[ source ] = self.learning_rate * gradient
                random_importance_weights = self.update_weights( random_importance_weights, sources_gradient )

            # normalize the weights so that their sum is 1
            norm_weights = self.normalize_weights( random_importance_weights )
            print tools_list[ tool_index ]
            print norm_weights
            print "---------------------------"
            tools_optimal_weights[ tools_list[ tool_index ] ] = norm_weights
        return tools_optimal_weights
