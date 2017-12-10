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
        # Gradient descent parameters
        self.number_iterations = 10000
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.sources = [ 'input_output', 'name_desc', 'edam_help' ]
        
    @classmethod
    def get_random_weights( self ):
        """
        Initialize the random weight matrices
        """
        weights = dict()
        for item in self.sources:
            weights[ item ] = np.random.random_sample( 1 )
        return self.normalize_weights( weights )

    @classmethod
    def update_weights( self, weights, gradient, previous_update ):
        """
        Update the weights for each source using gradient descent with momentum
        """
        weight_update = dict()
        for source in weights:
            velocity = previous_update[ source ] * self.momentum if source in previous_update else 0
            # compute weight update using a fraction of update from previous iteration update
            weight_update[ source ] = velocity + self.learning_rate * gradient[ source ]
            weights[ source ] = weights[ source ] - weight_update[ source ]
        return weights, weight_update

    @classmethod
    def normalize_weights( self, weights ):
        """
        Normalize the weights so that their sum is 1
        """
        sum_weights = 0
        for source in weights:
            sum_weights += weights[ source ]
        for source in weights:
            weights[ source ] = weights[ source ] / sum_weights
        return weights

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = list()
        for tool_index in range( num_all_tools ):
            # random weights to start with
            random_importance_weights = self.get_random_weights()
            print tools_list[ tool_index ]
            print random_importance_weights
            previous_update = dict()
            cost_iteration = list()
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                for source in similarity_matrix:
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    # generate a hypothesis similarity matrix using a random weight
                    hypothesis_similarity_matrix = random_importance_weights[ source ] * tools_score_source
                    self_tool_score = tools_score_source[ tool_index ]
                    ideal_tool_score = np.repeat( self_tool_score, num_all_tools )
                    # compute loss between the ideal score and hypothesised similarity score
                    loss = hypothesis_similarity_matrix - ideal_tool_score
                    cost_sources.append( np.mean( loss ) )
                    # compute average gradient
                    gradient = np.dot( tools_score_source, loss.transpose() ) / num_all_tools
                    sources_gradient[ source ] = self.learning_rate * gradient
                cost_iteration.append( np.mean( cost_sources ) )
                # update weights using gradient and previous update
                random_importance_weights, weight_update = self.update_weights( random_importance_weights, sources_gradient, previous_update )
                previous_update = weight_update
            cost_tools.append( cost_iteration )
            # normalize the weights so that their sum is 1
            norm_weights = self.normalize_weights( random_importance_weights )
            print norm_weights
            print "---------------------------"
            tools_optimal_weights[ tools_list[ tool_index ] ] = norm_weights
        return tools_optimal_weights, cost_tools, self.number_iterations
