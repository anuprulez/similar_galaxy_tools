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
        self.number_iterations = 100
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.sources = [ 'input_output', 'name_desc', 'edam_help' ]
        
    @classmethod
    def get_random_weights( self, num_all_tools ):
        """
        Initialize the random weight matrices
        """
        weights = dict()
        for item in self.sources:
            weights[ item ] = np.random.random_sample( ( num_all_tools, num_all_tools ) )
        #return weights
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
    def update_weights_adagrad( self, weights, gradient, historical_gradient ):
        """
        Define weight update rule for Gradient Descent with adaptive gradient
        """
        fudge_factor = 1e-6
        step_size = 1e-1
        for source in weights:
            diagonal_comp = np.diag( historical_gradient[ source ] )
            adaptive_grad_comp = np.mean( np.sqrt( diagonal_comp ) )
            adjusted_gradient = gradient[ source ] / ( fudge_factor + np.sqrt( historical_gradient[ source ] ) )
            weights[ source ] = weights[ source ] - step_size * adjusted_gradient
        return weights

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
        hist_gradient_init = np.zeros((num_all_tools, num_all_tools))
        for tool_index in range( num_all_tools ):
            # random weights to start with
            random_importance_weights = self.get_random_weights( num_all_tools )
            print tools_list[ tool_index ]
            previous_update = dict()
            cost_iteration = list()
            historical_gradient = { 'input_output': hist_gradient_init, 'name_desc': hist_gradient_init, 'edam_help': hist_gradient_init }
            # ideal_tool_score = np.ones( num_all_tools )
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                for source in similarity_matrix:
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    ideal_tool_score = np.repeat( tools_score_source[ tool_index ], num_all_tools )
                    # generate a hypothesis similarity matrix using a random weight
                    hypothesis_score_source = np.mean( random_importance_weights[ source ] ) * tools_score_source
                    # compute loss between the ideal score and hypothesised similarity score
                    loss = hypothesis_score_source - ideal_tool_score
                    cost_sources.append( np.mean( loss ) )
                    # compute average gradient
                    gradient = np.dot( tools_score_source.transpose(), loss )
                    sources_gradient[ source ] = gradient
                    historical_gradient[ source ] += gradient ** 2
                cost_iteration.append( np.mean( cost_sources ) )
                # update weights using gradient and previous update
                random_importance_weights = self.update_weights_adagrad( random_importance_weights, sources_gradient, historical_gradient )
            cost_tools.append( cost_iteration )
            optimal_weights = dict()
            for source in random_importance_weights:
                optimal_weights[ source ] = np.mean( random_importance_weights[ source ] )
            optimal_weights = self.normalize_weights( optimal_weights )
            print optimal_weights
            print "---------------------------"
            tools_optimal_weights[ tools_list[ tool_index ] ] = optimal_weights
        return tools_optimal_weights, cost_tools, self.number_iterations
