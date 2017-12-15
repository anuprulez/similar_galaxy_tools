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
        self.number_iterations = 3000
        self.learning_rate = 0.5
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
    def update_weights( self, weights, gradient, learning_rate ):
        """
        Update the weights for each source using gradient descent with momentum
        """
        for source in weights:
            weights[ source ] = weights[ source ] - learning_rate * gradient[ source ]
        return weights

    @classmethod
    def update_weights_adagrad( self, weights, gradient, historical_gradient ):
        """
        Define weight update rule for Gradient Descent with adaptive gradient
        """
        stability_factor = 1e-8 # to give numerical stability in case the gradient history is zero
        step_size = 1e-1
        for source in weights:
            #adjusted_gradient = gradient[ source ] / ( stability_factor + np.mean( np.sqrt( np.diag( historical_gradient[ source ] ) ) ) )
            adjusted_gradient = gradient[ source ] / ( stability_factor + np.sqrt( historical_gradient[ source ] ) )
            weights[ source ] = weights[ source ] - self.learning_rate * adjusted_gradient
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
    def step_decay_learning_rate( self, epoch ):
        drop = 0.95
        epochs_drop = 20.0
        lr_multiplier = np.power( drop, np.floor( ( 1. + epoch ) / epochs_drop ) )
        return self.learning_rate * lr_multiplier

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = list()
        for tool_index in range( num_all_tools ):
            print "Tool index: %d and tool name: %s" % ( tool_index, tools_list[ tool_index ] )
            # random weights to start with
            random_importance_weights = self.get_random_weights()
            print random_importance_weights
            cost_iteration = list()
            ideal_tool_score = np.ones( num_all_tools )
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                learning_rate = self.step_decay_learning_rate( iteration )
                for source in similarity_matrix:
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    hypothesis_score_source = random_importance_weights[ source ] * tools_score_source
                    # compute loss between the ideal score and hypothesised similarity score
                    loss = hypothesis_score_source - ideal_tool_score
                    # add cost for a tool's source
                    cost_sources.append( np.mean( loss ) )
                    # compute gradient
                    gradient = np.dot( tools_score_source, loss.transpose() ) / num_all_tools
                    # add gradient for a source
                    sources_gradient[ source ] = gradient
                cost_iteration.append( np.mean( cost_sources ) )
                random_importance_weights = self.update_weights( random_importance_weights, sources_gradient, learning_rate )
            # add cost for a tool for all iterations
            cost_tools.append( cost_iteration )
            random_importance_weights = self.normalize_weights( random_importance_weights )
            print random_importance_weights
            print "---------------------------------------------------------------------"
            tools_optimal_weights[ tools_list[ tool_index ] ] = random_importance_weights
        return tools_optimal_weights, cost_tools, self.number_iterations
