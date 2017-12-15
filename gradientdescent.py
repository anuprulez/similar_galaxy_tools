"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""

import numpy as np
import operator


class GradientDescentOptimizer:

    @classmethod
    def __init__( self ):
        # Gradient descent parameters
        self.number_iterations = 200
        self.learning_rate = 0.08
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
    def update_weights( self, weights, gradient ):
        """
        Update the weights for each source using vanilla gradient descent
        """
        for source in weights:
            weights[ source ] = weights[ source ] - self.learning_rate * gradient[ source ]
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
        for tool_index in range( num_all_tools ):
            print "Tool index: %d and tool name: %s" % ( tool_index, tools_list[ tool_index ] )
            # random weights to start with
            random_importance_weights = self.get_random_weights()
            print random_importance_weights
            cost_iteration = list()
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                for source in similarity_matrix:
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    ideal_tool_score = np.repeat( tools_score_source[ tool_index ], num_all_tools )
                    mean_weights = np.mean( random_importance_weights[ source ] )
                    hypothesis_score_source = mean_weights * tools_score_source
                    # compute loss between the ideal score and hypothesised similarity score
                    tools_score_source = tools_score_source.transpose()
                    loss = hypothesis_score_source - ideal_tool_score
                    # add cost for a tool's source
                    cost_sources.append( np.mean( loss ) )
                    # compute gradient
                    gradient = np.outer( tools_score_source, loss )
                    mean_gradient = np.mean( gradient )
                    # add gradient for a source
                    sources_gradient[ source ] = mean_gradient
                cost_iteration.append( np.mean( cost_sources ) )
                random_importance_weights = self.update_weights( random_importance_weights, sources_gradient )
            # add cost for a tool for all iterations
            cost_tools.append( cost_iteration )
            optimal_weights = dict()
            for source in random_importance_weights:
                optimal_weights[ source ] = np.mean( random_importance_weights[ source ] )
            optimal_weights = self.normalize_weights( optimal_weights )
            print optimal_weights
            print "---------------------------------------------------------------------"
            tools_optimal_weights[ tools_list[ tool_index ] ] = optimal_weights
        return tools_optimal_weights, cost_tools, self.number_iterations
