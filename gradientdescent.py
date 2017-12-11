"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""

import os
import numpy as np
import operator
import time


class GradientDescentOptimizer:

    @classmethod
    def __init__( self ):
        self.number_iterations = 10 
        # parameters related for Gradient Descent
        self.learning_rate = 0.05
        self.momentum = 0.9 # adds a portion of previous update to the current update
        self.sources = [ 'input_output', 'name_desc', 'edam_help' ]

    @classmethod
    def get_random_weights( self, num_all_tools ):
        """
        Initialize the weight matrices with random numbers between 0 and 1
        """
        weights = dict()
        for item in self.sources:
            weights[ item ] = np.random.random_sample( ( num_all_tools, num_all_tools ) )
        return weights

    @classmethod
    def update_weights_momentum( self, weights, gradient, previous_update ):
        """
        Define weight update rule for Gradient Descent with momentum
        """
        weight_update = dict()
        for source in weights:
            velocity = previous_update[ source ] * self.momentum if source in previous_update else 0
            weight_update[ source ] = velocity + self.learning_rate * gradient[ source ]
            weights[ source ] = weights[ source ] - weight_update[ source ]
        return weights, weight_update

    @classmethod
    def update_weights( self, weights, gradient ):
        """
        Define weight update rule for Vanilla Gradient Descent
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
        Gradient descent optimizer to learn importance weights of the 
        sources of annotations for the Galaxy tools
        """
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = list()
        previous_update = dict()
        cost_iteration = list()
        # get random weights
        random_importance_weights = self.get_random_weights( num_all_tools )
        # ideal scores are indentity matrices
        ideal_score_source = np.ones( ( num_all_tools, num_all_tools ) )
        start_time = time.time()
        for iteration in range( self.number_iterations ):
            cost_sources = list()
            gradient_sources = dict()
            # loop through each source
            for source in similarity_matrix:
                tool_score_source = similarity_matrix[ source ]
                # proposed similarity matrix
                hypothesis_score_source = np.dot( tool_score_source, random_importance_weights[ source ].transpose() )
                # as the scores are summed up, it is important to take average
                hypothesis_score_source = hypothesis_score_source / num_all_tools
                # determine how far is our proposed similarity matrix with the ideal one
                loss = hypothesis_score_source - ideal_score_source
                # mean loss for a source
                cost_sources.append( np.mean( loss ) )
                # gradient for a source
                gradient = np.dot( tool_score_source.transpose(), loss ) / num_all_tools
                gradient_sources[ source ] = gradient
            # apply gradient descent and update the weights
            random_importance_weights, weight_update = self.update_weights_momentum( random_importance_weights, gradient_sources, previous_update )
            previous_update = weight_update
            # add cost for an iteration
            cost_iteration.append( np.mean( cost_sources ) )
            print "Iteration: %d" % iteration
        end_time = time.time()
        print "Iterations finished in %d seconds" % int( end_time - start_time )
        optimal_weights = dict()
        for source in random_importance_weights:
            optimal_weights[ source ] = np.mean( random_importance_weights[ source ] )
        optimal_weights = self.normalize_weights( optimal_weights )
        print optimal_weights
        return optimal_weights, cost_iteration, self.number_iterations
