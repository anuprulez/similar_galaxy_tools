"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""

import numpy as np
import operator
import random as random
import utils


class GradientDescentOptimizer:

    @classmethod
    def __init__( self, number_iterations ):
        # Gradient descent parameters
        self.number_iterations = number_iterations
        self.learning_rate = 0.3
        self.sources = [ 'input_output', 'name_desc_edam_help' ]

    @classmethod
    def get_random_weights( self ):
        """
        Initialize the random weight matrices
        """
        weights = dict()
        for item in self.sources:
            weights[ item ] = random.random()
        return self.normalize_weights( weights )
        
    @classmethod
    def get_initial_learned_cost( self ):
        """
        Initialize the random weight matrices
        """
        cost = dict()
        for item in self.sources:
            cost[ item ] = -1.0
        return cost

    @classmethod
    def step_decay_lr( self, epoch ):
        """
        Decay the learning rate in steps
        """
        drop = 0.99
        epochs_drop = 100.0
        lr_multiplier = np.power( drop, np.floor( ( 1. + epoch ) / epochs_drop ) )
        return self.learning_rate * lr_multiplier

    @classmethod
    def update_weights( self, weights, gradient, learning_rate ):
        """
        Update the weights for each source using vanilla gradient descent + decay in learning rate
        """
        for source in weights:
            weights[ source ] = weights[ source ] - learning_rate * gradient[ source ]
        return weights
        
    @classmethod
    def normalize_weights( self, weights ):
        """
        Normalize the weights so that their sum is 1
        """
        sum_weights = np.sum( [ weights[ item ] for item in weights ] )            
        for source in weights:
            weights[ source ] = float( weights[ source ] ) / sum_weights
        return weights
        
    @classmethod
    def check_optimality( self, cost, learned_cost, convergence_cost_difference ):
        optimal = False
        for item in learned_cost:
            if cost[ item ] - learned_cost[ item ] <  convergence_cost_difference:
                optimal = True
            else:
                optimal = False
                break
        return optimal

    @classmethod
    def adjust_normalized_weights( self, weights ):
        """
        Adjust normalized weights in a way that if weights < 0.1, make them zero
        """
        threshold_weight = 0.5
        # if the weight of any source is less than a threshold, discard that component
        sum_weights = np.sum( [ weights[ item ] for item in weights if weights[ item ] > threshold_weight ] )            
        for source in weights:
            weight = weights[ source ]
            weights[ source ] = float( weight ) / sum_weights if weight > threshold_weight else 0.0
        return weights

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        convergence_cost_difference = 1e-6
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = list()
        for tool_index in range( num_all_tools ):
            print "Tool index: %d and tool name: %s" % ( tool_index, tools_list[ tool_index ] )
            # random weights to start with
            random_importance_weights = self.get_random_weights()
            learned_cost = self.get_initial_learned_cost()
            print random_importance_weights
            cost_iteration = list()
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                cost_source = dict()
                learning_rate = self.step_decay_lr( iteration )         
                for source in similarity_matrix:
                    weight = random_importance_weights[ source ]
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    # an array of ones - the ideal similarity score for a tool with other tools
                    #ideal_tool_score = np.repeat( 1, num_all_tools )
                    ideal_tool_score = np.repeat( tools_score_source[ tool_index ], num_all_tools )
                    # compute loss between the ideal score and hypothesised similarity score
                    hypothesis = weight * tools_score_source
                    loss = ( hypothesis - ideal_tool_score )
                    mean_loss = np.mean( loss )
                    # add cost for a tool's source
                    cost_sources.append( mean_loss )
                    cost_source[ source ] = mean_loss
                    # compute average gradient
                    gradient = np.dot( tools_score_source, loss ) / num_all_tools
                    # add gradient for a source
                    sources_gradient[ source ] = gradient
                mean_cost = np.mean( cost_sources )
                # define a point when to stop learning
                is_optimal = self.check_optimality( cost_source, learned_cost, convergence_cost_difference )
                if is_optimal == True:
                    print "optimal weights learned in %d iterations" % iteration
                    break
                learned_cost = cost_source
                cost_iteration.append( mean_cost )
                random_importance_weights = self.update_weights( random_importance_weights, sources_gradient, learning_rate )
            cost_tools.append( cost_iteration )
            optimal_weights = self.normalize_weights( random_importance_weights )
            print optimal_weights
            print "---------------------------------------------------------------------"
            tools_optimal_weights[ tools_list[ tool_index ] ] = optimal_weights
            del random_importance_weights
            del learned_cost
        learning_rates = [ self.step_decay_lr( iteration ) for iteration in range( self.number_iterations ) ]
        return tools_optimal_weights, cost_tools, self.number_iterations, learning_rates
