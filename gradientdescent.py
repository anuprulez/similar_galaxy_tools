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
        self.number_iterations = number_iterations
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
    def backtracking_line_search( self, weights, gradient, similarity, num_all_tools ):
        """
        Find the optimal step size/ learning rate for gradient descent
        """
        eta = 1
        beta = 0.6
        alpha = 0.15
        ideal_score = np.repeat( 1, num_all_tools )
        while True:
            eta = beta * eta
            update = dict()
            is_optimal = False
            for source in weights:
                w_1 = weights[ source ] - eta * gradient[ source ]
                loss_0 = weights[ source ] * similarity[ source ] - ideal_score
                loss_1 = w_1 * similarity[ source ] - ideal_score
                f_w1 = np.dot( loss_1, loss_1 )
                f_w0 = np.dot( loss_0, loss_0 )
                update[ source ] = f_w1 - f_w0 + alpha * eta * ( gradient[ source ] ** 2 )
                if update[ source ] <= 0:
                    is_optimal = True
                else:
                    is_optimal = False
                    break
            if is_optimal == True:
                break
        return eta

    @classmethod
    def update_weights( self, weights, gradient, learning_rate ):
        """
        Update the weights for each source using vanilla gradient descent + optimal learning step size
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
    def check_optimality( self, cost, learned_cost, epsilon ):
        """
        Check if the learning in the weights has become stable
        """
        optimal = False
        for item in learned_cost:
            if cost[ item ] - learned_cost[ item ] <  epsilon:
                optimal = True
            else:
                optimal = False
                break
        return optimal

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        epsilon = 1e-7 # convergence_cost_difference
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = list()
        learning_rates = dict()
        for tool_index in range( num_all_tools ):
            print "Tool index: %d and tool name: %s" % ( tool_index, tools_list[ tool_index ] )
            # random weights to start with
            random_importance_weights = self.get_random_weights()
            learned_cost = self.get_initial_learned_cost()
            print random_importance_weights
            cost_iteration = list()
            lr_iteration = list()
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                cost_source = dict()
                tool_similarity_scores = dict()        
                for source in similarity_matrix:
                    weight = random_importance_weights[ source ]
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    tool_similarity_scores[ source ] = tools_score_source
                    # an array of ones - the ideal similarity score for a tool with other tools
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
                learning_rate = self.backtracking_line_search( random_importance_weights, sources_gradient, tool_similarity_scores, num_all_tools )
                lr_iteration.append( learning_rate )
                # define a point when to stop learning
                is_optimal = self.check_optimality( cost_source, learned_cost, epsilon )
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
            learning_rates[ tools_list[ tool_index ] ] = lr_iteration
        return tools_optimal_weights, cost_tools, self.number_iterations, learning_rates
