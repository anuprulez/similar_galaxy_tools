"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""

import numpy as np
import operator
import random
import utils


class GradientDescentOptimizer:

    @classmethod
    def __init__( self, number_iterations ):
        self.number_iterations = number_iterations
        self.sources = [ 'input_output', 'name_desc_edam_help' ]
        self.best_similarity_score = 1.0
        
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
    def normalize_weights( self, weights ):
        """
        Normalize the weights
        """
        sum_weights = np.sum( [ weights[ item ] for item in weights ] )
        for source in weights:
            weights[ source ] = weights[ source ] / sum_weights
        return weights
    
    @classmethod
    def check_optimality_gradient( self, gradient, previous_gradient = None ):
        """
        Check if the learning in the weights has become stable
        """
        epsilon = 1e-15
        optimal = False
        for source in gradient:
            if abs( gradient[ source ] ) <  epsilon:
                optimal = True
            elif previous_gradient:
                if abs( previous_gradient[ source ] ) == abs( gradient[ source ] ):
                    optimal = True
                else:
                    optimal = False
                    break
            else:
                optimal = False
                break
        return optimal

    @classmethod
    def backtracking_line_search( self, weights, gradient, similarity, num_all_tools ):
        """
        Find the optimal step size/ learning rate for gradient descent
        """
        eta = 1
        beta = 0.75
        alpha = 0.1
        ideal_score = np.repeat( self.best_similarity_score, num_all_tools )
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
        Update the weights for each source using the learning rate and gradient and then normalize
        """
        for source in weights:
            weights[ source ] = weights[ source ] - learning_rate * gradient[ source ]
        return self.normalize_weights( weights )
        
    @classmethod
    def compute_loss( self, weight, uniform_weight, tool_scores, ideal_tool_score ):
        """
        Compute loss between the ideal score and hypothesised similarity scores
        """
        loss = weight * tool_scores - ideal_tool_score
        uniform_loss = uniform_weight * tool_scores - ideal_tool_score
        return loss, uniform_loss

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        num_all_tools = len( tools_list )
        similarity_matrix_learned = list()
        tools_optimal_weights = dict()
        cost_tools = dict()
        uniform_cost_tools = dict()
        learning_rates = dict()
        gradients = dict()
        uniform_weight = 1. / len( self.sources )
        similarity_matrix_sources = similarity_matrix
        # an array of maximum achievable similarity score for a pair of tools
        ideal_tool_score = np.repeat( self.best_similarity_score, num_all_tools )
        for tool_index in range( num_all_tools ):
            tool_id = tools_list[ tool_index ]
            print "Tool index: %d and tool name: %s" % ( tool_index, tool_id )
            # random weights to start with for each tool
            weights = self.get_random_weights()
            print weights
            cost_iteration = list()
            gradient_io_iteration = list()
            gradient_nd_iteration = list()
            uniform_cost_iteration = list()
            lr_iteration = list()
            previous_gradient = None
            # find optimal weights through these iterations
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                gradient_source = list()
                uniform_cost_sources = list()
                cost_source = dict()
                tool_similarity_scores = dict()
                # compute gradient, loss and update weight for each source   
                for source in similarity_matrix:
                    weight = weights[ source ]
                    tools_score_source = similarity_matrix_sources[ source ][ tool_index ]
                    tool_similarity_scores[ source ] = tools_score_source
                    # compute losses
                    loss, uniform_loss = self.compute_loss( weight, uniform_weight, tools_score_source, ideal_tool_score )
                    mean_loss = np.mean( loss )
                    mean_uniform_loss = np.mean( uniform_loss )
                    # add cost for a tool's source
                    cost_sources.append( mean_loss )
                    uniform_cost_sources.append( mean_uniform_loss )
                    cost_source[ source ] = mean_loss
                    # compute average gradient
                    gradient = np.dot( tools_score_source, loss ) / num_all_tools
                    # gather gradient for a source
                    sources_gradient[ source ] = gradient
                mean_cost = np.mean( cost_sources )
                # compute learning rate using line search
                learning_rate = self.backtracking_line_search( weights, sources_gradient, tool_similarity_scores, num_all_tools )
                lr_iteration.append( learning_rate )
                # gather cost for each iteration
                cost_iteration.append( mean_cost )
                # gather gradients
                gradient_io_iteration.append( sources_gradient[ self.sources[ 0 ] ] )
                gradient_nd_iteration.append( sources_gradient[ self.sources[ 1 ] ] )
                uniform_cost_iteration.append( np.mean( uniform_cost_sources ) )
                # update weights
                weights = self.update_weights( weights, sources_gradient, learning_rate )
                # define a point when to stop learning
                is_optimal = self.check_optimality_gradient( sources_gradient, previous_gradient )
                if is_optimal == True:
                    print "optimal weights learned in %d iterations" % iteration
                    break
                previous_gradient = sources_gradient
            # optimal weights learned
            
            print weights
            print "=================================================="
            tools_optimal_weights[ tool_id ] = weights
            cost_tools[ tool_id ] = cost_iteration
            learning_rates[ tool_id ] = lr_iteration
            uniform_cost_tools[ tool_id ] = uniform_cost_iteration
            gradients[ tool_id ] = { self.sources[ 0 ]: gradient_io_iteration, self.sources[ 1 ]: gradient_nd_iteration }
        return tools_optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients
