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
        weights = np.random.random( 2 )
        return [ float( item ) / np.sum( weights ) for item in weights ]

    @classmethod
    def backtracking_line_search( self, weights, gradient, similarity ):
        """
        Find the optimal step size/ learning rate for gradient descent
        """
        eta = 1
        beta = 0.8
        alpha = 0.1
        ideal_score = 1.0
        while True:
            eta = beta * eta
            update = dict()
            is_optimal = False
            for x in range( len( weights ) ):
                w_1 = weights[ x ] - eta * gradient[ x ]
                loss_0 = weights[ x ] * similarity[ x ] - ideal_score
                loss_1 = w_1 * similarity[ x ] - ideal_score
                f_w1 = np.dot( loss_1, loss_1 )
                f_w0 = np.dot( loss_0, loss_0 )
                update[ x ] = f_w1 - f_w0 + alpha * eta * ( gradient[ x ] ** 2 )
                if update[ x ] <= 0:
                    is_optimal = True
                else:
                    is_optimal = False
                    break
            if is_optimal == True:
                break
        return eta
     
    @classmethod
    def check_optimality_gradient( self, gradient, previous_gradient = None ):
        """
        Check if the learning in the weights has become stable
        """
        epsilon = 1e-30
        optimal = False
        for x in range( len( gradient ) ):
            if abs( gradient[ x ] ) <  epsilon:
                optimal = True
            elif previous_gradient:
                if abs( previous_gradient[ x ] ) == abs( gradient[ x ] ):
                    optimal = True
                else:
                    optimal = False
                    break
            else:
                optimal = False
                break
        return optimal
        
    @classmethod
    def update_weights( self, weights, gradient, learning_rate ):
    
        for x in range( len( weights ) ):
            weights[ x ] = weights[ x ] - learning_rate * gradient[ x ]
        return [ float( item ) / np.sum( weights ) for item in weights ]
     
    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        num_all_tools = len( tools_list )
        similarity_matrix_learned = list()
        tools_optimal_weights = dict()
        cost_tools = list()
        uniform_cost_tools = list()
        learning_rates = dict()
        ideal_score_input_output = np.repeat( self.best_similarity_score, num_all_tools )
        for tool_index in range( num_all_tools ):
            print "Tool index: %d and tool name: %s" % ( tool_index, tools_list[ tool_index ] )
            # random weights to start with
            weights = self.get_random_weights()
            print weights
            cost_iteration = list()
            uniform_cost_iteration = list()
            lr_iteration = list()
            previous_gradient = None
            
            input_output_scores = similarity_matrix[ self.sources[ 0 ] ][ tool_index ]
            name_desc_scores = similarity_matrix[ self.sources[ 1 ] ][ tool_index ]
            
            # take the count of all positive similarity scores
            positive_similarity_name_desc = len( [ x for x in name_desc_scores if x > 0 ] )
            name_desc_loss_value = 1 + np.log( 1 + num_all_tools - positive_similarity_name_desc )
            name_desc_loss_function = np.repeat( name_desc_loss_value, num_all_tools )
            
            for iteration in range( self.number_iterations ):
                # compute hypothesis score
                hypothesis_input_output = weights[ 0 ] * input_output_scores
                hypothesis_name_desc = weights[ 1 ] * name_desc_scores

                # compute hypothesis losses
                hypothesis_loss_input_output = hypothesis_input_output - ideal_score_input_output
                hypothesis_loss_name_desc = hypothesis_name_desc - name_desc_loss_function
                
                # mean loss
                mean_loss = np.mean( np.add( hypothesis_loss_input_output, hypothesis_loss_name_desc ) )
                cost_iteration.append( mean_loss )
                
                # compute original losses
                original_loss_input_output = input_output_scores - ideal_score_input_output
                original_loss_name_desc = name_desc_scores - ideal_score_input_output
                mean_loss_uniform = np.mean( np.add( original_loss_input_output, original_loss_name_desc ) )
                uniform_cost_iteration.append( mean_loss_uniform )
                
                # compute gradient for different sources
                gradient_input_output = np.dot( input_output_scores, hypothesis_loss_input_output ) / num_all_tools
                gradient_name_score = np.dot( name_desc_scores, hypothesis_loss_name_desc ) / num_all_tools

                similarity_sources = [ input_output_scores, name_desc_scores ]
                gradient_sources = [ gradient_input_output, gradient_name_score ]
                
                # compute learning rate
                learning_rate = self.backtracking_line_search( weights, gradient_sources, similarity_sources )
                
                # update weights with new gradient and learning rate
                weights = self.update_weights( weights, gradient_sources, learning_rate )

                lr_iteration.append( learning_rate )
                
                # define a point when to stop learning
                is_optimal = self.check_optimality_gradient( gradient_sources, previous_gradient )
                if is_optimal == True:
                    print "Optimal weights learned in %d iterations" % iteration
                    break
                previous_gradient = gradient_sources
            cost_tools.append( cost_iteration )
            uniform_cost_tools.append( uniform_cost_iteration )
            print weights
            print "=================================================="
            tools_optimal_weights[ tools_list[ tool_index ] ] = { self.sources[ 0 ]: weights[ 0 ], self.sources[ 1 ]: weights[ 1 ] }
            learning_rates[ tools_list[ tool_index ] ] = lr_iteration
        return tools_optimal_weights, cost_tools, self.number_iterations, learning_rates, uniform_cost_tools
