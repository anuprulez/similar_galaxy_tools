"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""
import numpy as np
import json


class GradientDescentOptimizer:

    @classmethod
    def __init__( self, number_iterations, data_sources, seed=2 ):
        self.number_iterations = number_iterations
        self.sources = data_sources
        self.best_similarity_score = 1.0
        np.random.seed( seed )

    @classmethod
    def get_random_weights( self ):
        """
        Initialize the uniform weight matrices
        """
        weights = dict()
        random_weights = np.random.rand( len( self.sources ), 1 )
        random_weights = [ float( item ) / np.sum( random_weights ) for item in random_weights ]
        for index, item in enumerate( self.sources ):
            weights[ item ] = random_weights[ index ]
        return weights

    @classmethod
    def normalize_weights( self, weights ):

        """
        Normalize the weights
        """
        sum_weights = np.sum( [ weights[ item ] for item in weights ] )
        for source in weights:
            weights[ source ] = weights[ source ] / float( sum_weights )
        return weights

    @classmethod
    def update_weights( self, weights, gradients, prev_update, iteration, eta, gamma=0.9, lr_decay=0.1 ):
       """
       Update weights with momentum and time-based decay of learning rate
       """
       eta = eta / ( 1. + ( lr_decay * iteration ) )
       for source in weights:
           new_update = gamma * prev_update[ source ] - eta * gradients[ source ]
           weights[ source ] = weights[ source ] + new_update
           prev_update[ source ] = new_update
       return self.normalize_weights( weights ), prev_update, eta 

    @classmethod
    def compute_loss( self, weight, uniform_weight, tool_scores, ideal_tool_score ):
        """
        Compute loss between the ideal score and hypothesised similarity scores
        """
        loss = weight * tool_scores - ideal_tool_score
        uniform_loss = uniform_weight * tool_scores - ideal_tool_score
        return loss, uniform_loss

    @classmethod
    def verify_gradients( self, ideal_score, weights, tool_index, sources_gradient, similarity_matrix, epsilon=1e-4 ):
        """
        Find the approximation of the gradient using mean squared error 
        """
        derivative_sources = dict()
        for source in weights:
            weight = weights[ source ]
            tools_scores_source = similarity_matrix[ source ][ tool_index ]
            loss_0 = ( weight + epsilon ) * tools_scores_source - ideal_score
            sq_error_0 = np.mean( loss_0 ** 2 )
            loss_1 = ( weight - epsilon ) * tools_scores_source - ideal_score
            sq_error_1 = np.mean( loss_1 ** 2 )
            derivative_sources[ source ] = ( sq_error_0 - sq_error_1 ) / ( 2 * epsilon )
        return derivative_sources

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = dict()
        uniform_cost_tools = dict()
        learning_rates = dict()
        gradients = dict()
        learning_rates_iterations_tool = dict()
        approx_gd_tools = dict()
        actual_gd_tools = dict()
        uniform_weight = 1. / len( self.sources )
        for tool_index in range( num_all_tools ):
            tool_id = tools_list[ tool_index ]
            print "Tool index: %d and tool name: %s" % ( tool_index, tool_id )
            # uniform weights to start with for each tool
            weights = self.get_random_weights()
            print weights
            cost_iteration = list()
            gradient_io_iteration = list()
            gradient_nd_iteration = list()
            gradient_ht_iteration = list()
            uniform_cost_iteration = list()
            approx_gd = list()
            actual_gd = list()
            lr_iteration = list()
            lr_rates = list()
            ideal_tool_score = np.repeat( self.best_similarity_score, num_all_tools )
            prev_weights_updates = { self.sources[ 0 ]: 0.0, self.sources[ 1 ]: 0.0, self.sources[ 2 ]: 0.0 }
            eta = 1e-1
            # find optimal weights through these iterations
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                uniform_cost_sources = list()
                tool_similarity_scores = dict()
                # compute gradient, loss and update weight for each source
                for source in similarity_matrix:
                    weight = weights[ source ]
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    # adjust for the position of the tool itself so that it does not account for the loss computation
                    tool_similarity_scores[ source ] = tools_score_source
                    # compute losses
                    loss, uniform_loss = self.compute_loss( weight, uniform_weight, tools_score_source, ideal_tool_score )
                    # compute average gradient
                    gradient = 2 * np.dot( tools_score_source, loss ) / num_all_tools
                    # gather gradient for a source
                    sources_gradient[ source ] = gradient
                    squared_loss = np.mean( loss ** 2 )
                    squared_uniform_loss = np.mean( uniform_loss ** 2 )
                    # add cost for a tool's source
                    cost_sources.append( squared_loss )
                    uniform_cost_sources.append( squared_uniform_loss )
                mean_cost = np.mean( cost_sources )
                derivative_sources = self.verify_gradients( ideal_tool_score, weights, tool_index, sources_gradient, similarity_matrix )
                approx_gd.append( derivative_sources )
                actual_gd.append( sources_gradient )
                weights, prev_weights_updates, learning_rate = self.update_weights( weights, sources_gradient, prev_weights_updates, iteration, eta )
                lr_iteration.append( learning_rate )
                # gather cost for each iteration
                cost_iteration.append( mean_cost )
                # gather gradients
                gradient_io_iteration.append( sources_gradient[ self.sources[ 0 ] ] )
                gradient_nd_iteration.append( sources_gradient[ self.sources[ 1 ] ] )
                gradient_ht_iteration.append( sources_gradient[ self.sources[ 2 ] ] )
                uniform_cost_iteration.append( np.mean( uniform_cost_sources ) )
            # optimal weights learned
            print weights
            print "=================================================="
            tools_optimal_weights[ tool_id ] = weights
            learning_rates[ tool_id ] = lr_iteration
            cost_tools[ tool_id ] = cost_iteration
            uniform_cost_tools[ tool_id ] = uniform_cost_iteration[ 0 ]
            gradients[ tool_id ] = { self.sources[ 0 ]: gradient_io_iteration, self.sources[ 1 ]: gradient_nd_iteration, self.sources[ 2 ]: gradient_ht_iteration }
            approx_gd_tools[ tool_id ] = approx_gd
            actual_gd_tools[ tool_id ] = actual_gd

        with open( "data/approx_gd_tools.json", 'w' ) as file:
            file.write( json.dumps( approx_gd_tools ) )

        with open( "data/actual_gd_tools.json", 'w' ) as file:
            file.write( json.dumps( actual_gd_tools ) )
        # write learning rates for all tools as json file iterations
        with open( "data/learning_rates.json", 'w' ) as file:
            file.write( json.dumps( learning_rates ) )
        return tools_optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients
