"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""
import numpy as np


class GradientDescentOptimizer:

    @classmethod
    def __init__( self, number_iterations ):
        self.number_iterations = number_iterations
        self.sources = [ 'input_output', 'name_desc_edam', 'help_text' ]
        self.best_similarity_score = 1.0

    @classmethod
    def get_uniform_weights( self ):
        """
        Initialize the uniform weight matrices
        """
        weights = dict()
        for item in self.sources:
            weights[ item ] = 1.0 / len( self.sources )
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
    def compute_combined_cost( self, cost ):
        """
        Compute combined gradient for the sources
        """
        return np.sqrt( np.sum( [ item for item in cost ] ) )

    @classmethod
    def compute_loss( self, weight, tool_scores, ideal_tool_score ):
        """
        Compute loss between the ideal score and hypothesised similarity scores
        """
        return weight * tool_scores - ideal_tool_score

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        """
        Apply gradient descent optimizer to find the weights for the sources of annotations of tools
        """
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = dict()
        gradients = dict()
        decay = 0.9
        mu = 0.9
        lr = 0.001
        eps = 1e-8
        for tool_index in range( num_all_tools ):
            tool_id = tools_list[ tool_index ]
            print "Tool index: %d and tool name: %s" % ( tool_index, tool_id )
            # uniform weights to start with for each tool
            weights = self.get_uniform_weights()
            print weights
            cost_iteration = list()
            gradient_io_iteration = list()
            gradient_nd_iteration = list()
            gradient_ht_iteration = list()
            momentum_sources = { 'input_output': 0, 'name_desc_edam': 0, 'help_text': 0 }
            mean_sq_gradient = { 'input_output': 0, 'name_desc_edam': 0, 'help_text': 0 }
            # find optimal weights through these iterations
            for iteration in range( self.number_iterations ):
                cost_sources = list()
                sources_gradient = dict()
                # compute gradient, loss and update weight for each source
                for source in similarity_matrix:
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    # compute losses
                    ideal_tool_score = np.repeat( self.best_similarity_score, num_all_tools )
                    loss = ( weights[ source ] * tools_score_source ) - ideal_tool_score
                    # compute average gradient
                    gradient = np.dot( tools_score_source, loss ) / num_all_tools
                    # gather gradient for a source
                    sources_gradient[ source ] = gradient

                    mean_sq_gradient[ source ] = decay * float( mean_sq_gradient[ source ] ) + ( ( 1 - decay ) * gradient ** 2 )
                    momentum_sources[ source ] = mu * momentum_sources[ source ] + ( lr * gradient ) / ( np.sqrt( mean_sq_gradient[ source ] ) + eps )
                    weights[ source ] -= momentum_sources[ source ]

                    squared_loss = np.sum( loss ** 2 )
                    cost_sources.append( squared_loss )
                mean_cost = np.mean( cost_sources )
                cost_iteration.append( mean_cost )
                # gather gradients
                gradient_io_iteration.append( sources_gradient[ self.sources[ 0 ] ] )
                gradient_nd_iteration.append( sources_gradient[ self.sources[ 1 ] ] )
                gradient_ht_iteration.append( sources_gradient[ self.sources[ 2 ] ] )
            # optimal weights learned
            print "Optimal weights:"
            weights = self.normalize_weights( weights )
            print weights
            print "=================================================="
            tools_optimal_weights[ tool_id ] = weights
            cost_tools[ tool_id ] = cost_iteration
            gradients[ tool_id ] = { self.sources[ 0 ]: gradient_io_iteration, self.sources[ 1 ]: gradient_nd_iteration, self.sources[ 2 ]: gradient_ht_iteration }
        return tools_optimal_weights, cost_tools, gradients
