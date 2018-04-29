"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""
import numpy as np
import json


class GradientDescentOptimizer:

    @classmethod
    def __init__( self, number_iterations, data_sources ):
        self.number_iterations = number_iterations
        self.sources = data_sources
        self.best_similarity_score = 1.0

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
    def backtracking_line_search( self, weights, gradient, similarity, num_all_tools, eta=1, beta=0.8, alpha=0.5 ):
        """
        Find the optimal step size/learning rate for gradient descent
        http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf
        https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
        """
        prev_gradient_update = None
        learning_rates = list()
        while True:
            step_update = list()
            learning_rates.append( eta )
            for source in weights:
                loss_0 = weights[ source ] * similarity[ source ] - np.ones( [ num_all_tools ] )
                loss_1 = ( weights[ source ] - eta * gradient[ source ] * similarity[ source ] ) - np.ones( [ num_all_tools ] )
                f_w1 = np.mean( loss_1 ** 2 )
                f_w0 = np.mean( loss_0 ** 2 )
                f_w0 = f_w0 - alpha * eta * ( gradient[ source ] ** 2 )
                if( f_w1 > f_w0 ):
                    step_update.append( True )
                else:
                    step_update.append( False )
            if all( n == False for n in step_update ) is True:
                weights[ source ] = weights[ source ] - eta * ( gradient[ source ] )
                break
            eta = beta * eta
        return eta, self.normalize_weights( weights ), learning_rates

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
        tools_optimal_weights = dict()
        cost_tools = dict()
        uniform_cost_tools = dict()
        learning_rates = dict()
        gradients = dict()
        learning_rates_iterations_tool = dict()
        lr_file_path = "data/learning_rates.json"
        uniform_weight = 1. / len( self.sources )
        ideal_tool_score = np.repeat( self.best_similarity_score, num_all_tools )
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
            lr_iteration = list()
            lr_rates = list()
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
                    tool_similarity_scores[ source ] = tools_score_source
                    # compute losses
                    loss, uniform_loss = self.compute_loss( weight, uniform_weight, tools_score_source, ideal_tool_score )
                    # compute average gradient
                    gradient = 2 * np.mean( np.dot( tools_score_source, loss ) )
                    # gather gradient for a source
                    sources_gradient[ source ] = gradient
                    squared_loss = np.mean( loss ** 2 )
                    squared_uniform_loss = np.mean( uniform_loss ** 2 )
                    # add cost for a tool's source
                    cost_sources.append( squared_loss )
                    uniform_cost_sources.append( squared_uniform_loss )
                mean_cost = np.mean( cost_sources )
                # compute learning rate using line search
                learning_rate, weights, lr_rates_drop = self.backtracking_line_search( weights, sources_gradient, tool_similarity_scores, num_all_tools )
                lr_iteration.append( learning_rate )
                lr_rates.append( lr_rates_drop )
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
            learning_rates_iterations_tool[ tool_id ] = lr_rates
            gradients[ tool_id ] = { self.sources[ 0 ]: gradient_io_iteration, self.sources[ 1 ]: gradient_nd_iteration, self.sources[ 2 ]: gradient_ht_iteration }
        # write learning rates for all tools as json file iterations
        with open( lr_file_path, 'w' ) as file:
            file.write( json.dumps( learning_rates_iterations_tool ) )
        return tools_optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients
