"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""
import numpy as np


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
    def backtracking_line_search( self, weights, gradient, similarity, num_all_tools, ideal_score, eta=1, beta=0.3, alpha=0.01 ):
        """
        Find the optimal step size/learning rate for gradient descent
        http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf
        """
        while True:
            eta = beta * eta
            step_update = list()
            is_optimal = False
            for source in weights:
                loss_0 = weights[ source ] * similarity[ source ] - ideal_score[ source ]
                weights[ source ] = weights[ source ] - eta * gradient[ source ]
                loss_1 = weights[ source ] * similarity[ source ] - ideal_score[ source ]
                f_w1 = np.dot( loss_1, loss_1 )
                f_w0 = np.dot( loss_0, loss_0 )
                update = f_w1 - f_w0 + alpha * eta * ( gradient[ source ] ** 2 )
                step_update.append( update )
            is_optimal = all( n <= 0 for n in step_update )
            if is_optimal is True:
                break
        return eta, self.normalize_weights( weights )

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
            lr_iteration = list()
            # find optimal weights through these iterations
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                ideal_score_sources = dict()
                cost_sources = list()
                uniform_cost_sources = list()
                tool_similarity_scores = dict()
                # compute gradient, loss and update weight for each source
                for source in similarity_matrix:
                    weight = weights[ source ]
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    tool_similarity_scores[ source ] = tools_score_source
                    # compute maximum possible scores that a weighted probability can reach
                    # in order to calculate the losses
                    ideal_tool_score = np.repeat( self.best_similarity_score, num_all_tools )
                    ideal_score_sources[ source ] = ideal_tool_score
                    # compute losses
                    loss, uniform_loss = self.compute_loss( weight, uniform_weight, tools_score_source, ideal_tool_score )
                    # compute average gradient
                    gradient = np.dot( tools_score_source, loss ) / num_all_tools
                    # gather gradient for a source
                    sources_gradient[ source ] = gradient
                    squared_loss = np.sum( loss ** 2 )
                    squared_uniform_loss = np.sum( uniform_loss ** 2 )
                    # add cost for a tool's source
                    cost_sources.append( squared_loss )
                    uniform_cost_sources.append( squared_uniform_loss )
                mean_cost = np.mean( cost_sources )
                # compute learning rate using line search
                learning_rate, weights = self.backtracking_line_search( weights, sources_gradient, tool_similarity_scores, num_all_tools, ideal_score_sources )
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
        return tools_optimal_weights, cost_tools, learning_rates, uniform_cost_tools, gradients
