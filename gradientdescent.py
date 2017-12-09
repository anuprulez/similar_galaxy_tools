"""
Performs optimization to find optimal importance weights for the sources
of tokens using Gradient Descent
"""

import os
import numpy as np
import operator


class GradientDescentOptimizer:

    @classmethod
    def __init__( self ):
        self.number_iterations = 10000
        self.learning_rate = 0.05
        self.sources = [ 'input_output', 'name_desc', 'edam_help' ]
        self.momentum = 0.5

    @classmethod
    def get_random_weights( self ):
        weights = dict()
        for item in self.sources:
            weights[ item ] = np.random.random_sample( 1 )
        return self.normalize_weights( weights )

    @classmethod
    def update_weights( self, weights, gradient, previous_update ):
        for source in weights:
            weight_update = self.learning_rate * gradient[ source ]
            velocity = previous_update[ source ] * self.momentum if source in previous_update else 0
            velocity = velocity + weight_update
            weights[ source ] = weights[ source ] - velocity
        return weights

    @classmethod
    def normalize_weights( self, weights ):
        sum_weights = 0
        for source in weights:
            sum_weights += weights[ source ]
        for source in weights:
            weights[ source ] = weights[ source ] / sum_weights
        return weights

    @classmethod
    def gradient_descent( self, similarity_matrix, tools_list ):
        num_all_tools = len( tools_list )
        tools_optimal_weights = dict()
        cost_tools = list()
        for tool_index in range( num_all_tools ):
            random_importance_weights = self.get_random_weights()
            print tools_list[ tool_index ]
            print random_importance_weights
            previous_update = dict()
            cost_iteration = list()
            for iteration in range( self.number_iterations ):
                sources_gradient = dict()
                cost_sources = list()
                for source in similarity_matrix:
                    tools_score_source = similarity_matrix[ source ][ tool_index ]
                    hypothesis_similarity_matrix = random_importance_weights[ source ] * tools_score_source
                    self_tool_score = tools_score_source[ tool_index ]
                    ideal_tool_score = np.repeat( self_tool_score, num_all_tools )
                    loss = hypothesis_similarity_matrix - ideal_tool_score
                    average_cost = np.linalg.norm( loss ) / ( num_all_tools )
                    cost_sources.append( average_cost )
                    gradient = np.dot( tools_score_source, loss.transpose() ) / num_all_tools
                    sources_gradient[ source ] = self.learning_rate * gradient
                cost_iteration.append( np.linalg.norm( cost_sources ) )
                random_importance_weights = self.update_weights( random_importance_weights, sources_gradient, previous_update )
                previous_update = sources_gradient
            cost_tools.append( cost_iteration )
            # normalize the weights so that their sum is 1
            norm_weights = self.normalize_weights( random_importance_weights )
            print norm_weights
            print "---------------------------"
            #norm_weights = { 'input_output': 0.75, 'name_desc': 0.15, 'edam_help': 0.1 }
            tools_optimal_weights[ tools_list[ tool_index ] ] = norm_weights
        return tools_optimal_weights, cost_tools
