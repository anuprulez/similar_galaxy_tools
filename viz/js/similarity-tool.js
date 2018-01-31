

$(document).ready(function(){

    var similarityData = null,
        list_tool_names = null,
        path = "data/similarity_matrix.json"; // https://raw.githubusercontent.com/anuprulez/similar_galaxy_tools/master/viz/data/similarity_matrix.json
    if ( path === "" ) {
        console.error( "Error in loading JSON file" );
        return;
    }
    $.getJSON( path, function( data ) {
        var toolIdsTemplate = "";
            list_tool_names = data[ data.length - 1 ]
            slicedData = data.slice( 0, data.length - 1 );
        // sort the tools in ascending order of their ids
        similarityData = slicedData.sort(function(a, b) {
            if( a.root_tool !== undefined && b.root_tool !== undefined ) {
                var first_id = a.root_tool.toLowerCase(),
                    second_id = b.root_tool.toLowerCase();
                if( first_id < second_id ) { return -1; }
                if( first_id > second_id ) { return 1; }
                return 0;
            }
        });
        for( var counter = 0, len = similarityData.length; counter < len; counter++ ) {
            var toolResults = similarityData[ counter ]; 
            if( toolResults.root_tool !== undefined ) {
                toolIdsTemplate += "<option value='" + toolResults.root_tool + "'>" + toolResults.root_tool + "</options>";
            }
        } // end of for loop
        $( ".tool-ids" ).append( toolIdsTemplate );
    });

    // Fire on change of a tool to show similar tools and plot cost vs iteration
    $( ".tool-ids" ).on( 'change', function( e ) {
        e.preventDefault();
        var selectedToolId = e.target.value,
            data = similarityData,
            availableSimilarTool = false,
            $el_tools = $( ".tool-results" );
        $el_tools.empty();
        for( var counter = 0, len = data.length; counter < len; counter++ ) {
            var toolResults = data[ counter ];
            if ( toolResults.root_tool === selectedToolId ) {
                var similarRootTools = toolResults.similar_tools,
                    template = "";
                rootTool = similarRootTools.slice( 0, 1 );
                console.log( rootTool );
                similarTools = similarRootTools.slice( 1, );
                // make html for the selected tool
                $el_tools.append( createHTML( rootTool, selectedToolId, "Selected tool: <b>" +  selectedToolId + "</b>", "", true ) );
                // show optimal weights
                //$el_tools.append( showWeights( toolResults.optimal_weights, "" ) );
                
                // make html for similar tools found by optimizing BM25 scores using Gradient Descent
                $el_tools.append( createHTML( similarTools, selectedToolId, "Similar tools for the selected tool: <b>" +  selectedToolId + " </b>found by optimal combination (Gradient Descent) of probabilities</h4>", "Weighted probability score", false ) );
                
                // make html for similar tools found using average scores of BM25
                //$el_tools.append( createHTML( aveToolScores, selectedToolId, "Similar tools for the selected tool: <b>" +  selectedToolId + " </b>found using average probabilities</h4>", "Average probability score", false ) );
                
                // plot optimal vs average scores
                //$el_tools.append( "<div id='scatter-optimal-average'></div>" );
                //plotScatterOptimalAverageScores( toolResults, "scatter-optimal-average", selectedToolId );
 
                //$el_tools.append( "<div id='scatter-optimal-average-top-results'></div>" );
                //plotScatterOptimalAverageScoresTopResults( toolResults, "scatter-optimal-average-top-results", selectedToolId );
                
                //$el_tools.append( "<div id='tool-combined-gradient-iterations'></div>" );
                //plotCombinedGradients( toolResults.gradient_io_iteration, toolResults.gradient_nd_iteration, 'tool-combined-gradient-iterations', selectedToolId );

                //$el_tools.append( "<div id='tool-io-gradient-iterations'></div>" );
                //plotGradients( toolResults.gradient_io_iteration, "tool-io-gradient-iterations", selectedToolId, "Gradient vs iterations for Input Output" );
                
                //$el_tools.append( "<div id='tool-nd-gradient-iterations'></div>" );
                //plotGradients( toolResults.gradient_nd_iteration, "tool-nd-gradient-iterations", selectedToolId, "Gradient vs iterations for Name Desc" );
                
                // plot loss drop vs iterations
                //$el_tools.append( "<div id='tool-cost-iterations'></div>" );
                //plotCostVsIterations( toolResults, "tool-cost-iterations", selectedToolId );
                
                // plot learning rate vs iterations
                //$el_tools.append( "<div id='learning-rate-iterations'></div>" );
                //plotLearningRatesVsIterations( toolResults, "learning-rate-iterations", selectedToolId );
                availableSimilarTool = true;
                break;
            }
         } // end of for loop
         if ( !availableSimilarTool ) {
             $el_tools.empty().html( "<p class='no-similar-tool-msg'>No similar tool available. <p>" );
         }
    });
    
    var showWeights = function( weights, headerText ) {
        var template = "";
        for( var item in weights ) {
            if( item === "input_output" ) {
                template += "<div>" + "Input and output file types ( weight_1 ): <b>" + toPrecisionNumber( weights[ item ] ) + "</b></div>";
            }
            else if( item === "name_desc_edam_help" ) {
                template += "<div>" + "Name, description, help and EDAM ( weight_2 ): <b>" + toPrecisionNumber( weights[ item ] )  + "</b></div>";
            }
        }
        template += "<p>Score = weight_1 * probability_input_output + weight_2 * probability_name_desc_edam_help</p>";
        template += "</div>";
        return template;
    };
    
    var toPrecisionNumber = function( number ) {
        return Math.round( parseFloat( number ) * 100) / 100;
    };

    var createHTML = function( toolScores, originalToolId, headerText, scoreHeaderText, isHeader ) {
        var template = "<div class='table-header-text'>" + headerText + "</div>",
            prevRank = 0,
            prevScore = 0,
            maxShowStringLen = 30;
        template += "<div class='table-responsive'><table class='table table-bordered table-striped thead-dark'><thead>";
        template += "<th>S.No.</th>";
        template += "<th>Id</th>";
        if ( !isHeader ) {
            //template += "<th> Input output probability score </th>";
            //template += "<th> Name desc. Edam help probability score </th>";
            template += "<th> " + scoreHeaderText + "</th>";
            template += "<th> Rank </th>";
        }
        template += "<th> Name and description </th>";
        template += "<th> Input files </th>";
        template += "<th> Output files </th>";
        template += "<th> Help text (what it does) </th>";
        template += "<th> EDAM </th>";
        template += "</thead><tbody>";
        
        for( var counter_ts = 0, len_ts = toolScores.length; counter_ts < len_ts; counter_ts++ ) {
            var tool = toolScores[ counter_ts ],
                toolScore = tool.score,
                rank = 0,
                helpText = tool.what_it_does,
                nameDesc = tool.name_description;
                showHelpText = ( helpText.length > maxShowStringLen && !isHeader ) ? helpText.substring(0, maxShowStringLen) + "..." : helpText;

            rank = ( prevScore === toolScore ) ? prevRank : parseInt( counter_ts + 1 );
            template += "<tr>";
            template += "<td>" + parseInt( counter_ts + 1 ) + "</td>";
            template += "<td>" + tool.id + "</td>";
            if ( !isHeader ) {
                //template += "<td>" + tool.input_output_score + "</td>";
                //template += "<td>" + tool.name_desc_edam_help_score + "</td>";
                template += "<td>" + toolScore + "</td>";
                template += "<td>" + rank + "</td>";
            }
            template += "<td>" + nameDesc + "</td>";
            template += "<td>" + tool.input_types + "</td>";
            template += "<td>" + tool.output_types + "</td>";
            template += "<td title='"+ helpText +"'>" + showHelpText + "</td>";
            template += "<td>" + tool.edam_text + "</td>";
            template += "</tr>";
            prevRank = rank;
            prevScore = toolScore;
        }
        template += "</tbody></table></div>";
        return template;
    };

    var plotCombinedGradients = function( dataIOGradients, dataNDGradients, $elPlot, selectedToolId ) {
        var iterations = dataIOGradients.length,
            xAxis = [],
            combinedGradients = [];
        for( var i = 0; i < iterations; i++ ) {
            xAxis.push( i + 1 );
            combinedGradients.push( dataIOGradients[ i ] * dataIOGradients[ i ] + dataNDGradients[ i ] * dataNDGradients[ i ] );
        }
        
	var trace1 = {
	    x: xAxis,
	    y: combinedGradients,
	    type: 'scatter',
	    mode: 'lines',
	    name: 'Combined gradient drop vs iterations'
	};
	
	var plotData = [ trace1 ];
	
	var layout = {
            title: "Combined gradient drop vs interations for the tool: " + selectedToolId,
            xaxis: {
                title: 'Iterations'
            },
            yaxis: {
                title: 'Combined gradient'
            }
        };
	Plotly.newPlot( $elPlot, plotData, layout );
    };
    
    var plotGradients = function( data, $elPlot, selectedToolId, plotTitle ) {
        var gradientIterations = data,
            iterations = gradientIterations.length,
            x_axis = [];
        for( var i = 0; i < iterations; i++ ) {
            x_axis.push( i + 1 );
        }
        
	var trace1 = {
	    x: x_axis,
	    y: gradientIterations,
	    type: 'scatter',
	    mode: 'lines',
	    name: 'Gradient drop vs iterations'
	};
	
	var plotData = [ trace1 ];
	
	var layout = {
            title: plotTitle,
            xaxis: {
                title: 'Iterations'
            },
            yaxis: {
                title: 'Gradient'
            }
        };
	Plotly.newPlot( $elPlot, plotData, layout );
    };
    
    var plotCostVsIterations = function( toolScores, $elPlot, selectedToolId ) {
        var costIterations = toolScores.cost_iterations,
            costUniformTools = toolScores.uniform_cost_tools,
            iterations = costIterations.length,
            x_axis = [];
        for( var i = 0; i < iterations; i++ ) {
            x_axis.push( i + 1 );
        }
        
	var trace1 = {
	    x: x_axis,
	    y: costIterations,
	    type: 'scatter',
	    mode: 'lines',
	    name: 'Optimal loss vs iterations'
	};
	
	var trace2 = {
	    x: x_axis,
	    y: costUniformTools,
	    type: 'scatter',
	    mode: 'lines',
	    name: 'Average loss'
	};
	
	var data = [ trace1, trace2 ];
	
	var layout = {
            title:'Cost vs Iterations for the tool: ' + selectedToolId,
            xaxis: {
                title: 'Iterations'
            },
            yaxis: {
                title: 'Cost (Learned and average)'
            }
        };
	Plotly.newPlot( $elPlot, data, layout );
    };
    
    var plotScatterOptimalAverageScoresTopResults = function( scores, $elPlot, selectedToolId ) {
        var optimal_scores = scores.similar_tools,
            average_scores = scores.average_similar_tools,
            x_axis = [],
            optimal_scores_list = [],
            average_scores_list = [],
            optimal_tool_names = [],
            average_tool_names = [];
        for( var i = 0, len = optimal_scores.length; i < len; i++ ) {
            x_axis.push( i + 1 );
            optimal_scores_list.push( optimal_scores[ i ].score );
            average_scores_list.push( average_scores[ i ].score );
            optimal_tool_names.push( optimal_scores[ i ].id );
            average_tool_names.push( average_scores[ i ].id );
        }
        
        var trace1 = {
	    x: x_axis,
	    y: optimal_scores_list,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Optimal probability scores',
	    text: optimal_tool_names
	};

	var trace2 = {
	    x: x_axis,
	    y: average_scores_list,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Average probability scores',
	    text: average_tool_names
	};

	var data = [ trace1, trace2 ];

	var layout = {
	    xaxis: {
	        range: [ -1, optimal_scores.length + 1 ],
	        title: "Tools"
	    },
	    yaxis: {
	        range: [ -0.1, 0.1 ],
	        title: "Similarity score"
	    },
	    title:'Scatter plot of optimal and average probability scores using input, output types for tool: ' + selectedToolId
	};
	Plotly.newPlot( $elPlot, data, layout );
    };
    
    var plotScatterOptimalAverageScores = function( scores, $elPlot, selectedToolId ) {
        var optimal_scores = scores.optimal_similar_scores,
            average_scores = scores.average_similar_scores,
            x_axis = [];
        for( var i = 0, len = optimal_scores.length; i < len; i++ ) {
            x_axis.push( i + 1 );
        }
        
        var trace1 = {
	    x: x_axis,
	    y: optimal_scores,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Optimal probability scores',
	    text: list_tool_names.list_tools
	};

	var trace2 = {
	    x: x_axis,
	    y: average_scores,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Average probability scores',
	    text: list_tool_names.list_tools
	};

	var data = [ trace1, trace2 ];
	var layout = {
	    xaxis: {
	        range: [ -5, optimal_scores.length + 5 ],
	        title: "Tools"
	    },
	    yaxis: {
	        range: [ -0.1, 0.1 ],
	        title: "Similarity score"
	    },
	    title:'Scatter plot of optimal and average combination of probability scores for tool: ' + selectedToolId
	};
	Plotly.newPlot( $elPlot, data, layout );
    };
    
    var plotLearningRatesVsIterations = function( toolScores, $elPlot, selectedToolId ) {
        var lrIterations = toolScores.learning_rates_iterations,
            iterations = lrIterations.length,
            x_axis = [];
        for( var i = 0; i < iterations; i++ ) {
            x_axis.push( i + 1 );
        }
	var data = [{
	    x: x_axis,
	    y: lrIterations,
	    type: 'scatter'
	}];
	
	var layout = {
            title:'Learning rates vs Iterations for the tool: ' + selectedToolId,
            xaxis: {
                title: 'Iterations'
            },
            yaxis: {
                title: 'Learning rate / Step size'
            }
        };
	Plotly.newPlot( $elPlot, data, layout );
    };
});

