

$(document).ready(function(){

    var similarityData = null,
        list_tool_names = null,
        path = ""; // download the similarity json file from https://github.com/anuprulez/large_files_repository
        
    if ( path === "" ) {
        console.error( "Download json file from 'https://github.com/anuprulez/large_files_repository' to your local computer and set this variable" );
        return;
    }
    $.getJSON( path, function( data ) {
        console.log(data);
        var toolIdsTemplate = "";
            list_tool_names = data[ data.length - 1 ]
            slicedData = data.slice( 0, data.length - 1 );
        // sort the tools in ascending order of their ids
        similarityData = slicedData.sort(function(a, b) {
            if( a.root_tool.id !== undefined && b.root_tool.id !== undefined ) {
                var first_id = a.root_tool.id.toLowerCase(),
                    second_id = b.root_tool.id.toLowerCase();
                if( first_id < second_id ) { return -1; }
                if( first_id > second_id ) { return 1; }
                return 0;
            }
        });
        for( var counter = 0, len = similarityData.length; counter < len; counter++ ) {
            var toolResults = similarityData[ counter ]; 
            if( toolResults.root_tool.id !== undefined ) {
                toolIdsTemplate += "<option value='" + toolResults.root_tool.id + "'>" + toolResults.root_tool.id + "</options>";
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
            if ( toolResults.root_tool.id === selectedToolId ) {
                var toolScores = toolResults.similar_tools,
                    aveToolScores = toolResults.aggregate_similar_tools,
                    template = "";
                    
                // make html for the selected tool
                $el_tools.append( createHTML( [ toolResults.root_tool ], selectedToolId, "<h4>Selected tool: " +  selectedToolId + "</h4>", "Score(optimal weights)" ) );
                // show optimal weights
                $el_tools.append( showWeights( toolResults.optimal_weights, "Optimal importance weights learned" ) );
                
                // make html for similar tools found by optimizing BM25 scores using Gradient Descent
                $el_tools.append( createHTML( toolScores, selectedToolId, "<h4> Similar tools for the selected tool: " +  selectedToolId + " found using optimal weights for the sources</h4>", "Score" ) );
                
                // make html for similar tools found using aggregate scores of BM25
                $el_tools.append( createHTML( aveToolScores, selectedToolId, "<h4> Similar tools for the selected tool: " +  selectedToolId + " found using aggregate BM25 similarity scores</h4>", "Score" ) );
                
                // plot loss drop vs iterations
                $el_tools.append( "<div id='tool-cost-iterations'></div>" );
                plotCostVsIterations( toolResults, "tool-cost-iterations", selectedToolId );
                
                // plot optimal vs aggregate scores
                $el_tools.append( "<div id='scatter-optimal-aggregate'></div>" );
                plotScatterOptimalAggreagateScores( toolResults, "scatter-optimal-aggregate", selectedToolId );
                
                // plot learning rate vs iterations
                $el_tools.append( "<div id='learning-rate-iterations'></div>" );
                plotLearningRatesVsIterations( toolResults, "learning-rate-iterations", selectedToolId );
                availableSimilarTool = true;
                break;
            }
         } // end of for loop
         if ( !availableSimilarTool ) {
             $el_tools.empty().html( "<p class='no-similar-tool-msg'>No similar tool available. <p>" );
         }
    });
    
    sumArray = function( a, b ) {
       return a + b;
    };
    
    var showWeights = function( weights, headerText ) {
        var template = "";
        template = "<div><h4> " + headerText + " </h4>"
        for( var item in weights ) {
            if( item === "input_output" ) {
                template += "<div>" + "Input and output file types ( weight_1 ): <b>" + toPrecisionNumber( weights[ item ] ) + "</b></div>";
            }
            else if( item === "name_desc_edam_help" ) {
                template += "<div>" + "Name, description, help and EDAM ( weight_2 ): <b>" + toPrecisionNumber( weights[ item ] )  + "</b></div>";
            }
        }
        template += "<p>Score = weight_1 * source_1 + weight_2 * source_2</p>";
        template += "</div>";
        return template;
    };
    
    var toPrecisionNumber = function( number ) {
        return Math.round( parseFloat( number ) * 100) / 100;
    };

    var createHTML = function( toolScores, originalToolId, headerText, scoreHeaderText ) {
        var template = headerText;
        template += "<table><thead>";
        template += "<th>Id</th>";
        template += "<th> Input output score - Source_1 </th>";
        template += "<th> Name desc. Edam help score - Source_2 </th>";
        template += "<th> " + scoreHeaderText + "</th>";
        template += "<th> Rank </th>";
        template += "<th> Name and description </th>";
        template += "<th> Input files </th>";
        template += "<th> Output files </th>";
        template += "<th> Help text (what it does) </th>";
        template += "<th> EDAM </th>";
        template += "</thead><tbody>";
        sum = 0;
        for( var counter_ts = 0, len_ts = toolScores.length; counter_ts < len_ts; counter_ts++ ) {
            var tool = toolScores[ counter_ts ],
                tool_score = tool.score.toFixed( 2 );
            sum += parseFloat( tool_score );
            template += "<tr>";
            template += "<td>" + tool.id + "</td>";
            template += "<td>" + tool.input_output_score + "</td>";
            template += "<td>" + tool.name_desc_edam_help_score + "</td>";
            template += "<td>" + tool_score + "</td>";
            template += "<td>" + parseInt( counter_ts + 1 ) + "</td>";
            template += "<td>" + tool.name_description + "</td>";
            template += "<td>" + tool.input_types + "</td>";
            template += "<td>" + tool.output_types + "</td>";
            template += "<td>" + tool.what_it_does + "</td>";
            template += "<td>" + tool.edam_text + "</td>";
            template += "</tr>";
        }
        template += "</tbody></table>";
        return template;
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
	    name: 'Aggregate loss'
	};
	
	var data = [ trace1, trace2 ];
	
	var layout = {
            title:'Cost vs Iterations for the tool: ' + selectedToolId,
            xaxis: {
                title: 'Iterations'
            },
            yaxis: {
                title: 'Cost (Learned and aggregate)'
            }
        };

	Plotly.newPlot( $elPlot, data, layout );
    };
    
    var plotScatterOptimalAggreagateScores = function( scores, $elPlot, selectedToolId ) {
        var optimal_scores = scores.optimal_similar_scores,
            aggregate_scores = scores.aggregate_similar_scores,
            x_axis = [];
        for( var i = 0, len = optimal_scores.length; i < len; i++ ) {
            x_axis.push( i + 1 );
        }
        
        var trace1 = {
	    x: x_axis,
	    y: optimal_scores,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Optimal scores',
	    text: list_tool_names.list_tools
	};

	var trace2 = {
	    x: x_axis,
	    y: aggregate_scores,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Aggregate scores',
	    text: list_tool_names.list_tools
	};

	var data = [ trace1, trace2 ];

	var layout = {
	    xaxis: {
	        range: [ -5, optimal_scores.length + 10 ]
	    },
	    yaxis: {
	        range: [ -2.5, 10 ]
	    },
	    title:'Scatter plot of optimal and aggregate scores for tool: ' + selectedToolId
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

