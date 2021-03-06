$(document).ready(function() {
    var similarityData = null,
        list_tool_names = null,
        //pathLocal = "data/similarity_matrix.json",
        pathOnline = "https://raw.githubusercontent.com/anuprulez/similar_galaxy_tools/doc2vec/viz/data/similarity_matrix.json",
        path = pathOnline,
        $elLoader = $( ".loader-place" );
    if ( path === "" ) {
        console.error( "Error in loading JSON file" );
        return;
    }
    $elLoader.show();
    $.getJSON( path, function( data ) {
        console.log(data);
        var toolIdsTemplate = "";
        $elLoader.hide();
        list_tool_names = data[ data.length - 1 ];
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
                    template = "";
                // make html for the selected tool
                $el_tools.append( createHTML( [ toolResults.root_tool ], selectedToolId, "Selected tool: <b>" +  selectedToolId + "</b>", "", true ) );
                // show optimal weights
                $el_tools.append( showWeights( toolResults.optimal_weights, "" ) );
                
                // make html for similar tools found by optimizing similarity scores using Gradient Descent
                $el_tools.append( createHTML( toolScores, selectedToolId, "Similar tools for the selected tool: <b>" +  selectedToolId + " </b>found by optimal combination (Gradient Descent) of similarity scores from multiple sources</h4>", "Weighted similarity score", false ) );
                
                // plot optimal vs average scores
                $el_tools.append( "<div id='scatter-optimal-average'></div>" );
                plotScatterOptimalAverageScores( toolResults, "scatter-optimal-average", selectedToolId );
 
                $el_tools.append( "<div id='tool-combined-gradient-iterations'></div>" );
                plotCombinedGradients( toolResults.combined_gradients, 'tool-combined-gradient-iterations', selectedToolId );

                // plot loss drop vs iterations
                $el_tools.append( "<div id='tool-cost-iterations'></div>" );
                plotCostVsIterations( toolResults, "tool-cost-iterations", selectedToolId );

                $el_tools.append( "<div id='learning-rate-iterations'></div>" );
                plotLearningRatesVsIterations( toolResults, "learning-rate-iterations", selectedToolId );

                /*$el_tools.append( "<div id='mutual-similarity-io-nd'></div>" );
                $el_tools.append( "<div id='mutual-similarity-nd-ht'></div>" );
                $el_tools.append( "<div id='mutual-similarity-io-ht'></div>" );
                plotMutualSimilarity( toolScores, "mutual-similarity-io-nd", "mutual-similarity-nd-ht", "mutual-similarity-io-ht" );

                $el_tools.append( "<div id='io-nd-scores'></div>" );
                $el_tools.append( "<div id='nd-ht-scores'></div>" );
                $el_tools.append( "<div id='io-ht-scores'></div>" );
                getMutualScores( toolResults, "io-nd-scores", "nd-ht-scores", "io-ht-scores", selectedToolId );*/
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
                template += "<div>" + "Input and output file types ( weight1 ): <b>" + toPrecisionNumber( weights[ item ] ) + "</b></div>";
            }
            else if( item === "name_desc_edam" ) {
                template += "<div>" + "Name, description and EDAM ( weight2 ): <b>" + toPrecisionNumber( weights[ item ] )  + "</b></div>";
            }
            else if( item === "help_text" ) {
                template += "<div>" + "Help text ( weight3 ): <b>" + toPrecisionNumber( weights[ item ] )  + "</b></div>";
            }
        }
        template += "<p>Score = weight1 * similarity_input_output + weight2 * similarity_name_desc_edam + weight3 * similarity_help_text </p>";
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
            template += "<th> Input and output </th>";
            template += "<th> Name and description  </th>";
            template += "<th> Help text  </th>";
            template += "<th> " + scoreHeaderText + "</th>";
            template += "<th> Rank </th>";
        }
        template += "<th> Name and description </th>";
        template += "<th> Input files </th>";
        template += "<th> Output files </th>";
        template += "<th> Help text </th>";
        template += "<th> EDAM </th>";
        template += "</thead><tbody>";
        
        for( var counter = 0, len = toolScores.length; counter < len; counter++ ) {
            var tool = toolScores[ counter ],
                toolScore = tool.score,
                rank = 0,
                helpText = tool.what_it_does,
                nameDesc = tool.name_description,
                showHelpText = ( helpText.length > maxShowStringLen && !isHeader ) ? helpText.substring( 0, maxShowStringLen ) + "..." : helpText;

            rank = ( prevScore === toolScore ) ? prevRank : parseInt( counter + 1 );
            template += "<tr>";
            template += "<td>" + parseInt( counter + 1 ) + "</td>";
            template += "<td>" + tool.id + "</td>";
            if ( !isHeader ) {
                template += "<td>" + toPrecisionNumber( tool.input_output_score ) + "</td>";
                template += "<td>" + toPrecisionNumber( tool.name_desc_edam_score ) + "</td>";
                template += "<td>" + toPrecisionNumber( tool.help_text_score ) + "</td>";
                template += "<td>" + toPrecisionNumber( toolScore ) + "</td>";
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

    var plotCombinedGradients = function( combinedGradients, $elPlot, selectedToolId ) {
        var iterations = combinedGradients.length,
            xAxis = [];
        for( var i = 0; i < iterations; i++ ) {
            xAxis.push( i + 1 );
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
    
    var plotCostVsIterations = function( toolScores, $elPlot, selectedToolId ) {
        var costIterations = toolScores.cost_iterations,
            iterations = costIterations.length,
            costUniformTools = [],
            x_axis = [];
        for( var i = 0; i < iterations; i++ ) {
            x_axis.push( i + 1 );
            costUniformTools.push( toolScores.uniform_cost_tools );
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

    var plotMutualSimilarity = function( scores, $el1, $el2, $el3 ) {
        var ioScores = [],
            ndScores = [],
            htScores = [];
        for( var counter = 0, len = scores.length; counter < len; counter++ ) {
            var tool = scores[ counter ];
            ioScores.push( tool.input_output_score );
            ndScores.push( tool.name_desc_edam_score );
            htScores.push( tool.help_text_score );
        }

        // Input/output and name desc.
        var trace1 = {
	    x: ioScores,
	    y: ndScores,
	    mode: 'markers',
	    type: 'scatter'
	};

	var layout1 = {
	    xaxis: {
	        title: "Input/output similarity scores"
	    },
	    yaxis: {
	        title: "Name/desc similarity scores"
	    },
	    title: "Scatter plot for input/output and name/desc similarity scores"
	};
	Plotly.newPlot( $el1,[ trace1 ], layout1 );


        // Name desc and help text
        var trace2 = {
	    x: ndScores,
	    y: htScores,
	    mode: 'markers',
	    type: 'scatter'
	};

	var layout2 = {
	    xaxis: {
	        title: "Name/desc similarity scores"
	    },
	    yaxis: {
	        title: "Help text similarity scores"
	    },
	    title: "Scatter plot for name/desc and help text similarity scores"
	};
	Plotly.newPlot( $el2,[ trace2 ], layout2 );


        // IO and helptext
        var trace3 = {
	    x: ioScores,
	    y: htScores,
	    mode: 'markers',
	    type: 'scatter'
	};

	var layout3 = {
	    xaxis: {
	        title: "Input/output similarity scores"
	    },
	    yaxis: {
	        title: "Help text similarity scores"
	    },
	    title: "Scatter plot for input/output and help text similarity scores"
	};
	Plotly.newPlot( $el3,[ trace3 ], layout3 );
    };

    

    var getMutualScores = function( scores, $elIOND, $elNDHT, $elIOHT, selectedToolId ) {
        let ioScores = [],
            ndScores = [],
            htScores = [],
            xAxis = [];
        for( var counter = 0, len = scores.io_ht_corr.length; counter < len; counter++ ) {
            xAxis.push( counter + 1 );
        }

        plotMutualScatterPlot( scores.io_nd_corr, $elIOND, "Correlation between input/output and name/desc similarities", xAxis );
        plotMutualScatterPlot( scores.nd_ht_corr, $elNDHT, "Correlation between name/desc and help text similarities", xAxis );
        plotMutualScatterPlot( scores.io_ht_corr, $elIOHT, "Correlation between input/output and help text similarities", xAxis );
    };

    var plotMutualScatterPlot = function( scores, $el, legend, xAxis ) {
        var trace1 = {
	    x: xAxis,
	    y: scores,
	    mode: 'markers',
	    type: 'scatter'
	};

	var data = [ trace1 ];
	var layout = {
	    xaxis: {
	        title: "Tools"
	    },
	    yaxis: {
	        title: "Correlation scores"
	    },
	    title: legend
	};
	Plotly.newPlot( $el, data, layout );
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
	    name: 'Optimal similarity scores',
	    text: list_tool_names.list_tools
	};

	var trace2 = {
	    x: x_axis,
	    y: average_scores,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Average similarity scores',
	    text: list_tool_names.list_tools
	};

	var data = [ trace1, trace2 ];
	var layout = {
	    xaxis: {
	        title: "Tools"
	    },
	    yaxis: {
	        title: "Similarity score"
	    },
	    title:'Scatter plot of optimal and average combination of similarity scores for tool: ' + selectedToolId
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

