

$(document).ready(function(){

    var similarityData = null,
        listTools = null,
        path = "data/similarity_matrix.json";
    if ( path === "" ) {
        console.error( "Error in reading the JSON file path" );
        return;
    }
    $.getJSON( path, function( data ) {
        var toolIdsTemplate = "";
            listTools = data[ data.length - 1 ][ "list_tools" ];
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
                var sourceProbScores = toolResults.source_prob_similar_tools,
                    template = "";  
                // make html for the selected tool
                $el_tools.append( createHTML( [ toolResults.root_tool ], selectedToolId, "Selected tool: <b>" +  selectedToolId + "</b>", "", true ) );
                // make html for similar tools found using joint probability scores
                $el_tools.append( createHTML( sourceProbScores, selectedToolId, "Similar tools for the selected tool: <b>" +  selectedToolId + "</b> found using source probability score", "Source probability score", false ) );
                
                // plots for probability scores
                $el_tools.append( "<div id='io-prob-dist'></div>" );
                plotScatterData( toolResults.source_prob_dist, "io-prob-dist", "Probability scores for input output source for: " + selectedToolId, listTools );
                availableSimilarTool = true;
                break;
            }
         } // end of for loop
         if ( !availableSimilarTool ) {
             $el_tools.empty().html( "<p class='no-similar-tool-msg'>No similar tool available. <p>" );
         }
    });

    var createHTML = function( toolScores, originalToolId, headerText, scoreHeaderText, isHeader ) {
        var template = "<div class='table-header-text'>" + headerText + "</div>",
            prevRank = 0,
            prevScore = 0,
            maxShowStringLen = 30,
            digitPrecision = 5;
        template += "<div class='table-responsive'><table class='table table-bordered table-striped thead-dark'><thead>";
        template += "<th>S.No.</th>";
        template += "<th>Id</th>";
        if ( !isHeader ) {
            template += "<th> Input output name desc. edam help probability score </th>";
            template += "<th> " + scoreHeaderText + "</th>";
            template += "<th> Rank </th>";
        }
        template += "<th> Name and description </th>";
        template += "<th> Input files </th>";
        template += "<th> Output files </th>";
        template += "<th> Help text (what it does) </th>";
        template += "<th> EDAM </th>";
        template += "</thead><tbody class='table-striped'>";
        
        for( var counterTS = 0, len = toolScores.length; counterTS < len; counterTS++ ) {
            var tool = toolScores[ counterTS ],
                toolScore = tool.score.toFixed(7),
                rank = 0,
                helpText = tool.what_it_does,
                nameDesc = tool.name_description,
                showHelpText = ( helpText.length > maxShowStringLen && !isHeader ) ? helpText.substring(0, maxShowStringLen) + "..." : helpText;

            rank = ( prevScore === toolScore ) ? prevRank : parseInt( counterTS + 1 );
            template += "<tr>";
            template += "<td>" + parseInt( counterTS + 1 ) + "</td>";
            template += "<td>" + tool.id + "</td>";
            if ( !isHeader ) {
                template += "<td>" + tool.source_prob.toFixed(digitPrecision) + "</td>";
                template += "<td>" + toolScore + "</td>";
                template += "<td>" + rank  + "</td>";
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
    
    var plotScatterData = function( scores, $elPlot, plotTitle, toolNames ) {
        var xAxis = [];
        for( var i = 0, len = scores.length; i < len; i++ ) {
            xAxis.push( i + 1 );
        }
        
        var trace1 = {
	    x: xAxis,
	    y: scores,
	    mode: 'markers',
	    type: 'scatter',
	    name: 'Probability scores',
	    text: toolNames
	};

	var data = [ trace1 ];

	var layout = {
	    xaxis: {
	        title: "Tools"
	    },
	    yaxis: {
	        title: "Probability"
	    },
	    title: plotTitle
	};
	Plotly.newPlot( $elPlot, data, layout );
    };   
});

