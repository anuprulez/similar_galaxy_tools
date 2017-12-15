

$(document).ready(function(){

    var similarityData = null;
    $.getJSON( "data/similarity_matrix.json", function( data ) {
        var toolIdsTemplate = "";
        // sort the tools in ascending order of their ids
        similarityData = data.sort(function(a, b) {
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
                toolIdsTemplate += "<option value=" + toolResults.root_tool.id + ">" + toolResults.root_tool.id + "</options>";
            }
        } // end of for loop
        $( ".tool-ids" ).append( toolIdsTemplate );
    });


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
                $el_tools.append( createHTML( [ toolResults.root_tool ], selectedToolId, "<h4>Selected tool: " +  selectedToolId + "</h4>" ) );
                // make html for similar tools
                $el_tools.append( createHTML( toolScores, selectedToolId, "<h4>Similar tools for the selected tool: " +  selectedToolId + "</h4>" ) );
                availableSimilarTool = true;
                break;
            }
         } // end of for loop
         if ( !availableSimilarTool ) {
             $el_tools.empty().html( "<p class='no-similar-tool-msg'>No similar tool available. <p>" );
         }
    });

    var createHTML = function( toolScores, originalToolId, headerText ) {
        var template = headerText;
        template += "<table><thead>";
        template += "<th>Id</th>";
        template += "<th> Similarity score </th>";
        template += "<th> Name and description </th>";
        template += "<th> Input files </th>";
        template += "<th> Output files </th>";
        template += "<th> Help text (what it does) </th>";
        template += "<th> EDAM </th>";
        template += "</thead><tbody>";
        for( var counter_ts = 0, len_ts = toolScores.length; counter_ts < len_ts; counter_ts++ ) {
            var tool = toolScores[ counter_ts ];
            template += "<tr>";
            template += "<td>" + tool.id + "</td>";
            template += "<td>" + tool.score + "</td>";
            template += "<td>" + tool.name_description + "</td>";
            template += "<td>" + tool.input_types + "</td>";
            template += "<td>" + tool.output_types + "</td>";
            template += "<td>" + tool.what_it_does + "</td>";
            template += "<td>" + tool.edam_text + "</td>";
            template += "</tr>";
        }
        template += "</tbody></table>";
        return template;
    }
});

