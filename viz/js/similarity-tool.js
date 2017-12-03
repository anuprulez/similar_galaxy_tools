

$(document).ready(function(){

    var similarityData = null;
    $.getJSON( "data/similarity_matrix.json", function( data ) {
        var toolIdsTemplate = "";

        // sort the tools in ascending order of their ids
        similarityData = data.sort(function(a, b) {
            var first_id = a.id.toLowerCase(),
                second_id = b.id.toLowerCase();
            if( first_id < second_id ) { return -1; }
            if( first_id > second_id ) { return 1; }
            return 0;
        });
        for( var counter = 0, len = similarityData.length; counter < len; counter++ ) {
            var toolResults = similarityData[ counter ];
            toolIdsTemplate += "<option value=" + toolResults.id + ">" + toolResults.id + "</options>";
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
            if ( toolResults.id === selectedToolId ) { 
                var toolScores = toolResults.scores,
                    template = "";
                toolScores = toolScores.sort(function( a, b ) {
                    return parseFloat( b.score ) - parseFloat( a.score );
                });
                template = "<table><thead>";
                template += "<th>Id</th>";
                template += "<th> Similarity score </th>";
                template += "<th> Name and Description </th>";
                template += "<th> Input types </th>";
                template += "<th> Output types </th>";
                //template += "<th> What it does </th>";
                template += "</thead><tbody>";
                for( var counter_ts = 0, len_ts = toolScores.length; counter_ts < len_ts; counter_ts++ ) {
                    var tool = toolScores[ counter_ts ];
                    template += "<tr>";
                    template += "<td>" + tool.id + "</td>";
                    template += "<td>" + tool.score + "</td>";
                    template += "<td>" + tool.name_description + "</td>";
                    template += "<td>" + tool.input_types + "</td>";
                    template += "<td>" + tool.output_types + "</td>";
                    //template += "<td>" + tool.what_it_does + "</td>";
                    template += "</tr>";
                }
                template += "</tbody></table>";
                $el_tools.html( template );
                availableSimilarTool = true;
                break;
            }
         } // end of for loop
         if ( !availableSimilarTool ) {
             $el_tools.empty().html( "<p class='no-similar-tool-msg'>No similar tool available. <p>" );
         }
    });
});

