"""
Extract attributes and their values from the xml files of all tools
"""

import os
import urllib2
import pandas as pd
import xml.etree.ElementTree as et
import utils
import json

class ExtractToolXML:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.repo_path = 'data/alltools/' # path to the directory containing all tools
        self.tools_directories = [ name for name in os.listdir( self.repo_path ) ]
        self.file_extension = 'xml'

    @classmethod
    def read_tool_directory( self ):
        """
        Loop through all the directories in the Galaxy tools folder
        """
        all_tools = list()
        for item in self.tools_directories:
            tool_path = os.path.join( self.repo_path, item )
            files = os.listdir( tool_path )
            for file in files:
                if file.endswith( self.file_extension ):
                    dataframe = self.convert_xml_dataframe( os.path.join( tool_path, file ) )
                    if dataframe is None:
                        continue
                    else:
                        # remove any duplication of tool ids
                        if not any( item.get( "id", None ) == dataframe[ "id" ] for item in all_tools ):
                            all_tools.append( dataframe )
        tools_dataframe = pd.DataFrame( all_tools )
        dname = os.path.dirname( os.path.abspath(__file__) )
        os.chdir(dname + '/data')
        tools_dataframe.to_csv( 'all_tools.csv' )
   
    @classmethod
    def convert_xml_dataframe( self, xml_file_path ):
        """
        Convert xml file of a tool to a record with its attributes
        """
        record = dict()
        try:
            tree = et.parse( xml_file_path )
            root = tree.getroot()
            record_id = root.get( "id", None )
            # read those xml only if it is a tool
            if root.tag == "tool" and record_id is not None and record_id is not '':
                record[ "id" ] = record_id
                record[ "name" ] = root.get( "name" )
                for child in root:
                    if child.tag == 'description':
                        record[ child.tag ] = str( child.text ) if child.text else ""
                    elif child.tag == 'inputs' or child.tag == 'outputs':
                        file_formats = list()
                        for item in child:
                            file_format = item.get( "format", None )
                            if file_format not in file_formats and file_format is not None:
                                file_formats.append( file_format )
                        record[ child.tag ] = '' if file_formats is None else ",".join( file_formats )
                    elif child.tag == 'help':
                        help_text = child.text
                        help_split = help_text.split('\n\n')
                        for index, item in enumerate( help_split ):
                            if 'What it does' in item or 'Syntax' in item:
                                record[ child.tag ] =  utils._remove_special_chars( help_split[ index + 1 ] )
                                break
                    elif child.tag == "edam_topics":
                        for item in child:
                           if item.tag == "edam_topic":
                              edam_annotations = ''
                              edam_text = urllib2.urlopen( 'https://www.ebi.ac.uk/ols/api/ontologies/edam/terms?iri=http://edamontology.org/' + item.text )
                              edam_json = json.loads( edam_text.read() )
                              edam_terms = edam_json[ "_embedded" ][ "terms" ][ 0 ]
                              edam_annotations += edam_terms[ "description" ][ 0 ] + ' '
                              edam_annotations += edam_terms[ "label" ] + ' '
                              for syn in edam_terms[ "synonyms" ]:
                                  edam_annotations +=  syn + ' '
                              record[ child.tag ] = utils._remove_special_chars( edam_annotations )
                return record
        except Exception as exp:
           return None


if __name__ == "__main__":
    extract_tool = ExtractToolXML()
    extract_tool.read_tool_directory()
