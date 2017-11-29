"""
Extract attributes and their values from the xml files of all tools
"""

import os
import pandas as pd
import xml.etree.ElementTree as et
import re
import utils


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
        help_header = 'What it does'
        record = dict()
        try:
            tree = et.parse( xml_file_path )
            root = tree.getroot()
            record_id = root.get( "id", None )
            if record_id is not None and record_id is not '':
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
                            if help_header in item:
                                record[ child.tag ] =  utils._remove_special_chars( help_split[ index + 1 ] )
                                break
                return record
        except Exception as exp:
           return None



if __name__ == "__main__":
    extract_tool = ExtractToolXML()
    extract_tool.read_tool_directory()
