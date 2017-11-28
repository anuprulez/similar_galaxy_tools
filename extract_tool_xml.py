import os
import pandas as pd
import xml.etree.ElementTree as et


class ExtractToolXML:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.repo_path = '../../galaxy/galaxy/tools/' # path to the directory containing all tools
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
                    all_tools.append( dataframe )
        tools_dataframe = pd.DataFrame( all_tools )
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname + '/data')
        tools_dataframe.to_csv( 'all_tools.csv' )
   
    @classmethod
    def convert_xml_dataframe( self, xml_file_path ):
        """
        Convert xml file of a tool to a record with its attributes
        """
        record = dict()
        tree = et.parse( xml_file_path )
        root = tree.getroot()
        record[ "id" ] = root.get( "id" )
        record[ "name" ] = root.get( "name" )
        for child in root:
            if child.tag == 'description':
                record[ child.tag ] = child.text if child.text else ""
            elif child.tag == 'inputs':
                input_formats = list()
                for item in child:
                    input_format = item.get( "format" )
                    if input_format not in input_formats and input_format is not None:
                        input_formats.append( input_format )
                record[ child.tag ] = '' if input_formats is None else ",".join( input_formats )
        return record



if __name__ == "__main__":
    extract_tool = ExtractToolXML()
    extract_tool.read_tool_directory()
