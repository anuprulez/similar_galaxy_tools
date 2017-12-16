"""
Extract attributes and their values from the xml files of all tools
"""

import os
#import urllib2
import pandas as pd
import xml.etree.ElementTree as et
import utils
import json
import base64
import requests
import time


class ExtractToolXML:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.file_extension = 'xml'
        self.auth = ('anuprulez', '*******************')
        
    @classmethod
    def read_tool_xml( self, data_source_config ):
        tree = et.parse( data_source_config )
        url = 'https://api.github.com/repos/'
        root = tree.getroot()
        processed_tools = list()
        start_time = time.time()
        if root.tag == "datasources":
            for child in root:
                repo_path = child.get( "path" )
                tool_source = ''
                tool_collection_source = ''
                tool_source = url + repo_path + 'contents/tools'
                print "Reading repository: %s" % repo_path
                
                # read tools
                tools_request = self.make_get_requests( tool_source )
                tools = self.read_tool_dir( tools_request )
                for item in tools:
                    processed_tools.append( item )
                
                # read tool collections
                tool_collection_source = url + repo_path + 'contents/tool_collections'
                tool_collection_request = self.make_get_requests( tool_collection_source )
                for tool_collection in tool_collection_request:
                    print tool_collection[ "url" ]
                    collection = self.make_get_requests( tool_collection[ "url" ] )
                    if collection is not None:
                        tools_collect = self.read_tool_dir( collection )
                        if tools_collect is not None:
                            for item in tools_collect:
                                processed_tools.append( item )
       
        tools_dataframe = pd.DataFrame( processed_tools )
        dname = os.path.dirname( os.path.abspath(__file__) )
        os.chdir(dname + '/data')
        tools_dataframe.to_csv( 'processed_tools.csv' )
        end_time = time.time()
        print "%d tools read in %d seconds" % ( len( processed_tools ), int( end_time - start_time ) )

    @classmethod
    def read_tool_dir( self, all_tools ):
        file_extension = '.xml'
        tools = list()
        for tool in all_tools:
            if tool[ "type" ] == "dir":
                files = self.make_get_requests( tool[ "url" ] )
                for item in files:
                    name = item[ "name" ]
                    if name.endswith( file_extension ) and item[ "type" ] == "file":
                        tool = self.read_tool_xml_file( item )
                        if tool is not None:
                            tools.append( tool )
                            print "Added tool: %s" % item[ "name" ]
            elif tool[ "type" ] == "file":
                name = tool[ "name" ]
                if name.endswith( file_extension ):
                    tool_item = self.read_tool_xml_file( tool )
                    if tool_item is not None:
                        tools.append( tool )
                        print "Added tool: %s" % name
        return tools
    
             
    @classmethod
    def read_tool_xml_file( self, item ):
        file = self.make_get_requests( item[ "url" ] )       
        file_content = base64.b64decode( file[ "content" ] )
        return self.convert_xml_dataframe( file_content )
            
    @classmethod
    def make_get_requests( self, path ):
        try:
            request = requests.get( path, auth=self.auth )
            if request.status_code == requests.codes.ok:
                return request.json()
        except Exception as exception:
            print "Error in make get requests: %s" % exception
   
    @classmethod
    def convert_xml_dataframe( self, xml_file_path ):
        """
        Convert xml file of a tool to a record with its attributes
        """
        record = dict()
        try:
            root = et.fromstring( xml_file_path )
            record_id = root.get( "id", None )
            # read those xml only if it is a tool
            if root.tag == "tool" and record_id is not None and record_id is not '':
                record[ "id" ] = record_id
                record[ "name" ] = root.get( "name" )
                for child in root:
                    if child.tag == 'description':
                        record[ child.tag ] = str( child.text ) if child.text else ""
                    elif child.tag == 'inputs':
                        file_formats = list()
                        for children in child.findall( './/param' ):
                            if children.get( 'type' ) == 'data':
                                file_format = children.get( 'format', None )
                                if file_format not in file_formats and file_format is not None:
                                    file_formats.append( file_format )
                        
                        record[ child.tag ] = '' if file_formats is None else ",".join( file_formats )
                    elif child.tag == 'outputs':
                        file_formats = list()
                        for children in child:
                            file_format = children.get( 'format', None )
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
            else:
                return None
        except Exception as exp:
           print "Exception in converting xml to dict"
           return None

if __name__ == "__main__":
    extract_tool = ExtractToolXML()
    extract_tool.read_tool_xml( "data_source.config" )