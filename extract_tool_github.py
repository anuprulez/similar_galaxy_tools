"""
Extract attributes and their values from the xml files of all tools.
It pulls tools' files from GitHub
"""
import re
import sys
import os
import pandas as pd
import xml.etree.ElementTree as et
import utils
import json
import base64
import requests
import time
import urllib2


class ExtractToolXML:

    @classmethod
    def __init__( self, auth ):
        """ Init method. """
        self.file_extension = '.xml'
        self.base_url = 'https://api.github.com/repos/'
        self.directory = '/data'
        self.tool_data_filename = 'processed_tools4.csv'
        # please supply your GitHub's username and password to authenticate yourself
        # in order to be able to read files
        self.auth = auth

    @classmethod
    def read_tool_xml( self, data_source_config ):
        """
        Read the directories for tools and tool collections on GitHub
        """
        start_time = time.time()
        processed_tools = list()
        tools_url_suffix = 'contents/tools'
        tool_collections_url_suffix = 'contents/tool_collections'
        tree = et.parse( data_source_config )
        root = tree.getroot()
        if root.tag == "datasources":
            for child in root:
                repo_path = child.get( "path" )
                tool_source = ''
                tool_collection_source = ''
                tool_source = self.base_url + repo_path + tools_url_suffix
                print "Reading repository: %s" % repo_path

                # read tools
                print "Reading tools from: %s" % tools_url_suffix
                tools_request = self.make_get_requests( tool_source )
                tools = self.read_tool_dir( tools_request )
                if tools is not None:
                    for item in tools:
                        processed_tools.append( item )

                # read tool collections
                tool_collection_source = self.base_url + repo_path + tool_collections_url_suffix
                print "Reading tool collections from: %s" % tool_collection_source
                tool_collection_request = self.make_get_requests( tool_collection_source )
                if tool_collection_request is not None:
                    for tool_collection in tool_collection_request:
                        collection = self.make_get_requests( tool_collection[ "url" ] )
                        if collection is not None:
                            tools_collect = self.read_tool_dir( collection )
                            if tools_collect is not None:
                                for item in tools_collect:
                                    # avoid duplications of tools if any
                                    if not any( item[ "id" ] == tool[ "id" ] for tool in processed_tools ):
                                        processed_tools.append( item )

        tools_dataframe = pd.DataFrame( processed_tools )
        dname = os.path.dirname( os.path.abspath(__file__) )
        file_dir = dname + self.directory
        if not os.path.exists( file_dir ):
            os.makedirs( file_dir )
        os.chdir( file_dir )
        tools_dataframe.to_csv( self.tool_data_filename, encoding='utf-8' )
        end_time = time.time()
        print "%d tools read in %d seconds" % ( len( processed_tools ), int( end_time - start_time ) )

    @classmethod
    def read_tool_dir( self, all_tools ):
        """
        Read the tool directories
        """
        tools = list()
        for tool in all_tools:
            # if the tools are in a directory
            if tool[ "type" ] == "dir":
                files = self.make_get_requests( tool[ "url" ] )
                for item in files:
                    name = item[ "name" ]
                    if name.endswith( self.file_extension ) and item[ "type" ] == "file":
                        tool = self.read_tool_xml_file( item )
                        if tool is not None:
                            tools.append( tool )
                            print "Added tool: %s" % item[ "name" ]
            elif tool[ "type" ] == "file":
                name = tool[ "name" ]
                if name.endswith( self.file_extension ):
                    tool_item = self.read_tool_xml_file( tool )
                    if tool_item is not None:
                        tools.append( tool_item )
                        print "Added tool: %s" % name
        return tools

    @classmethod
    def read_tool_xml_file( self, item ):
        """
        Read a tool's xml file and convert it to text
        """
        file = self.make_get_requests( item[ "url" ] )
        file_content = base64.b64decode( file[ "content" ] )
        return self.convert_xml_dataframe( file_content )

    @classmethod
    def make_get_requests( self, path ):
        """
        Make GET requests to GitHub to fetch content of files
        """
        try:
            auth = self.auth.split( ":" )
            request = requests.get( path, auth=( auth[ 0 ], auth[ 1 ] ) )
            if request.status_code == requests.codes.ok:
                return request.json()
        except Exception as exception:
            print "Error in making get requests: %s" % exception

    @classmethod
    def clear_urls( self, text ):
        """
        Clear the URLs from the text
        """ 
        list_urls = re.findall( 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text )
        if len( list_urls ) > 0:
            for url in list_urls:
                text = text.replace( url, '' )
        return text

    @classmethod
    def convert_xml_dataframe( self, xml_file_path ):
        """
        Convert xml file of a tool to a record with its attributes
        """
        record = dict()
        try:
            # read as xml from content string
            root = et.fromstring( xml_file_path )
            record_id = root.get( "id", None )
            # read those xml only if it is a tool
            if root.tag == "tool" and record_id is not None and record_id is not '':
                # print xml_file_path
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
                        '''help_text = child.text
                        help_split = help_text.split( '\n\n' )
                        for index, item in enumerate( help_split ):
                            if 'What it does' in item or 'Syntax' in item:
                                hlp_txt = help_split[ index + 1 ]
                                list_urls = re.findall( 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', hlp_txt )
                                if len( list_urls ) > 0:
                                    for url in list_urls:
                                        hlp_txt = hlp_txt.replace( url, '' )
                                record[ child.tag ] = utils._remove_special_chars( hlp_txt )
                                break'''
                        clean_helptext = ''
                        help_text = child.text
                        help_split = help_text.split( '\n\n' )
                        for index, item in enumerate( help_split ):
                            if 'What it does' in item:
                                hlp_txt = help_split[ index + 1 ]
                                clean_helptext = hlp_txt
                        # if these categories inside help text are not present, then add complete help text
                        if clean_helptext == "":
                            clean_helptext = child.text
                        helptext_lines = clean_helptext.split( "." )
                        helptext_lines = helptext_lines[ :2 ]
                        clean_helptext = " ".join( helptext_lines )
                        clean_helptext = self.clear_urls( clean_helptext )
                        clean_helptext = utils._remove_special_chars( clean_helptext )
                        print "=========================================="
                        print clean_helptext
                        
                        record[ child.tag ] = clean_helptext
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
        except Exception:
            print "Exception in converting xml to dict"
            return None


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print( "Usage: python extract_tool_github.py 'username:password'" )
        exit( 1 )
    extract_tool = ExtractToolXML( sys.argv[ 1 ] )
    extract_tool.read_tool_xml( "data_source.config" )
