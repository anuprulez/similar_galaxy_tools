# Prediction of similar tools for Galaxy tools

It lists a set of similar tools for the selected tool computed using BM25 and Gradient Descent.
For predicting similar tools, the following attributes of tools have been considered:

- Name
- Description
- Input and output file types
- What it does
- EDAM

The tokens/ annotations have been stemmed using Natural Language Toolkit. Also for a token to be the representative of a tool, it needs to be at least 3 characters.

Since all these sources of annotations/text are not informative in a balanced way (for example input and output file types information could be more informative to find similar tools compared to the description). To accommodate this, we have separated the sources of information and categorized them into three sources - Input and output file types, name and description and help text ( and EDAM annotations). For each source, we compute similar tools for each tool using BestMatch25 (BM25) algorithm. As a result, we have three N x N ( tool x tools ) similarity matrices. Now the question arises how to combine these matrices to get the best results. A naive way is to take an average giving uniform prior to all the sources.

To better the results, we need to optimize the mixing of these matrices. To achieve this, we use Gradient Optimizer ( or a variant) to learn the importance ( or weights) for each of these sources. After learning the weights, we compute weighted similarity matrices and create an output as JSON file. To display the JSON file and see similar tools for any tool, there is an HTML file in `/viz` folder. It lists a set of similar tools differentiated by a column called "similarity score". The value varies between 1 and 0 where 1 is the most similar and 0 is the most dissimilar.

# How to use

## Extract data

In the repository, there is an script `extract_tool_github.py` to read all the 'xml' files of tools (sources of all these tools are defined in `data_source.config` file) and condense them as tabular data. In order to run this script, the GitHub authentication is needed.

Execute this script as `python extract_tool_github.py "username:password"`

When this script finishes successfully, a file named `processed_tools.csv` is created at `/data` folder. 

## Find similar tools

Further, run `predict_similarity` script to find similar tools for each tool. Run this script using following command:

`python predict_similarity.py <path to csv file> <number_of_iterations>`. 

The number of iterations should be a positive number (say 50). The number of iterations and learning rate (0.7) should be carefully selected so that good results are obtained in reasonable amount of time. The learning rate decays with time (number of iterations). When this script finishes, a couple of plots are generated to show the difference between random and learned weights, drop in the cost with iterations and decay of learning rate. It creates a JSON file with all the data. From the `/data` folder, copy the file `similarity_matrix.json` and paste here '/viz/data'. Please visit `/viz` folder and open the html file "similarity_viz.html" in your browser. When the page opens, it lists all the tools in a select box. Please select any one and see the similar tools for the selected one. If there are no similar tool(s), an error message will appear. Along with the similar tools, it also shows the initial weights and learned weights for the sources and a plot of drop in cost with iterations for this selected tool.
