# Prediction of similar tools for Galaxy tools

It lists a set of similar tools for the selected tool computed using BM25 and Gradient Descent.
For predicting similar tools, the following attributes of tools have been considered:

- Name
- Description
- Input and output file types
- What it does
- EDAM

Since all these sources of annotations/text are not informative in a balanced way (for example input and output file types information could be more informative to find similar tools compared to the description). To accommodate this, we have separated the sources of information and categorized them into three sources - Input and output file types, name and description and help text ( and EDAM annotations). For each source, we compute similar tools for each tool using BestMatch25 (BM25) algorithm. As a result, we have three N x N ( tool x tools ) similarity matrices. Now the question arises how to combine these matrices to get the best results. A naive way is to take an average giving uniform prior to all the sources.

To better the results, we need to optimize the mixing of these matrices. To achieve this, we use Gradient Optimizer ( or a variant) to learn the importance ( or weights) for each of these sources. After learning the weights, we compute weighted similarity matrices and create an output as JSON file. To display the JSON file and see similar tools for any tool, there is an HTML file in `/viz` folder. It lists a set of similar tools differentiated by a column called "similarity score". The value varies between 1 and 0 where 1 is the most similar and 0 is the most dissimilar.

# How to use

Please visit `/viz` folder and open the html file "similarity_viz.html" in your browser. When the page opens, it lists all the tools in a select box. Please select any one and see the similar tools for the selected one. If there are no similar tool(s), an error message will appear.
