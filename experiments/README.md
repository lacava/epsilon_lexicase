# Experiment files 

This directory contains the code necessary to replicate the experiments in the paper. 

**code**: The python scripts in code can be used with an LSF job scheduler to re-run the algorithm
comparisons. The GP runs use [ellyn](http://lacava.github.io/ellyn). 

**data**: The data folder contains the results of the experiment. 

**ep-lex-benchmark.Rmd** contains the processing for generating the figures and tables in the paper.
This R markdown file generates an html output contained in **ep-lex-benchmark.html**. 

The probability analysis of lexicase selection is demonstrated in Jupyter notebooks
**selection_probability_analysis.ipynb** and **effect_of_N_and_T_on_probability.ipynb**. 



