General Function Workflow:

1) Get sentence data in Libby's cleaned formats, as seen in Mandarin_Sentences.txt, English_Sentences.txt
	Column names are hardcoded in the cleaning code for further processing, so make sure they match in future sentence dumps

2) Take raw sentences with linguistic tags and turn them into regular sentences with a single masked preposition per sentence	
Sentence Cleaning  	(my_cleaning.py/sentence_cleaning)
Prep Masking		(my_cleaning.py/sentence_masking)

2a) Prep Masking has the option to either give you all the sentences, or to specify a single prep and only return sentences containing that single prep

3) Passing Sentence and Correct Prep to a specified Masked Language Model
testBERT.py/BERT_predictions
XLMRoberta_tests.py/XLM_Roberta_predictions

BERT and XLM_Roberta have different masking styles and sentence limiters, but outside of calling sentence_masking with model='xlm' and using code from XLMRoberta_tests.py, there is no difference in how the data is returned if you need to switch models.

4) Turning BERT Output into Confusion Maps
prep_filtering_functions.py/single_prep_outputs_BERT
	This is the main driver for taking a single text file in proper Libby Column (tm) form and turning it into raw data
prep_filtering_functions.py/make_heatmap
	Takes the raw data from single_prep_outputs_BERT and formats it into a readable heatmap, with options
	to display coloring based on raw numbers or scaled percentages

How to Create Heatmaps:

1) Gather sentence data in LibbyColumn format

2) Set up conda env

conda env create --name BERTEnv37 -f=BERTEnv37.yml

3) Modify prep_filtering_functions.py to choose which language you wish to create confusion maps for
	Uncomment out whatever you need, this is a highly efficient operation
	
4) Run berttest shell script (AMAREL instructions for SLURM)

sbatch berttest.sh

5) Outputs

Heatmaps will be png files in your current working directory

slurm.volta002.(jobID).out:
	This results file will have all the specific per prep data on what each model did on each prep.

Example Outputs:
----------------------------------	
Using --> bert-base-uncased <--
	BERT predictions finished in: 155.23 sec(s) 
	4122 correct predictions out of 6919--> 59.58% Correct
There were 0 sentences which went over allowed prediction time.
This data was only for >> on << as the masked prep

Using --> bert-base-multilingual-uncased <--
	BERT predictions finished in: 218.51 sec(s) 
	3649 correct predictions out of 6919--> 52.74% Correct
There were 0 sentences which went over allowed prediction time.
This data was only for >> on << as the masked prep

Using --> xlm-roberta-base <--
	BERT predictions finished in: 657.77 sec(s) 
	837 correct predictions out of 6919--> 12.10% Correct
There were 40 sentences which went over allowed prediction time.
This data was only for >> on << as the masked prep
------------------------------------


A large number of text files will also be created in the working directory:
	sent_progress_markers: Lets you know how long it takes to process every 25% of the input data, allow for timescale comparisons
	problem_sentences:  Will contain any sentence which was not solved in under 1 second (time variable, see testBERT code)
	correct_sentences_(language, model name): Contains all sentences with correct predictions, including sentenceID from original text file
	(language)_truncated: List of all unaltered sentences used after being passed through the prep filters (ie, all sentences using only the preps you chose)
	(language)_(model_name): Raw data from before heatmap creation, to allow for other visualizaton methods if needed