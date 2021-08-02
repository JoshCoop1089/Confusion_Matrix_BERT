# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 10:42:05 2021

@author: joshc

Josh's code for cleaning and masking prepositions in sentences for BERT

"""
import time
import pandas as pd

pd.set_option('max_columns', None)
pd.set_option('max_colwidth', 50)

def sentence_cleaning(filename, prep_column_name, sentence_column_name):
    """
    Stage One of the cleaning process:
        Taking raw data from text files and removing duplicated sentences,
        problematic characters, and single word sentences.
        
    Parameters
    ----------
    filename : string
        The specific raw datafile to import and clean
    prep_column_name : string
        Which preposition we're using as our prediction (this is specifying the language)
    sentence_column_name : string
        Which sentences we're masking to use as BERT masked prediction bases'

    Returns
    -------
    check3 : DataFrame
        Data containing cleaner version of imported text file, with specific 
            preps for prediction identified
    prep_set : List of strings
        The set of unique prepositions present in the original data
    """
    sentence = pd.read_csv(filename, sep = "\t", error_bad_lines=False)
    # print(sentence)
    start_t = time.perf_counter()
    check = sentence.loc[:,[prep_column_name, sentence_column_name]]
    # print(check)    
    
    # English_Sentences.txt doesn't require splitting prep column value
    prep_location = 0
    prep_splitting_symbol = " "
    
    # Mandarin_Sentences.txt requires extra work
    if "Mandarin" in filename:
        prep_location = 1
        if prep_column_name == "Prep_Mandarin":
            prep_splitting_symbol = "="
        elif prep_column_name == "Prep_original":
            prep_splitting_symbol = " "
            prep_location = 0

    # Spanish_Sentences.txt requires extra work
    if "Spanish" in filename:
        prep_location = 1
        if prep_column_name == "Prep_Spanish":
            prep_splitting_symbol = "="
        elif prep_column_name == "Prep_original":
            prep_splitting_symbol = " "
            prep_location = 0
        
            
    check["prep_split"] = check[prep_column_name].str.split(prep_splitting_symbol).str[prep_location]
    # print(check["prep_split"].value_counts())
    check.reset_index(inplace = True)
    
    # Reduce the data by duplicates, sentence size, or improper preposition choice
    check = check.drop_duplicates(subset = [prep_column_name, sentence_column_name], keep = "first")
    check["len"] = check[sentence_column_name].str.split(" ").str.len()
    check2 = check[check["len"]>1]
    # print(check2["prep_split"].value_counts(normalize = True)*100)
    # check2.dropna(inplace = True)
    check3 = check2[~check2["prep_split"].str.contains("_")]
    if "Spanish" in filename:
        check3 = check3[check3["prep_split"].str.isalpha()]
    
    prep_set = check3["prep_split"].unique().tolist()
    end_t = time.perf_counter()
    print('\nTotal Time to Clean: {0:.2f} sec(s)\n'.format(end_t-start_t))
    
    # print(f"{filename}: \n{prep_column_name}: {len(prep_set)} unique preps")
    # print(check3["prep_split"].value_counts(normalize = True)*100)
    
    print(f"---> Using {prep_column_name} prepositions and {sentence_column_name} sentences. <---")    
    print("\nNumber of sentences after:")
    print(f"\tImport from file: {len(sentence)}")
    print(f"\tRemoving duplicates: {len(check)}")
    print(f"\tRemoving single word sentences: {len(check2)}")
    print(f"\tRemoving improper preps (nonAlpha chars): {len(check3)}")
    return check3, prep_set
    
def sentence_masking(sentence_df, sentence_column_name, prep = "", model = ""):   
    """
    Stage two of cleaning process:
    Take a dataframe consisting of sentences with parts of speech still attached,
        mask the correct preposition, and return a trio of lists the masked sentence, the 
        correct prep, and the index of the sentence in the original file.

    Parameters
    ----------
    sentence_df : DataFrame
        Contains correct prep, and sentence with PoS tags still attached
    sentence_column_name : string
        The specific list of PoS tagged sentences we're using
    prep : string, optional
        Allows for filtering a dataframe by a single preposition for use in BERT.  
        Not implemented in other functions yet. The default is "".

    Returns
    -------
    masked_sents : List of strings
        All sentences with the preposition replaced by [MASK] for BERT.
    correct_preps : List of strings
        The expected answer for the masked prep
    prep_set : List of strings
        All the unique prepositions which occur in the passed in DataFrame,
        to help limit BERT's guessing options
    index_out : List of ints
        The original location of a sentence from the original datafile, for use in error checking
    """
    # Splitting up data based on specific preps to check individually if needed  
    if prep != "":
        sentence_df = sentence_df[sentence_df["prep_split"]==prep]
    
    # print(sentence_df.head())
    
    prep_set = sentence_df["prep_split"].unique().tolist()
    check_preps = sentence_df["prep_split"].tolist()
    check_sents = sentence_df[sentence_column_name].tolist()
    check_index = sentence_df["index"].tolist()

    correct_preps = []
    masked_sents = []
    index_out = []
    skipped = []
    for sent, prep, index in zip(check_sents, check_preps, check_index):
        # print(f'Predicting >{prep}<:\t{sent}')
        out = []
        sent2 = sent.split(" ")
        prep_added = False
        repeat_prep = False
        
        # Format change in Mand_sent and Spanish_sent
        if sentence_column_name != "English_Original":
            for word in sent2:
                if word == prep and not repeat_prep:
                    if model == "":
                        mask = "[MASK]"
                    elif model == "xlm":
                        mask = "<mask>"
                    out.append(mask)
                    correct_preps.append(word)
                    prep_added = True
                    repeat_prep = True
                else:
                    out.append(word)
        
        # Old formatting in the english text file
        else:
            for parts in sent2:
                part = parts.split("|")
                
                # Is it a prep, does it match what it should be, and has it been seen yet
                if part[0] == 'prep' and part[-1] == prep and not repeat_prep:
                    if model == "":
                        mask = "[MASK]"
                    elif model == "xlm":
                        mask = "<mask>"
                    out.append(mask)
                    correct_preps.append(part[-1])
                    prep_added = True
                    repeat_prep = True
                else:
                    out.append(part[-1])
                
        if prep_added:
            masked_sents.append(" ".join(out).replace("_", " "))
            index_out.append(index)
        else:
            
            # This finds sentences who don't match their predicted prep
            skipped.append((prep, sent))
        
    # print(f"\n\tTotal mismatched prep sentences: {len(skipped)}")
    # print("\nSkipped the following sentences due to mismatched preps:")
    # print("Line\tPrep\tParsed")
    # for skip in skipped:
    #     number = check[check[sentence_column_name] == skip[1]].index[0]
    #     print(f'{number+2}\t\t{skip[0]}  \t{skip[1]}')
    
    num_sentences = len(masked_sents)
    
    return masked_sents, correct_preps, prep_set, index_out, num_sentences