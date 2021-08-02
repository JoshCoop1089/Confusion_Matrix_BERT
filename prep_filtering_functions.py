# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:02:51 2021

@author: joshc
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

import my_cleaning as mc
import compare_multi_to_mono as cmm
import testBERT as tb
import XLMRoberta_tests as xlmr
sns.set(style="whitegrid")
sns.set_color_codes("pastel")
sns.set(font="SimSun")

# Create the Confusion Matrix for Monolinglual and Multilingual BERT Data
def make_heatmap(datafile, typeGraph):
    """
    Given a text file containing output from single_prep_outputs_BERT, 
        form the raw data into a square heatmap showing how often the MASK 
        fill was either correct or incorrect, and what the other frequent guesses were
        
    Parameters
    ----------
    datafile : String
        Name of the datafile with the raw data
    typeGraph : String
        Whether the output graph displayed the actual number of instances, or a scaled percentage count
            Use "scaled" for scaled graph, otherwise it outputs raw numbers

    Returns
    -------
        Saves png file to disk

    """
    filename = datafile+'.txt'
    df4 = pd.read_csv(filename, index_col=0)
    indexlist = df4.index.tolist()
    indexlist.reverse()
    df5 = df4.reindex(indexlist)
    # print(df5)
    fig, ax = plt.subplots(figsize = (20,20))
    raw = ""

    if typeGraph == 'scaled':
        # Scaled Values
        for column in df5.columns.tolist():
            df5[column] = MinMaxScaler().fit_transform(np.array(df5[column]).reshape(-1,1))    
        hm = sns.heatmap(df5, cmap="YlGnBu", mask=df5.isnull(), xticklabels=True, yticklabels=True)
        raw = "\n(Scaled by sklearn MinMax Scaler on a per column basis)"
        fileOut = datafile+".png"
    
    else:
        # Raw Values
        hm = sns.heatmap(df5, annot = True, fmt = 'g', linewidths = 0.3, cmap="YlGnBu", mask=df5.isnull(), xticklabels=True, yticklabels=True)
        fileOut = datafile+"_raw.png"
        
    hm.set_xticklabels(hm.get_xticklabels(), rotation = 45, fontsize=20)
    hm.set_yticklabels(hm.get_yticklabels(), rotation = 45, fontsize=20)
    plt.title(f"{datafile} Prep Predictions"+raw, fontsize=20)
    plt.xlabel("Expected Answer", fontsize=20)
    plt.ylabel("Model Answer", fontsize=20)
    
    plt.savefig(fileOut)

def single_prep_outputs_BERT(filename, prep_column_name, sentence_column_name, 
                             bert_model_mono, cor_prep_numbers, original_lang, end_lang, df = ""):
    """
    Create the raw data to use in the heatmap.  Sends a single prep from a single language along 
        with the full set of allowed answers into BERT, and records all given outputs

    Parameters
    ----------
    filename : string
        The specific raw datafile to import and clean
    prep_column_name : string
        Which preposition we're using as our prediction (this is specifying the language)
    sentence_column_name : string
        Which sentences we're masking to use as BERT masked prediction bases'
    bert_model_mono : String
        DESCRIPTION.
    cor_prep_numbers : List of Ints
        Which preps (when ordered by frequency) to use as the set of preps in 
    original_lang : string
        Shorthand for creating result files.  What language is the text coming from
    end_lang : string
        Shorthand for creating result files.  What language was the original text translated into
    df : DataFrame, optional
        Some frame that already passed through sentence_cleaning.py. The default is "". 
        If you pass in the default, the code will treat the data from 'filename' 
        as raw sentence data that needs to be fully cleaned and organized

    Returns
    -------
    None.

    """
    
    # Taking the original data file and filtering out low frequency preps and incorrectly ID'd preps
    try:
        # You can pass in a premade dataframe using the df input (but make sure it went through mc.sentence_cleaning to set up the columns correctly)
        new_merge = df
        num_sentences = len(df)
        full_prep_set = df["prep_split"].unique()
    
    # Otherwise, you're passing in a full original text file and filtering out any sentence which doesn't have a prep in the allowed prep list
    except Exception:
        orig, preps = mc.sentence_cleaning(filename, prep_column_name, sentence_column_name)
        masked_sents, correct_preps, prep_set, index_out, num_sentences = mc.sentence_masking(orig, sentence_column_name)
        # print(f"\nUsing {original_lang} to {original_lang} original data file")
        # print(f"Number Sentences Used to Predict: {num_sentences}")
        # print(f"Unique Preps: {len(prep_set)}")
        
        # Identifying the most frequent preps (or the correctly used preps in the mandarin data) and filtering out everything else
        results = cmm.make_result_df(orig, "prep_split")
        trunc_res = results[results.index.isin(cor_prep_numbers)]
        # print(f"\nThere are now: {len(trunc_res)} unique preps left.")
        # print(f"This accounts for {trunc_res['Count'].sum()} sentences out of {num_sentences}")
        # print(trunc_res)
        
        # Forming a single dataframe only consisting of valid preps and sentences in the same characters
        new_merge = orig[orig["prep_split"].isin(trunc_res["Prep"])]
        new_merge.to_csv(f"{original_lang}_to_{end_lang}_truncated.txt", index = False)
        # # counter_examples = orig[~orig[prep_column_name].isin(trunc_res["Prep"])]
        # # print(counter_examples.head())
        
        # print(new_merge.head())
        # print(len(new_merge), len(new_merge["prep_split"].unique()))
        # print(new_merge["prep_split"].value_counts())
        full_prep_set = new_merge["prep_split"].unique()
    
    # Setting up the result dataframes to put final raw data into
    false_prep_mono = pd.DataFrame(index = full_prep_set)
    false_prep_multi = pd.DataFrame(index = full_prep_set)
    false_prep_xlmr = pd.DataFrame(index = full_prep_set)
    tot_correct_mono, tot_correct_multi, tot_correct_xlmr = 0,0,0
    tot_sentences_mono, tot_sentences_multi, tot_sentences_xlmr = 0,0,0
    
    for prep in full_prep_set:
    
        masked_sents, correct_preps, prep_set, index_out, num_sentences = mc.sentence_masking(new_merge, sentence_column_name, prep)
        
        # if num_sentences == 0:
        #     print(f"BERT Sentences for >> {prep} <<: {num_sentences}")
        bert_model_name = bert_model_mono
        c_mono, num_sentences_mono, total_time, result_text, false_results_mono = tb.BERT_predictions(masked_sents, correct_preps,
                                                                                full_prep_set, bert_model_name, num_sentences,
                                                                                index_out, original_lang, end_lang)
        tot_correct_mono += c_mono
        tot_sentences_mono += num_sentences_mono
        false_prep_mono[prep] = false_results_mono["Predicted Prep"].value_counts()

        bert_model_name = 'bert-base-multilingual-uncased'
        c_multi, num_sentences_multi, total_time, result_text, false_results_multi = tb.BERT_predictions(masked_sents, correct_preps,
                                                                                full_prep_set, bert_model_name, num_sentences,
                                                                                index_out, original_lang, end_lang)
        tot_correct_multi += c_multi
        tot_sentences_multi += num_sentences_multi
        false_prep_multi[prep] = false_results_multi["Predicted Prep"].value_counts()

        bert_model_name = 'xlm-roberta-base'
        masked_sents, correct_preps, prep_set, index_out, num_sentences = mc.sentence_masking(new_merge, sentence_column_name, prep, model = 'xlm')
        # if num_sentences == 0:
        #     print(f"RoBERTa Sentences for >> {prep} <<: {num_sentences}")
        c_xlmr, num_sentences_xlmr, total_time, result_text, false_results_xlmr = xlmr.XLM_Roberta_predictions(masked_sents, correct_preps,
                                                                                full_prep_set, bert_model_name, num_sentences,
                                                                                index_out, original_lang, end_lang)
        tot_correct_xlmr += c_xlmr
        tot_sentences_xlmr += num_sentences_xlmr
        false_prep_xlmr[prep] = false_results_xlmr["Predicted Prep"].value_counts()
    
    # print(false_prep_mono)
    false_prep_mono.to_csv(f"{original_lang}_to_{end_lang}_mono.txt")
    false_prep_multi.to_csv(f"{original_lang}_to_{end_lang}_multi.txt")
    false_prep_xlmr.to_csv(f"{original_lang}_to_{end_lang}_xlmr.txt")
    
    print("-"*20, "Final Results:", "-"*20)
    print(f'{original_lang} to {end_lang} in BERT-Monolingual:\n\t{tot_correct_mono} out of {tot_sentences_mono} ({100*tot_correct_mono/tot_sentences_mono:0.2f}% Correct)')
    print(f'{original_lang} to {end_lang} in BERT-Multilingual:\n\t{tot_correct_multi} out of {tot_sentences_multi} ({100*tot_correct_multi/tot_sentences_multi:0.2f}% Correct)')
    print(f'{original_lang} to {end_lang} in XLM-RoBERTa:\n\t{tot_correct_xlmr} out of {tot_sentences_xlmr} ({100*tot_correct_xlmr/tot_sentences_xlmr:0.2f}% Correct)')
    
    datafiles = [f'{original_lang}_to_{end_lang}_mono', f'{original_lang}_to_{end_lang}_multi', f'{original_lang}_to_{end_lang}_xlmr']
    for datafile in datafiles:
        make_heatmap(datafile, 'scaled')
        make_heatmap(datafile, 'raw')

# Mand to Mand
filename = "Mandarin_Sentences.txt"
prep_column_name = "Prep_original"
sentence_column_name = "Original"
original_lang = "Mand"
end_lang = "Mand"
bert_model_mono = 'bert-base-chinese'
# # From scott re 3/9 10.06pm picture in slack
incor_mand_prep_ids = [10, 16, 20, 21, 22, 23, 25, 26, 27, 32, 33, 34, 35,
                            36, 39, 40, 41, 42, 43, 44 , 45, 47, 48, 49]
cor_prep_numbers = [x for x in range(50) if x not in incor_mand_prep_ids]
# single_prep_outputs_BERT(filename, prep_column_name, sentence_column_name, bert_model_mono, cor_prep_numbers, original_lang, end_lang)

# Mand to English (Filtering by useful Mand Preps, then finding the english translations of those valid sentences)
prep_column_name = "Prep_Mandarin"
sentence_column_name = "Mandarin"
original_lang = "Mand"
end_lang = "English"
bert_model_mono = 'bert-base-uncased'    
# orig, preps = mc.sentence_cleaning(filename, prep_column_name, sentence_column_name)
# masked_sents, correct_preps, prep_set, index_out, num_sentences = mc.sentence_masking(orig, sentence_column_name)
# results = cmm.make_result_df(orig, "prep_split")
# trunc_res = results[results.index.isin(cor_prep_numbers)]
# new_merge = orig[orig["prep_split"].isin(trunc_res["Prep"])]
# new_merge2 = new_merge[~new_merge["Prep_Mandarin"].str.contains("_")]
# new_merge2 = new_merge2.drop_duplicates(subset = [prep_column_name, sentence_column_name], keep = "first")
# new_merge2["len"] = new_merge2[sentence_column_name].str.split(" ").str.len()
# new_merge2 = new_merge2[new_merge2["len"]>1]
# single_prep_outputs_BERT(filename, prep_column_name, sentence_column_name, bert_model_mono, cor_prep_numbers, original_lang, end_lang, new_merge2)


# English to English
filename = "English_Sentences.txt"
prep_column_name = "Prep_English"
sentence_column_name = "English_Original"
original_lang = "English"
end_lang = "English"
cor_prep_numbers = [x for x in range(20)]
single_prep_outputs_BERT(filename, prep_column_name, sentence_column_name, bert_model_mono, cor_prep_numbers, original_lang, end_lang)


# Spanish to Spanish
filename = "Spanish_Sentences.txt"
prep_column_name = "Prep_original"
sentence_column_name = "Original"
original_lang = "Spanish"
end_lang = "Spanish"
bert_model_mono = "dccuchile/bert-base-spanish-wwm-uncased"
cor_prep_numbers = [x for x in range(15)]
# single_prep_outputs_BERT(filename, prep_column_name, sentence_column_name, bert_model_mono, cor_prep_numbers, original_lang, end_lang)

#Spanish to English
prep_column_name = "Prep_Spanish"
sentence_column_name = "Spanish"
end_lang = "English"
bert_model_mono = 'bert-base-uncased'    
# single_prep_outputs_BERT(filename, prep_column_name, sentence_column_name, bert_model_mono, cor_prep_numbers, original_lang, end_lang)


# # Actually building the heatmaps based on the
# datafiles = [f'{original_lang}_to_{end_lang}_mono', f'{original_lang}_to_{end_lang}_multi', f'{original_lang}_to_{end_lang}_xlmr']
# for datafile in datafiles:
#     make_heatmap(datafile, 'scaled')
#     make_heatmap(datafile, 'raw')



# # Comparing prep_limited dataset and outputs to see which preps are predicted correctly.  not examining individual prep by pre results
# filename = "Mand_to_Mand_truncated.txt"
# prep_column_name = "Prep_original"
# sentence_column_name = "Original"
# original_lang = "Mand"
# end_lang = "Mand"
# bert_model_name = 'bert-base-chinese'
# df = pd.read_csv("Mand_to_Mand_truncated.txt", sep = ",")


# cmm.correct_comparison(filename, prep_column_name, sentence_column_name, original_lang, end_lang, bert_model_name, df)




