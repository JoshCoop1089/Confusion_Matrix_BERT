# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 22:49:07 2021

@author: joshc
"""
import pandas as pd
import my_cleaning as mc
def make_result_df(correct_mono, typeOfDF):
    mono_out = correct_mono[f"{typeOfDF}"].value_counts()
    mono_out = mono_out.iloc[:50]
    mono_df = mono_out.to_frame()
    mono_df.reset_index(inplace = True)
    mono_df.rename(columns = {"index": "Prep", f"{typeOfDF}": "Count"}, inplace=True)
    mono_df['Percent'] = round(mono_df["Count"]/len(correct_mono)*100, 2)
    # print(mono_df)
    return mono_df
        
def correct_comparison(filename, prep_column_name, sentence_column_name, original_lang, end_lang, bert_model_name, df = ""):
    
    # Full data from original file
    try:
        orig = df
        num_sentences = len(df)
        prep_set = df["prep_split"].unique()
    except Exception:
        orig, preps = mc.sentence_cleaning(filename, prep_column_name, sentence_column_name)
        masked_sents, correct_preps, prep_set, index_out, num_sentences = mc.sentence_masking(orig, sentence_column_name)
    print(f"\nUsing {original_lang} to {original_lang} original data file")
    print(f"Number Sentences Used to Predict: {num_sentences}")
    print(f"Unique Preps: {len(prep_set)}")
    return make_result_df(orig, "prep_split")
    bert_model_name = bert_model_name.replace("/", "_")
    
    # # Import correct sentences from BERTMonolingual
    # corr_file_name = f"correct_sentences_{original_lang}_to_{end_lang}_{bert_model_name}.txt"
    # correct_mono = pd.read_csv(corr_file_name, sep = '\t', header = None)
    # correct_mono.rename(columns = {0: "index", 1:'Prep', 2: "Original_{original_lang}_stripped"}, inplace=True)
    
    # print(f"\nMonolingual BERT Results: {len(correct_mono)} out of {len(orig)}\n"+\
    #       "Unique Preps in Correct Sentences: {correct_mono['Prep'].nunique()}")
    # make_result_df(correct_mono, "Prep")

    # # Import correct sentences from BERTMulti
    # corr_file_name_multi = f"correct_sentences_{original_lang}_to_{end_lang}_bert-base-multilingual-uncased.txt"
    # correct_multi = pd.read_csv(corr_file_name_multi, sep = '\t', header = None)
    # correct_multi.rename(columns = {0: "index", 1:'Prep_multi', 2: f"Original_{original_lang}_stripped_multi"}, inplace=True)
    
    # print(f"\nMultilingual BERT Results: {len(correct_multi)} out of {len(orig)}\n"+\
    #       "Unique Preps in Correct Sentences: {correct_multi['Prep_multi'].nunique()}")
    # make_result_df(correct_multi, "Prep_multi")
     
    # cor_merged = pd.merge(correct_mono, correct_multi, how = "inner", on = 'index')
    # # print(cor_merged.head())
    # print("\n(BERTMono and BERTMulti both made correct predictions)\n" + \
    #       f"Overlapping Sentences: {len(cor_merged)} sentences\nMerged Unique Preps: {cor_merged['Prep'].nunique()}")
    # make_result_df(cor_merged, "Prep")
    

# filename = "Spanish_Sentences.txt"
# prep_column_name = "Prep_original"
# sentence_column_name = "Original"
# original_lang = "Spanish"
# end_lang = "Spanish"

# # filename = "English_Sentences.txt"
# # prep_column_name = "Prep_English"
# # sentence_column_name = "English_Original"
# # original_lang = "English"
# # end_lang = "English"
# # bert_model_name = 'bert-base-uncased' 

# filename = "Mand_to_Mand_truncated.txt"
# df = pd.read_csv(filename)
# print(df.head())
# df2 = pd.read_csv("Mandarin_sentences.txt", sep = '\t')
# df2 = df2[["Prep_Mandarin", "Mandarin"]]
# df2.reset_index(inplace = True)
# print(df2.head())
# merge = pd.merge(df,df2, on = 'index', how = 'inner')
# print(merge.head())
# prep_column_name = "Prep_original"
# sentence_column_name = "Original"
# prep_column_name = "Prep_Mandarin"
# sentence_column_name = "Mandarin"
# original_lang = "Mand"
# end_lang = "Eng"
# bert_model_name = 'bert-base-chinese'

# # # bert_model_name = 'bert-based-uncased'    
# eng_df = correct_comparison(filename, prep_column_name, sentence_column_name, original_lang, end_lang, bert_model_name, merge)
# # # print(eng_df["Count"].sum())
# # eng2 = eng_df.iloc[:20, :]
# # print(eng2['Count'].sum())