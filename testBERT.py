# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:33:56 2020

@author: joshc

BERT_prediction code by Chirag, modularized, documented, and outputs recorded by Josh

Main driver code to send all preprocessed sentences with masks directly into BERT for analysis
"""
import pandas as pd
pd.set_option('display.max_columns', None)
import time
from transformers import BertTokenizer, BertForMaskedLM
from fastai.text.all import *

# from my_cleaning import sentence_cleaning, sentence_masking

def BERT_predictions(masked_sentences, ground_truth_preps, preposition_set, bert_model_name,
                                 num_sentences, index_orig, original_lang, end_lang):
    """
    Stage three of cleaning process:
        Using BERT to predict the masked preposition in a sentence, and 
        reporting data on correct/incorrect choices, as well as sentences 
        which BERT cannot make a prediction for in a reasonable amount of time

    Parameters
    ----------
    masked_sentences : List of strings
        All sentences with the preposition replaced by [MASK] for BERT.
    ground_truth_preps : List of strings
        The expected answer for the masked prep
    preposition_set : List of strings
        All the unique prepositions which occur in the passed in DataFrame,
        to help limit BERT's guessing options
    bert_model_name : string
        The specific model to use from Huggingface's Transformer library
    num_sentences : int
        How many sentences to use from the input sentences
    index_orig : List of ints
        The original location of a sentence from the original datafile, for use in error checking
    original_lang : string
        Shorthand for creating result files.  What language is the text coming from
    end_lang : string
        Shorthand for creating result files.  What language was the original text translated into

    Returns
    -------
    c : int
        The number of correctly predicted prepositions by BERT
    num_sentences : int
        The total number of sentences attempted by BERT
    total_time : float
        How long it took to run BERT over the whole input
    result_text : string
        All the results wrapped up in a nice text block for writting to a file
    false_results : Dataframe
        Contains all info about which sentence predicted which prep specifically
    """
    if num_sentences > len(masked_sentences): num_sentences = len(masked_sentences)

    # Loading the specific BERT model and building the wordpiece tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True) 
    maskedLM_model = BertForMaskedLM.from_pretrained(bert_model_name)
    maskedLM_model.eval();
    x = time.perf_counter()
    p_en = []
    start_time = time.time()
    total_count = 0
    
    # Progress tracker outside of file, just in case
    with open("sent_progress_markers.txt", "a") as file_progress:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        file_progress.write("\n"+"-"*10 +f' {original_lang}_to_{end_lang} '+"-"*10 + str(date_time))
    
    # Sentences which cause BERT to run in an infinite loop of predictions are logged
    with open(f"problem_sentences_{original_lang}_to_{end_lang}.txt", "a") as problem_sentences:
         problem_sentences.write("The following sentences did not find an answer in under 1 second:\n")
    
    # Turning cleaned sentences into Tensors and passing into BERT
    for idx, (s, real_index) in enumerate(zip(masked_sentences[:num_sentences+1], index_orig)):
        
        # Progress tracking outside of file, just in case
        middle_time = time.time()
        progress_text = f"\n--- {bert_model_name} --- "
        check_marks = [int(i*num_sentences//4)-1 for i in range(1,5)]
        for percent, num in enumerate(check_marks,1):
            if idx == num:
                with open("sent_progress_markers.txt", "a") as file_progress:
                    progress_text += f"{percent*25}% completed after "
                    progress_text += "{0:.2f} sec(s)".format(middle_time-start_time)
                    file_progress.write(progress_text)

        # Getting the original sentence ready for BERT and PyTorch
        txt = "[CLS] " + s + " [SEP]"
        tokens_txt = tokenizer.tokenize(txt)
        idx_tokens = [tokenizer.convert_tokens_to_ids(tokens_txt)]
        masked_idx = tokens_txt.index('[MASK]')
        segments_ids = [1] * len(tokens_txt)
        
        # Convert inputs to PyTorch tensors
        segments_tensors = torch.tensor(segments_ids)
        segments_tensors = segments_tensors.reshape(1,-1)
        tokens_tensor = torch.tensor(idx_tokens)
        
        # Predict the missing token (indicated with [MASK]) with `BertForMaskedLM`
        with torch.no_grad(): preds = maskedLM_model(tokens_tensor, segments_tensors)
        preds_idx = [torch.argmax(preds[0][0, masked_idx,:]).item()]
        pred_token = tokenizer.convert_ids_to_tokens(preds_idx)[0]
        elapsed_time = 0
        guess_start_time = time.time()
        while pred_token not in preposition_set:
            preds[0][0,masked_idx,preds_idx]=0
            preds_idx = [torch.argmax(preds[0][0, masked_idx,:]).item()]
            pred_token = tokenizer.convert_ids_to_tokens(preds_idx)[0]  
            guess_end_time = time.time()
            elapsed_time = guess_end_time-guess_start_time
            if elapsed_time > 1:
                total_count += 1
                break
                
        #Store the sentences which require excessive prediction time
        if elapsed_time > 1:   
            with open(f"problem_sentences_{original_lang}_to_{end_lang}.txt", "a") as problem_sentences:
                excess_text = f"{total_count}\t{real_index}\t{ground_truth_preps[idx]}\t{s}\n"
                problem_sentences.write(excess_text)

        p_en.append(pred_token)
    y = time.perf_counter()   
    # avg_count = round(total_count/num_sentences, 2)
    
    # Comparing BERT predicitons to ground truths
    correct_answers_en = ground_truth_preps[:num_sentences]
    c = 0
    false_ids_en = []
    correct_en = []
    predicted_en = []
    cor_sent = []
    for i, (pred, true, index) in enumerate(zip(p_en, correct_answers_en, index_orig)):
        if pred==true:
            c += 1
            bert_model_name = bert_model_name.replace("/", "_")
            # # Used for taking correct sentences to check in other languages
            with open(f"correct_sentences_{original_lang}_to_{end_lang}_{bert_model_name}.txt", "a") as corr_sents:
                correct_sentence = masked_sentences[i].replace("[MASK]", pred)
                correct_pred = pred
                output_text = str(index) + "\t" + correct_pred + "\t" + correct_sentence
                corr_sents.write(output_text+"\n") 
            false_ids_en.append(-1)
        else:
            false_ids_en.append(index)
        correct_en.append(true)
        predicted_en.append(pred)  
        cor_sent.append(masked_sentences[i])
        
    
    total_time = y-x
    result_text = f"Using --> {bert_model_name} <--" + \
          "\n\tBERT predictions finished in: {0:.2f} sec(s)".format(total_time) + \
          f" \n\t{c} correct predictions out of {num_sentences}"+"--> {0:.2f}% Correct".format(100*c/num_sentences) + \
          f"\nThere were {total_count} sentences which went over allowed prediction time."

    result_dict = {"Index": false_ids_en, "Correct Prep": correct_en, "Predicted Prep": predicted_en, "Sentence": cor_sent}
    # print(result_dict)
    name = f"{original_lang}_to_{end_lang}_incorrect_sentences_{bert_model_name}"
    if len(set(correct_answers_en)) == 1:
        name += f"_for_{ground_truth_preps[0]}"
        result_text += f"\nThis data was only for >> {ground_truth_preps[0]} << as the masked prep"
        
        
    false_results = pd.DataFrame(result_dict)
    # false_results.to_csv(name + ".txt", sep = "\t")    

    print()
    print(result_text)
    return c, num_sentences, total_time, result_text, false_results
    