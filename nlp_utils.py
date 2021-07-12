### Module containing auxiliary functions and classes for performing NER using Transformers


## Load text

import os

def load_text_files(file_names, path):
    """
    It loads the text contained in a set of files into a returned list of strings.
    Code adapted from https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(path + f, "r") as file:
            output.append(file.read())
            
    return output


def load_ss_files(file_names, path):
    """
    It loads the start-end pair of each split sentence from a set of files (start + \t + end line-format expected) into a 
    returned dictionary, where keys are file names and values a list of tuples containing the start-end pairs of the 
    split sentences.
    """
    output = dict()
    for f in file_names:
        with open(path + f, "r") as file:
            f_key = f.split('.')[0]
            output[f_key] = []
            for sent in file:
                output[f_key].append(tuple(map(int, sent.strip().split('\t'))))
            
    return output


import numpy as np
import pandas as pd

def process_brat_ner(brat_files):
    """
    Primarly dessign to process Cantemist-NER annotations.
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    """
    
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            for line in ann_file:
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                text_ref = line_split[2]
                location = ' '.join(line_split[1].split(' ')[1:]).split(';')
                # Discontinuous annotations are split into a sequence of continuous annotations
                for loc in location:
                    split_loc = loc.split(' ')
                    df_res.append([doc_name, text_ref, int(split_loc[0]), int(split_loc[1])])

    return pd.DataFrame(df_res, 
columns=["doc_id", "text_ref", "start", "end"])



## Whitespace-punctuation tokenization (same as BERT pre-tokenization)
# The next code is adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

import unicodedata

def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

def is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

def is_space(ch):
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'

def is_control(ch):
    """
    Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py#L64
    """
    return unicodedata.category(ch).startswith("C")


def word_start_end(text, start_i=0, cased=True):
    """
    Our aim is to produce both a list of strings containing the text of each word and a list of pairs containing the start and
    end char positions of each word.
    
    start_i: the start position of the first character in the text.
    
    Code adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101
    """
    
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    # Store the start positions of each considered character (ch) in start_arr, 
    # such that sum([len(word) for word in spaced.strip().split()]) = len(start_arr)
    start_arr = [] 
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            start_arr.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not(ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            start_arr.append(start_i)
        # If it is a control char we skip it but take its offset into account
        start_i += 1
    
    assert sum([len(word) for word in spaced.strip().split()]) == len(start_arr)
    
    text_arr, start_end_arr = [], []
    i = 0
    for word in spaced.strip().split():
        text_arr.append(word)
        j = i + len(word)
        start_end_arr.append((start_arr[i], start_arr[j - 1] + 1))
        i = j
        
    return text_arr, start_end_arr



## NER-annotations

# Tokenization analysis

def check_overlap_ner(df_ann, doc_list):
    """
    This function returns the pairs of named entities in a single document with overlapping
    spans.
    """
    res = []
    for doc in doc_list:
        df_doc = df_ann[df_ann['doc_id'] == doc]
        len_doc = df_doc.shape[0]
        for i in range(len_doc - 1):
            start_i = df_doc.iloc[i]['start']
            end_i = df_doc.iloc[i]['end']
            for j in range(i + 1, len_doc):
                start_j = df_doc.iloc[j]['start']
                end_j = df_doc.iloc[j]['end']
                if start_i < end_j and start_j < end_i:
                    res.append((doc, start_i, end_i, start_j, end_j, (start_i >= start_j and end_i <= end_j) or \
                                                                     (start_j >= start_i and end_j <= end_i)))
                
    return pd.DataFrame(res, columns=["doc_id", "start_1", "end_1", "start_2", "end_2", "contained"])


from tqdm import tqdm

def eliminate_overlap(df_ann):
    """
    For each pair of existing overlapping annotations in a document, the longer is eliminated.
    Then, the existence of overlapping annotations is re-evaluated.
    """
    df_res = df_ann.copy()
    for doc in tqdm(sorted(set(df_ann['doc_id']))):
        doc_over = check_overlap_ner(df_ann=df_res, doc_list=[doc])
        while doc_over.shape[0] > 0:
            # There are overlapping annotations in current doc
            aux_row = doc_over.iloc[0]
            len_1 = aux_row['end_1'] - aux_row['start_1']
            len_2 = aux_row['end_2'] - aux_row['start_2']
            if len_1 >= len_2:
                elim_start = aux_row['start_1']
                elim_end = aux_row['end_1']
            else:
                elim_start = aux_row['start_2']
                elim_end = aux_row['end_2']
            # Eliminate longer overlapping annotation
            df_res = df_res[(df_res['doc_id'] != aux_row['doc_id']) | (df_res['start'] != elim_start) | (df_res['end'] != elim_end)]
            doc_over = check_overlap_ner(df_ann=df_res, doc_list=[doc])
        
    return df_res


# Creation of a NER corpus

def ner_iob2_annotate(arr_start_end, df_ann):
    """
    Annotate a sequence of words following IOB-2 NER format, using the start-end char positions of each NER-annotation.
    All annotations are expected to be contained within the input sequence.
    
    Time complexity: O(n*m); n = df_ann.shape[0], m = len(arr_start_end)
    """
    
    labels = ["O"] * len(arr_start_end)
    for index, row in df_ann.iterrows():
        # First word of annotation
        tok_start = np.where(arr_start_end[:, 0] <= row['start'])[0][-1] # last word <= annotation start
        # Last word of annotation
        tok_end = np.where(arr_start_end[:, 1] >= row['end'])[0][0] # first word >= annotation end
        assert tok_start <= tok_end
        ## Annotate first word
        # Sanity check (no overlapping annotations are expected, with an exception)
        if labels[tok_start] != "O": # Because of the "pT3N2Mx" annotation (two ann in a single word) in dev-set2 cc_onco1427
            print(labels[tok_start])
            print(row)
            print(tok_start)
            print(tok_end)
            print(arr_start_end)
        
        labels[tok_start] = "B"
        if tok_start < tok_end:
            # Annotation spanning multiple words
            for i in range(tok_start + 1, tok_end + 1):
                assert labels[i] == "O" # no overlapping annotations are expected
                labels[i] = "I"
    
    return labels


def start_end_tokenize(text, tokenizer, start_pos=0):
    """
    Our aim is to produce both a list of sub-tokens and a list of tuples containing the start and
    end char positions of each sub-token.
    """
    
    start_end_arr = []
    token_text = tokenizer(text, add_special_tokens=False)
    for i in range(len(token_text['input_ids'])):
        chr_span = token_text.token_to_chars(i)
        start_end_arr.append((chr_span.start + start_pos, chr_span.end + start_pos))
        
    return tokenizer.convert_ids_to_tokens(token_text['input_ids']), start_end_arr


def convert_word_token(word_text, word_start_end, word_labels, tokenizer, ign_value, strategy, word_pos):
    """
    Given a list of words, the function converts them to subtokens.
    """
    res_sub_token, res_start_end, res_word_id, res_labels = [], [], [], []
    for i in range(len(word_text)):
        w_text = word_text[i]
        w_start_end = word_start_end[i]
        w_label = word_labels[i]
        sub_token, _ = start_end_tokenize(text=w_text, tokenizer=tokenizer, start_pos=w_start_end[0])
        tok_start_end = [w_start_end] * len(sub_token) # using the word start-end pair as the start-end position of the subtokens
        tok_word_id = [i + word_pos] * len(sub_token)
        labels = [ign_value] * len(sub_token)
        labels[0] = w_label
        if strategy.split('-')[1] == "all":
            for j in range(1, len(sub_token)):
                labels[j] = w_label
        res_sub_token.extend(sub_token)
        res_start_end.extend(tok_start_end)
        res_word_id.extend(tok_word_id)
        res_labels.extend(labels)
        
    return res_sub_token, res_start_end, res_word_id, res_labels


def start_end_tokenize_ner(text, max_seq_len, tokenizer, start_pos, df_ann, ign_value, strategy="word-all", cased=True, word_pos=0):
    """
    Given an input text, this function tokenizes the text into sub-tokens, annotates the text at the word-level, and finally
    converts the annotations to subtoken-level.
    """
    
    out_sub_token, out_start_end, out_labels, out_word_id = [], [], [], []
    # Apply whitespace and punctuation pre-tokenization to extract the words from the input text
    word_text, word_chr_start_end = word_start_end(text=text, start_i=start_pos, cased=cased) 
    assert len(word_text) == len(word_chr_start_end)
    # Obtain IOB-2 labels at word-level
    word_labels = ner_iob2_annotate(arr_start_end=np.array(word_chr_start_end), df_ann=df_ann)
    assert len(word_labels) == len(word_chr_start_end)
    # Convert word-level arrays to subtoken-level
    sub_token, start_end, word_id, labels = convert_word_token(word_text=word_text, word_start_end=word_chr_start_end, 
                                        word_labels=word_labels, tokenizer=tokenizer, ign_value=ign_value, strategy=strategy, 
                                        word_pos=word_pos)
        
    assert len(sub_token) == len(start_end) == len(word_id) == len(labels)
    # Re-split large sub-tokens sequences
    for i in range(0, len(sub_token), max_seq_len):
        out_sub_token.append(sub_token[i:i+max_seq_len])
        out_start_end.append(start_end[i:i+max_seq_len])
        out_labels.append(labels[i:i+max_seq_len])
        out_word_id.append(word_id[i:i+max_seq_len])
    
    return out_sub_token, out_start_end, out_labels, out_word_id
    
        
def ss_start_end_tokenize_ner(ss_start_end, max_seq_len, text, tokenizer, df_ann, ign_value, strategy="word-all", cased=True):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of 
                  the split sentences from the input document text.
    text: document text.
    strategy: possible values are "word-first", "word-all"
    
    return: 4 lists of lists, the first for the sub-tokens from the re-split sentences, the second for the 
            start-end char positions pairs of the sub-tokens from the re-split sentences, the third for
            the IOB-2 labels associated to the sub-tokens from the re-split sentences, and the fourth for the 
            word id of each sub-token.
    """
    out_sub_token, out_start_end, out_labels, out_word_id = [], [], [], []
    n_ss_words = 0
    for ss_start, ss_end in ss_start_end:
        ss_text = text[ss_start:ss_end]
        # annotations spanning multiple adjacent sentences are not considered
        ss_ann = df_ann[(df_ann['start'] >= ss_start) & (df_ann['end'] <= ss_end)]
        ss_sub_token, ss_start_end, ss_labels, ss_word_id = start_end_tokenize_ner(text=ss_text, max_seq_len=max_seq_len,
                            tokenizer=tokenizer, start_pos=ss_start, df_ann=ss_ann, ign_value=ign_value, 
                            strategy=strategy, cased=cased, word_pos=n_ss_words)
        out_sub_token.extend(ss_sub_token)
        out_start_end.extend(ss_start_end)
        out_labels.extend(ss_labels)
        out_word_id.extend(ss_word_id)
        
        # We update the number of words contained in the previous sentences
        n_ss_words += len(set([w_id for frag in ss_word_id for w_id in frag]))
    
    return out_sub_token, out_start_end, out_labels, out_word_id


def ss_fragment_greedy_ner(ss_token, ss_start_end, ss_labels, ss_word_id, max_seq_len):
    """
    Implementation of the multiple-sentence fine-tuning approach developed in http://ceur-ws.org/Vol-2664/cantemist_paper15.pdf,
    which consists in generating text fragments containing the maximum number of adjacent split sentences, such that the length of 
    each fragment is <= max_seq_len.
    """
    
    frag_token, frag_start_end, frag_labels, frag_word_id = [[]], [[]], [[]], [[]]
    i = 0
    while i < len(ss_token):
        assert len(ss_token[i]) <= max_seq_len
        if len(frag_token[-1]) + len(ss_token[i]) > max_seq_len:
            # Fragment is full, so create a new empty fragment
            frag_token.append([])
            frag_start_end.append([])
            frag_labels.append([])
            frag_word_id.append([])
            
        frag_token[-1].extend(ss_token[i])
        frag_start_end[-1].extend(ss_start_end[i])
        frag_labels[-1].extend(ss_labels[i])
        frag_word_id[-1].extend(ss_word_id[i])
        i += 1
          
    return frag_token, frag_start_end, frag_labels, frag_word_id


def format_token_ner(token_list, label_list, tokenizer, seq_len, lab_encoder, ign_value):
    """
    Given a list of sub-tokens and their assigned NER-labels, as well as a tokenizer, it returns their corresponding lists of 
    indices, attention masks, tokens types and transformed labels. Padding is added as appropriate.
    """
    
    token_ids = tokenizer.convert_tokens_to_ids(token_list)
    # Add [CLS] and [SEP] tokens (single sequence)
    token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    
    # Generate attention mask
    token_len = len(token_ids)
    attention_mask = [1] * token_len
    
    # Generate token types
    token_type = [0] * token_len
    
    # Add special tokens labels
    token_labels = [ign_value] + [lab_encoder.transform([label])[0] if label != ign_value else label \
                                  for label in label_list] + [ign_value]
    assert len(token_labels) == token_len
    
    # Padding
    pad_len = seq_len - token_len
    token_ids += [tokenizer.pad_token_id] * pad_len
    attention_mask += [0] * pad_len
    token_type += [0] * pad_len
    token_labels += [ign_value] * pad_len

    return token_ids, attention_mask, token_type, token_labels


from copy import deepcopy

def ss_create_input_data_ner(df_text, text_col, df_ann, doc_list, ss_dict, tokenizer, lab_encoder, seq_len, ign_value, 
                             strategy="word-all", greedy=False, cased=True):
    """
    This function generates the data needed to fine-tune a transformer model on a NER multi-class token classification task, 
    such as Cantemist-NER subtask, following the IOB-2 annotation format.
    
    df_text: DataFrame containing the documents IDs ("doc_id" column expected) and the text from the documents.
    text_col: name of the column of df_text DataFrame that contains the text from the documents.
    df_ann: DataFrame containing the NER annotations of the documents, in the same format as the DataFrame 
            returned by process_brat_ner function.
    doc_list: list containing the documents IDs to be considered. df_text, df_ann and ss_dict are expected to
              contain all documents present in doc_list.
    ss_dict: dict where keys are documents IDs and each value is a list of tuples containing the start-end char positions 
             pairs of the split sentences in each document. It uses the same format as the dict returned by the 
             load_ss_files function. If None, the function implements the text-stream fragment-based fine-tuning approach
             (see https://doi.org/10.1109/ACCESS.2021.3080085).
    tokenizer: transformers.PreTrainedTokenizerFast instance.
    lab_encoder: sklearn.preprocessing.LabelEncoder instance already fit.
    seq_len: maximum input sequence size.
    ign_value: label value assigned to the special tokens (CLS, SEQ, PAD). Tokens assigned this value are ignored
               when computing the loss.
    strategy: approach followed to convert the word-level annotations to subtoken-level. 
              Possible values are "word-all", "word-first".
    greedy: boolean parameter indicating the strategy followed to generate the text fragments. 
            If True, the multiple-sentence approach is followed, which consists in generating fragments containing the maximum 
            number of adjacent split sentences, such that the length of each fragment is <= seq_len-2.
            If False, the single-sentence strategy is implemented. See https://doi.org/10.1109/ACCESS.2021.3080085
    cased: boolean parameter indicating whether casing is preserved during text tokenization.
    
    returns: indices: np.array of shape total_n_frag x seq_len, containing the indices (input IDs) of each generated 
                      sub-tokens fragment.
             attention_mask: np.array of shape total_n_frag x seq_len, containing the attention mask of each generated 
                      sub-tokens fragment.
             token_type: np.array of shape total_n_frag x seq_len, containing the token type IDs of each generated 
                       sub-tokens fragment.
             labels: np.array of shape total_n_frag x seq_len, containing label with which each subtoken is annotated.
             n_fragments: np.array of shape n_doc, containing the number of fragments generated for each document.
             start_end_offsets: list of lists of tuples of shape total_n_frag x frag_len (per frag) x 2, containing, for each
                                sub-token, the start and end character offset of the word from which the sub-token is obtained.
             word_ids: list of lists of shape total_n_frag x frag_len (per frag), containing, for each sub-token, the ID (int) of
                       the word from which the sub-token is obtained.
    """
    
    indices, attention_mask, token_type, labels, n_fragments, start_end_offsets, word_ids = [], [], [], [], [], [], []
    for doc in tqdm(doc_list):
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        ## Generate annotated subtokens sequences
        if ss_dict is not None:
            # Perform sentence split (SS) on doc text
            doc_ss = ss_dict[doc] # SS start-end pairs of the doc text
            doc_ss_token, doc_ss_start_end, doc_ss_label, doc_ss_word_id = ss_start_end_tokenize_ner(ss_start_end=doc_ss, 
                                        max_seq_len=seq_len-2, text=doc_text, 
                                        tokenizer=tokenizer, df_ann=doc_ann, ign_value=ign_value, strategy=strategy, cased=cased)
            assert len(doc_ss_token) == len(doc_ss_start_end) == len(doc_ss_label) == len(doc_ss_word_id)
            if greedy:
                # Split the list of sub-tokens sentences into sequences comprising multiple sentences
                frag_token, frag_start_end, frag_label, frag_word_id = ss_fragment_greedy_ner(ss_token=doc_ss_token, 
                                ss_start_end=doc_ss_start_end, ss_labels=doc_ss_label, ss_word_id=doc_ss_word_id, max_seq_len=seq_len-2)
            else: 
                frag_token = deepcopy(doc_ss_token)
                frag_start_end = deepcopy(doc_ss_start_end)
                frag_label = deepcopy(doc_ss_label)
                frag_word_id = deepcopy(doc_ss_word_id)
        else:
            # Generate annotated sequences using text-stream strategy (without considering SS)
            frag_token, frag_start_end, frag_label, frag_word_id = start_end_tokenize_ner(text=doc_text, max_seq_len=seq_len-2,
                            tokenizer=tokenizer, start_pos=0, df_ann=doc_ann, ign_value=ign_value, 
                            strategy=strategy, cased=cased, word_pos=0)
            
        assert len(frag_token) == len(frag_start_end) == len(frag_label) == len(frag_word_id)
        # Store the start-end char positions of all the sequences
        start_end_offsets.extend(frag_start_end)
        # Store the sub-tokens word ids of all the sequences
        word_ids.extend(frag_word_id)
        # Store the number of sequences of each doc text
        n_fragments.append(len(frag_token))
        ## Subtokens sequences formatting
        for f_token, f_start_end, f_label, f_word_id in zip(frag_token, frag_start_end, frag_label, frag_word_id):
            # sequence length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) == len(f_label) == len(f_word_id) <= seq_len-2
            f_id, f_att, f_type, f_label = format_token_ner(f_token, f_label, tokenizer, seq_len, lab_encoder, ign_value)
            indices.append(f_id)
            attention_mask.append(f_att)
            token_type.append(f_type)
            labels.append(f_label)
            
    return np.array(indices), np.array(attention_mask), np.array(token_type), np.array(labels), \
           np.array(n_fragments), start_end_offsets, word_ids



## NER performance evaluation

def extract_seq_preds_iob2(doc_id, iob_seq_preds, seq_start_end, df_text, text_col):
    res = []
    left = 0
    while left < len(iob_seq_preds):
        if iob_seq_preds[left] == "B":
            right = left + 1
            while right < len(iob_seq_preds):
                if iob_seq_preds[right] != "I":
                    break
                right += 1
            # Add annotation
            res.append({'clinical_case': doc_id, 'text': df_text[df_text['doc_id'] == doc_id][text_col].values[0], 
                                'start': seq_start_end[left][0], 'end': seq_start_end[right - 1][1]})
            left = right # next sub-token different from "I", or len(iob_seq_preds) (out of bounds)
        else:
            left += 1
    
    return res


def word_seq_preds(tok_seq_word_id, tok_seq_preds, tok_seq_start_end, strategy):
    """
    Implemented strategies: "word-first", "word-max", "word-prod".
    """

    # Convert subtoken-level predictions to word-level predictions
    arr_word_seq_preds, arr_word_seq_start_end = [], []
    left = 0
    while left < len(tok_seq_word_id):
        cur_word_id = tok_seq_word_id[left]
        right = left + 1
        while right < len(tok_seq_word_id):
            if tok_seq_word_id[right] != cur_word_id:
                break
            right += 1
        # cur_word_id spans from left to right - 1 subtoken positions
        assert len(set(tok_seq_start_end[left:right])) == 1 # start-end positions of the subtokens correpond to the word start-end pair
        arr_word_seq_start_end.append(tok_seq_start_end[left])
        
        if strategy.split('-')[-1] == "first":
            # predictions made on the first subtoken of the word
            arr_word_seq_preds.append(tok_seq_preds[left]) 
        
        elif strategy.split('-')[-1] == "max":
            # max of predictions made in all subtokens of the word 
            arr_word_seq_preds.append(np.max(tok_seq_preds[left:right], axis=0))
        
        elif strategy.split('-')[-1] == "prod":
            # product of predictions made in all subtokens of the word 
            arr_word_seq_preds.append(np.prod(tok_seq_preds[left:right], axis=0))
        
        else:
            raise Exception('Word strategy not implemented!')

        left = right
    
    assert cur_word_id == tok_seq_word_id[-1]
    
    return arr_word_seq_preds, arr_word_seq_start_end


def seq_ner_preds_brat_format(doc_list, fragments, arr_start_end, arr_word_id, arr_preds, strategy="word-all"):
    """
    Implemented strategies: "word-first", "word-max", "word-prod".
    """
    
    arr_doc_seq_preds, arr_doc_seq_start_end = [], []
    i = 0
    for d in tqdm(range(len(doc_list))):
        n_frag = fragments[d]
        # Extract subtoken-level arrays for each document (joining adjacent fragments)
        doc_tok_start_end = [ss_pair for frag in arr_start_end[i:i+n_frag] for ss_pair in frag]
        doc_tok_word_id = [w_id for frag in arr_word_id[i:i+n_frag] for w_id in frag]
        assert len(doc_tok_start_end) == len(doc_tok_word_id)
        
        # Extract predictions, ignoring special tokens (CLS, SEQ, PAD)
        # doc_tok_preds shape: n_tok (per doc) x n_labels (3)
        doc_tok_preds = np.array([preds for j in range(i, i+n_frag) \
            for preds in arr_preds[j][1:len(arr_start_end[j])+1]])
        assert doc_tok_preds.shape[0] == len(doc_tok_start_end)
        
        # Convert arrays to word-level
        doc_word_seq_preds, doc_word_seq_start_end = word_seq_preds(tok_seq_word_id=doc_tok_word_id, 
                                    tok_seq_preds=doc_tok_preds, tok_seq_start_end=doc_tok_start_end, strategy=strategy)
        assert len(doc_word_seq_preds) == len(doc_word_seq_start_end) == (doc_tok_word_id[-1] + 1)

        arr_doc_seq_preds.append(doc_word_seq_preds) # final shape: n_doc x n_words (per doc) x n_labels (3)
        arr_doc_seq_start_end.append(doc_word_seq_start_end) # final shape: n_doc x n_words (per doc) x 2 (start-end pair)

        i += n_frag

    return arr_doc_seq_preds, arr_doc_seq_start_end


def ner_preds_brat_format(doc_list, fragments, preds, start_end, word_id, lb_encoder, df_text, text_col, 
                          strategy="word-all"):
    
    # Post-process the subtoken annotations predicted for each document
    arr_doc_seq_preds, arr_doc_seq_start_end = seq_ner_preds_brat_format(doc_list=doc_list, fragments=fragments, 
                            arr_start_end=start_end, arr_word_id=word_id, arr_preds=preds, strategy=strategy)
    ann_res = []
    for d in tqdm(range(len(doc_list))):
        doc = doc_list[d]
        arr_seq_preds = arr_doc_seq_preds[d]
        arr_seq_start_end = arr_doc_seq_start_end[d]
        arr_seq_iob_preds = [lb_encoder.classes_[pred] for pred in np.argmax(arr_seq_preds, axis=1)]
        ann_res.extend(extract_seq_preds_iob2(doc_id=doc, iob_seq_preds=arr_seq_iob_preds, 
                                     seq_start_end=arr_seq_start_end, df_text=df_text, text_col=text_col))
    
    return pd.DataFrame(ann_res)


def ens_ner_preds_brat_format(doc_list, ens_doc_word_preds, ens_doc_word_start_end, lb_encoder, df_text, text_col, 
                              strategy="max"):
    """
    Implemented strategies: "max", "prod", "sum".
    """
    
    # Sanity check: same word start-end arrays obatined from different models
    doc_word_start_end = ens_doc_word_start_end[0]
    for i in range(len(doc_list)):
        aux_san_arr = np.array(doc_word_start_end[i])
        for j in range(1, len(ens_doc_word_start_end)):
            comp_arr = np.array(ens_doc_word_start_end[j][i])
            assert np.array_equal(aux_san_arr, comp_arr)
    
    # Merge predictions per document
    ann_res = []
    i = 0
    for d in tqdm(range(len(doc_list))):
        arr_ens_word_preds = np.array([word_preds[d] for word_preds in ens_doc_word_preds]) 
        # shape: n_ens x n_words (per doc) x n_labels (3) 
        if strategy == "max":
            arr_word_preds = np.max(arr_ens_word_preds, axis=0)
        elif strategy == "prod":
            arr_word_preds = np.prod(arr_ens_word_preds, axis=0)
        elif strategy == "sum":
            arr_word_preds = np.sum(arr_ens_word_preds, axis=0)
        else:
            raise Exception('Ensemble evaluation strategy not implemented!')

        arr_word_iob_preds = [lb_encoder.classes_[pred] for pred in np.argmax(arr_word_preds, axis=1)]
        ann_res.extend(extract_seq_preds_iob2(doc_id=doc_list[d], iob_seq_preds=arr_word_iob_preds, 
                                     seq_start_end=doc_word_start_end[d], df_text=df_text, text_col=text_col))
    
    return pd.DataFrame(ann_res)


import shutil

def write_ner_ann(df_pred_ann, out_path, ann_label="MORFOLOGIA_NEOPLASIA"):
    """
    Write a set of NER-annotations from different documents in BRAT format.
    """
    
    # Create a new output directory
    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.mkdir(out_path)
    
    for doc in sorted(set(df_pred_ann['clinical_case'])):
        doc_pred_ann = df_pred_ann[df_pred_ann['clinical_case'] == doc]
        with open(out_path + doc + ".ann", "w") as out_file:
            i = 1
            for index, row in doc_pred_ann.iterrows():
                start = row['start']
                end = row['end']
                out_file.write("T" + str(i) + "\t" + ann_label + " " + str(start) + " " + 
                               str(end) + "\t" + row['text'][start:end].replace("\n", " ") + "\n")
                i += 1
    return

import sys
sys.path.append("../resources/cantemist-evaluation-library/src/")
import ann_parsing

def format_ner_gs(gs_path, subtask='ner'):
    """
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_ner_norm.py
    """
    
    if subtask=='norm':
        gs = ann_parsing.main(gs_path, ['MORFOLOGIA_NEOPLASIA'], with_notes=True)
        
        if gs.shape[0] == 0:
            raise Exception('There are not parsed Gold Standard annotations')
        
        gs.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 'code_gs',
                      'start_pos_gs', 'end_pos_gs']
        
    elif subtask=='ner':
        gs = ann_parsing.main(gs_path, ['MORFOLOGIA_NEOPLASIA'], with_notes=False)
        
        if gs.shape[0] == 0:
            raise Exception('There are not parsed Gold Standard annotations')
        
        gs.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 
                      'start_pos_gs', 'end_pos_gs']
        
    else:
        raise Exception('Error! Subtask name not properly set up')
    
    return gs


def format_ner_pred(gs_path, pred_path, subtask='ner'):
    """
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_ner_norm.py
    """
    
    # Get ANN files in Gold Standard
    ann_list_gs = list(filter(lambda x: x[-4:] == '.ann', os.listdir(gs_path)))
    
    if subtask=='norm':
        pred = ann_parsing.main(pred_path, ['MORFOLOGIA_NEOPLASIA','MORFOLOGIA-NEOPLASIA'], with_notes=True)
        
        if pred.shape[0] == 0:
            raise Exception('There are not parsed predicted annotations')
        
        pred.columns = ['clinical_case', 'mark', 'label', 'offset', 'span', 'code_pred',
                      'start_pos_pred', 'end_pos_pred']
    elif subtask=='ner':
        pred = ann_parsing.main(pred_path, ['MORFOLOGIA_NEOPLASIA','MORFOLOGIA-NEOPLASIA'], with_notes=False)
        
        if pred.shape[0] == 0:
            raise Exception('There are not parsed predicted annotations')
        
        pred.columns = ['clinical_case', 'mark', 'label', 'offset', 'span',
                      'start_pos_pred', 'end_pos_pred']
    else:
        raise Exception('Error! Subtask name not properly set up')

    # Remove predictions for files not in Gold Standard
    pred_gs_subset = pred.loc[pred['clinical_case'].isin(ann_list_gs),:]
    
    return pred_gs_subset


def format_ner_pred_df(gs_path, df_preds):
    """
    df_preds: same format as returned by ner_preds_brat_format function.
    
    return: pd.DataFrame with the two columns expected in calculate_ner_metrics function.
    """
    
    # Get ANN files in Gold Standard
    ann_list_gs = list(filter(lambda x: x[-4:] == '.ann', os.listdir(gs_path)))
    
    df_preds_res = df_preds.copy()
    
    # Add .ann suffix
    df_preds_res['clinical_case'] = df_preds_res['clinical_case'].apply(lambda x: x + '.ann')
    
    # Remove predictions for files not in Gold Standard
    df_pred_gs_subset = df_preds_res.loc[df_preds_res['clinical_case'].isin(ann_list_gs),:]
    
    df_pred_gs_subset['offset'] = df_pred_gs_subset.apply(lambda x: str(x['start']) + ' ' + str(x['end']), axis=1)
    
    return df_pred_gs_subset[['clinical_case', 'offset', 'start', 'end']]


def calculate_ner_metrics(gs, pred, subtask='ner'):
    """
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_ner_norm.py
    """
    
    # Predicted Positives:
    Pred_Pos = pred.drop_duplicates(subset=['clinical_case', "offset"]).shape[0]
    
    # Gold Standard Positives:
    GS_Pos = gs.drop_duplicates(subset=['clinical_case', "offset"]).shape[0]
    
    # Eliminate predictions not in GS (prediction needs to be in same clinical
    # case and to have the exact same offset to be considered valid!!!!)
    df_sel = pd.merge(pred, gs, 
                      how="right",
                      on=["clinical_case", "offset"])
    
    if subtask=='norm':
        # Check if codes are equal
        df_sel["is_valid"] = \
            df_sel.apply(lambda x: (x["code_gs"] == x["code_pred"]), axis=1)
    elif subtask=='ner':
        is_valid = df_sel.apply(lambda x: x.isnull().any()==False, axis=1)
        df_sel = df_sel.assign(is_valid=is_valid.values)
    else:
        raise Exception('Error! Subtask name not properly set up')
    

    # There are two annotations with two valid codes. Any of the two codes is considered as valid
    if subtask=='norm':
        df_sel = several_codes_one_annot(df_sel)
        
    # True Positives:
    TP = df_sel[df_sel["is_valid"] == True].shape[0]
    
    # Calculate Final Metrics:
    P = TP / Pred_Pos
    R = TP / GS_Pos
    if (P+R) == 0:
        F1 = 0
    else:
        F1 = (2 * P * R) / (P + R)
    
    
    if (any([F1, P, R]) > 1):
        warnings.warn('Metric greater than 1! You have encountered an undetected bug, please, contact antonio.miranda@bsc.es!')
                                            
    return round(P, 3), round(R, 3), round(F1, 3)


def several_codes_one_annot(df_sel):
    
    # If any of the two valid codes is predicted, give both as good
    if any(df_sel.loc[(df_sel['clinical_case']=='cc_onco838.ann') & 
                  (df_sel['offset'] == '2509 2534')]['is_valid']):
        df_sel.loc[(df_sel['clinical_case']=='cc_onco838.ann') &
                   (df_sel['offset'] == '2509 2534'),'is_valid'] = True
            
    if any(df_sel.loc[(df_sel['clinical_case']=='cc_onco1057.ann') & 
                      (df_sel['offset'] == '2791 2831')]['is_valid']):
        df_sel.loc[(df_sel['clinical_case']=='cc_onco1057.ann') &
                   (df_sel['offset'] == '2791 2831'),'is_valid'] = True
        
    # Remove one of the entries where there are two valid codes
    df_sel.drop(df_sel.loc[(df_sel['clinical_case']=='cc_onco838.ann') &
                    (df_sel['offset'] == '2509 2534') & 
                    (df_sel['code_gs']=='8441/0')].index, inplace=True)
    
    df_sel.drop(df_sel.loc[(df_sel['clinical_case']=='cc_onco1057.ann') &
            (df_sel['offset'] == '2791 2831') & 
            (df_sel['code_gs']=='8803/3')].index, inplace=True)
        
    return df_sel



## NER loss and callbacks

import tensorflow as tf

class TokenClassificationLoss(tf.keras.losses.Loss):
    """
    Code adapted from https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFTokenClassificationLoss
    """
    
    def __init__(self, from_logits=True, ignore_val=-100, **kwargs):
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        super(TokenClassificationLoss, self).__init__(**kwargs)
        
    
    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val
        reduced_preds = tf.boolean_mask(tf.reshape(y_pred, (-1, y_pred.shape[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
        
        return loss_fn(labels, reduced_preds)
