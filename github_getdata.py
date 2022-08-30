#!/usr/bin/env python
# coding: utf-8

# In[9]:


def getdata(filename):
    # Step 0: read crawler data，then turn the all Simplified Chinese word into Tradionnal Chinese
    import csv 
    import opencc  
    converter = opencc.OpenCC('s2t.json') #converter of Simplified Chinese into Tradionnal Chinese
    with open(filename,'r' ,  encoding='UTF-8') as csvfile: 
        reader = csv.reader(csvfile) 
        column = [row[4] for row in reader]

    column = [converter.convert(x) for x in column] 

    # Step 1: word segmentation
    from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
    ws = WS("./data")
    word_to_weight = {
        "老高與小茉 Mr & Mrs Gao": 1,
    }

    dictionary = construct_dictionary(word_to_weight)
    sentence_list = column
    word_sentence_list = ws(
        sentence_list,
        sentence_segmentation=True,  #To consider delimiters
        segment_delimiter_set={",", "。", ":", "?", "!", ";", "【", "】", "|"},  # This is the defualt set of delimiters
        coerce_dictionary=dictionary,  # words in this dictionary are forced
    )

    del ws

    # Step 2: import pre_trained Word2Vec model
    import gensim
    from gensim.models.word2vec import Word2Vec
    model = gensim.models.KeyedVectors.load_word2vec_format('pre_trained.bin',
                                                            unicode_errors='ignore',
                                                            binary=True)

    # Step 3: remove meaningless words
    word_rm = []

    del word_sentence_list[0] #remove title

    word_sentence_list_copy = word_sentence_list.copy()
    for words_list in word_sentence_list_copy:
        # if len(words_list) < 2:    # To deal with titles that are too short (less than 2 hyphenated words), add the word "video"
        #     words_list.append("影片")

        words_list_copy = words_list.copy()
        for e in words_list_copy:
            if len(e) < 2:          # remove conjunction
                words_list.remove(e)
                word_rm.append(e)
            elif len(e) > 6:        # remove the words that are too long
                words_list.remove(e)
                word_rm.append(e)
            elif e[0] == ' ' or e[-1] == ' ':
                words_list.remove(e)
                word_rm.append(e)


    # Step 4: remove the word that don't exist in the word2vector pre-trained model
    word_notInModel = []
    for words_list in word_sentence_list:
        words_list_copy = words_list.copy()
        for w in words_list_copy:
            try:  
                model[w]
            except:  
                words_list.remove(w)
                word_notInModel.append(w)

        while len(words_list) <= 2:    # To deal with titles that are too short (less than 2 hyphenated words), add the word "video"
            words_list.append("影片")

    #After processing the data, save it in word_sentence_list, and go to the next step
    return word_sentence_list

# print((getdata('YT_output.csv')))
# print(len(getdata('YT_output.csv')))