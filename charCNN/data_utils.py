import torch
import numpy as np
#from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils
import pickle
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import numpy as np
import re
import csv
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import sentencepiece as spm
from nltk.tokenize import WhitespaceTokenizer
tk = WhitespaceTokenizer()
from itertools import chain

class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """
    #alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
    def __init__(self, data_source,data_labels,data_source1,data_labels1,data_source2,data_labels2,alphabet_size,input_size=1014, num_of_classes=4,perform=False):
        """
        Initialization of a Data object.

        Args:
            data_source (str): Raw data file path
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        
        self.length = input_size
        self.data_source = data_source
        self.data_labels=data_labels
        self.data_source1= data_source1
        self.data_labels1=data_labels1
        self.data_source2 = data_source2
        self.data_labels2=data_labels2
        
        self.alphabet_size=alphabet_size
        self.perform=perform
        
        
        
        #self.merge_size=merge_size
        
        #set a flag "perform" if you want to do this experiment, otherwise just use the given alphabet_size
    def min_vocab(self,sentences):
        raw = " ".join(sentences)
        alphabet = sorted(list(set(raw)))
        
        
        vocab_start = [len(alphabet)]
        return vocab_start
        
    def opt_merge_size(self,sentences):
        s1=[tk.tokenize(x) for x in sentences]
        words = list(chain(*s1))
        words=list(set(words))
        
        m_size=round(0.4*len(words))
        print(len(words),m_size)
        return m_size
        
        
        
        
    def merge_op(self,vocab_file):
        with open(vocab_file) as f:
            f1=f.readlines()
            f2=[x.split("\t")[0] for x in f1]
            chars=[]
            special=['<unk>','<s>','</s>']
            for x in f2:
                if x.startswith("_"):
                    if len(x)==2:
                        chars.append(x)
                if len(x)==1:
                    chars.append(x)
                if x in special:
                    chars.append(x)
            self.sw=[x for x in f2 if x not in chars]
            print(len(f2))
            print(len(chars))
            print(len(self.sw))
            return len(self.sw)

    def load_data(self):
        """
        Load raw data from the source file into data variable.

        Returns: None

        """
        self.m=self.data_source.split("/")[1]
        
        print("dataset",self.m)
        print("vocab_size_predefined",self.alphabet_size)
        print(self.data_source)
        print(self.data_source2)
       
        
            
        if self.m=="adi":
            df1 = pd.read_csv(self.data_source, delimiter='\t',header=None, names=["text_id","text"])
            df1.replace(np.nan,'NIL', inplace=True)
            self.sentences = df1.text.values
            df2 = pd.read_csv(self.data_labels, delimiter='\t',header=None, names=["text_id","labels"])
            df2.replace(np.nan,'NIL', inplace=True)
            self.labels = df2.labels.values
            
            df3 = pd.read_csv(self.data_source1, delimiter='\t',header=None, names=["text_id","text"])
            df3.replace(np.nan,'NIL', inplace=True)
            self.sentences1 = df3.text.values
            df4 = pd.read_csv(self.data_labels1, delimiter='\t',header=None, names=["text_id","labels"])
            df4.replace(np.nan,'NIL', inplace=True)
            self.labels1 = df4.labels.values
            
            df5 = pd.read_csv(self.data_source2, delimiter='\t',header=None, names=["text_id","text"])
            df5.replace(np.nan,'NIL', inplace=True)
            self.sentences2 = df5.text.values
            df6 = pd.read_csv(self.data_labels2, delimiter='\t',header=None, names=["text_id","labels"])
            df6.replace(np.nan,'NIL', inplace=True)
            self.labels2 = df6.labels.values
        if self.m=="nadi":
            print("NADI")
            df = pd.read_csv(self.data_source,delimiter='\t')
            df.replace(np.nan,'NIL', inplace=True)
            self.sentences= df["#2_content"].values
            self.labels = df["#3_label"].values
            
            df1 = pd.read_csv(self.data_source1,delimiter='\t')
            df1.replace(np.nan,'NIL', inplace=True)
            self.sentences1= df1["#2_content"].values
            self.labels1 = df1["#3_label"].values
            
            #df2 = pd.read_csv(self.data_source2,delimiter='\t',names=["#1_id","#2_content","#3_label"])
            df2 = pd.read_csv(self.data_source2,delimiter='\t')
            df2.replace(np.nan,'NIL', inplace=True)
            self.sentences2= df2["#2_content"].values
            #self.labels2 = df2["#3_label"].values
        model_type="unigram"    
        print("Perform vocab size experiments is %s"%(self.perform))
        if not self.perform:
            #model_type="unigram"
          
            if os.path.exists("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size)):
                print("Sentence Piece Model found..","%s_%s_%s.model"%(self.m,model_type,self.alphabet_size))
                
            
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size))
            else:
                print("Training Sentence Piece Model with %s"%(model_type))
                raw = " ".join(self.sentences)
                with open("%s_train_raw.txt"%(self.m),'w') as f:
                    for i in self.sentences:
                        f.write(str(i))
                        f.write(str("\n"))
                spm_args = "--input=%s_train_raw.txt"%(self.m)
                spm_args += " --model_prefix=%s_%s_%s"%(self.m,model_type,self.alphabet_size)
                spm_args += " --vocab_size=%s"%(self.alphabet_size)
                spm_args += " --model_type=unigram"
                spm.SentencePieceTrainer.Train(spm_args)
                #self.alphabet_size = 200
            
                self.sp = spm.SentencePieceProcessor()
                self.sp.Load("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size))
        else:
            self.merges=[]
            self.vocabs=[]
            #self.vocabs=[]
            #self.merge_size=self.opt_merge_size(self.sentences)#actual optimum merge_size (0.4*vocab_size_word_level)
            #self.list_alphabet=self.min_vocab(self.sentences) #character_vocab
            self.list_alphabet=[3045]
            #self.list_alphabet=[20045]
            #self.list_alphabet=[7045]
            #self.list_alphabet=[5045]
            #merge=0
            for alphabet_size in self.list_alphabet:
                self.alphabet_size=alphabet_size
                
                #while merge<self.merge_size:
                    
                print("Running alphabet_size %s"%(self.alphabet_size))
                vocab_file="%s_%s_%s.vocab"%(self.m,model_type,self.alphabet_size)
                if os.path.exists("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size)):
                    print("Sentence Piece Model found..")
                    self.sp = spm.SentencePieceProcessor()
                    self.sp.Load("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size))
                    
                    merge=self.merge_op(vocab_file)
                else:
                    print("Training Sentence Piece Model..")
                    raw = " ".join(self.sentences)
                    with open("%s_train_raw.txt"%(self.m),'w') as f:
                        for i in self.sentences:
                            f.write(str(i))
                            f.write(str("\n"))
                    spm_args = "--input=%s_train_raw.txt"%(self.m)
                    spm_args += " --model_prefix=%s_%s_%s"%(self.m,model_type,self.alphabet_size)
                    spm_args += " --vocab_size=%s"%(self.alphabet_size)
                    spm_args += " --model_type=unigram"
                    spm.SentencePieceTrainer.Train(spm_args)
                    self.sp = spm.SentencePieceProcessor()
                    self.sp.Load("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size))
                        
                        #self.alphabet_size = 200
                    
                        
                        #merge=self.merge_op(vocab_file)
                    #self.merges.append(merge)
                self.vocabs.append(self.alphabet_size)
                print(self.vocabs)
                    #print("merge",merge)
                    #print("merge_size",self.merge_size)
                    # if merge>=self.merge_size:
                    #     print("alphabet_size exceeded merge_size",self.alphabet_size)
                    #     self.sp = spm.SentencePieceProcessor()
                    #     self.sp.Load("%s_%s_%s.model"%(self.m,model_type,self.alphabet_size))
                    # else:
                    #     if self.alphabet_size<=1000:
                    #         self.alphabet_size+=100
                    #     elif self.alphabet_size<=10000:
                    #         self.alphabet_size+=1000
                    #     elif self.alphabet_size<=100000:
                    #         self.alphabet_size+=10000
                    #     if self.alphabet_size>48969:
                    #         print("##alphabet_size>48969")
                    #         merge=self.merge_size
                            
                    
            
        
        #alphabet = sorted(list(set(raw)))
        #self.alphabet = alphabet
        
        #self.alphabet_size = len(self.alphabet)
        #self.dict = {}  # Maps each character to an integer
        #self.no_of_classes = num_of_classes
        #for idx, char in enumerate(self.alphabet):
            #self.dict[char] = idx + 1
        #print(self.dict)
        #print(len(self.dict))
        #print("Data loaded from " + self.data_source)
    
    def convert_labels_TestA(self):
        print("%%To categorical")
        y_train = list(self.labels)
        le.fit(y_train)
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(le_name_mapping)
        print("# of classes computed",len(le.classes_))
        y_train=le.transform(y_train)
        self.labels_train = to_categorical(y_train)
        #self.labels_train = y_train
        
        y_valid= list(self.labels1)
        y_valid=le.transform(y_valid)
        self.labels_valid = to_categorical(y_valid)
        
        #y_test= list(self.labels2)
        #y_test=le.transform(y_test)
        #self.labels_test = to_categorical(y_test)
        #self.labels_test = y_test
        #self.classes=len(le.classes_)
        
    def convert_labels_TestB(self):
        class_map={'algeria': 0, 'bahrain': 1, 'egypt': 2, 'iraq': 3, 'jordan': 4, 'ksa': 5, 'kuwait': 6, 'lebanon': 7, 'libya': 8, 'morocco': 9, 'oman': 10, 'palestine': 11, 'qatar': 12, 'sudan': 13, 'syria': 14, 'tunisia': 15, 'uae': 16, 'yemen': 17}
        print("%%To categorical")
        y_train = list(self.labels)
        y_train=[class_map[k] for k in y_train]
        #le.fit(y_train)
        #le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        #print(le_name_mapping)
        print("# of classes computed",len(list(set(y_train))))
        #y_train=le.transform(y_train)
        self.labels_train = to_categorical(y_train)
        #self.labels_train = y_train
        
        y_valid= list(self.labels1)
        y_valid=[class_map[k] for k in y_valid]
        #y_valid=le.transform(y_valid)
        self.labels_valid = to_categorical(y_valid)
        
        #y_test= list(self.labels2)
        #y_test=le.transform(y_test)
        #y_test=[class_map[k] for k in y_test]
        #self.labels_test = to_categorical(y_test)
        #self.labels_test = y_test
        #self.classes=len(list(set(y_train)))
        
        
        
        
        
        # data = []
        # with open(self.data_source, 'r', encoding='utf-8') as f:
        #     rdr = csv.reader(f, delimiter=',', quotechar='"')
        #     for row in rdr:
        #         txt = ""
        #         for s in row[1:]:
        #             txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
        #         data.append((int(row[0]), txt))  # format: (label, text)
        # self.data = np.array(data)
        # print("Data loaded from " + self.data_source)
    def get_all_data(self):
        print("Get Data!!!")
        
        # Convert string to index
        self.sequences = [self.sp.EncodeAsIds(k) for k in self.sentences]
        # Padding
        self.data_padded = pad_sequences(self.sequences, maxlen=self.length, padding='post',truncating='post')
        # Convert to numpy array
        self.input_data = np.array(self.data_padded, dtype='float32')
        
        # Convert string to index
        self.sequences_valid =  [self.sp.EncodeAsIds(k) for k in self.sentences1]
        # Padding
        self.data_padded_valid = pad_sequences(self.sequences_valid, maxlen=self.length, padding='post',truncating='post')
        # Convert to numpy array
        self.input_data_valid= np.array(self.data_padded_valid, dtype='float32')
        
        # Convert string to index
        self.sequences_test =  [self.sp.EncodeAsIds(k) for k in self.sentences2]
        # Padding
        self.data_padded_test = pad_sequences(self.sequences_test, maxlen=self.length, padding='post',truncating='post')
        # Convert to numpy array
        self.input_data_test = np.array(self.data_padded_test, dtype='float32')
        
        if self.perform:
            return self.input_data,self.labels_train,self.input_data_valid,self.labels_valid,self.input_data_test,self.alphabet_size,self.m,self.vocabs
        else:
            return self.input_data,self.labels_train,self.input_data_valid,self.labels_valid,self.input_data_test,self.alphabet_size,self.m
        
    # def get_all_data(self):
    #     #tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    #     #tk.fit_on_texts(self.sentences)
    #     # Use char_dict to replace the tk.word_index
    #     #tk.word_index = self.dict.copy()
    #     # Add 'UNK' to the vocabulary
    #     #tk.word_index[tk.oov_token] = max(self.dict.values()) + 1
    #     # Convert string to index
    #     self.sequences = [self.sp.EncodeAsIds(k) for k in self.sentences]
    #     # Padding
    #     self.data_padded = pad_sequences(self.sequences, maxlen=self.length, padding='post',truncating='post')
    #     # Convert to numpy array
    #     #self.input_data = np.array(self.data_padded, dtype='float32')
        
    #     # Convert string to index
    #     self.sequences_test =  [self.sp.EncodeAsIds(k) for k in self.sentences1]
    #     # Padding
    #     self.data_padded_test = pad_sequences(self.sequences_test, maxlen=self.length, padding='post',truncating='post')
    #     # Convert to numpy array
    #     self.input_data_test = np.array(self.data_padded_test, dtype='float32')
    #     #word_to_id=self.dict
    #     #id_to_word = {value:key for key,value in word_to_id.items()}
    #     #print(x_test)
    #     #x_train_pad = pad_sequences(x_train,maxlen=max_len)
    #     #x_test_pad = pad_sequences(x_test,maxlen=max_len)
 
    #     batch_size=128
    #     train_data = data_utils.TensorDataset(torch.from_numpy(self.data_padded).type(torch.LongTensor),torch.from_numpy(self.labels_train).type(torch.LongTensor))
    #     train_loader = data_utils.DataLoader(train_data,batch_size=batch_size,drop_last=True)
    #     #return train_loader,x_train,x_test,x_test_pad,word_to_id,y_train,y_test,x_train_pad
    #     test_data = data_utils.TensorDataset(torch.from_numpy(self.data_padded_test).type(torch.LongTensor),torch.from_numpy(self.labels_test).type(torch.LongTensor))
    #     test_loader = data_utils.DataLoader(test_data,batch_size=8,drop_last=True)
    #     return train_loader,self.sentences,self.sentences1,self.data_padded_test,self.labels_train,self.labels_test,self.data_padded ,self.alphabet_size,self.m,test_loader
    

    # def get_all_data(self):
    #     """
    #     Return all loaded data from data variable.

    #     Returns:
    #         (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

    #     """
    #     data_size = len(self.data)
    #     start_index = 0
    #     end_index = data_size
    #     batch_texts = self.data[start_index:end_index]
    #     batch_indices = []
    #     one_hot = np.eye(self.no_of_classes, dtype='int64')
    #     classes = []
    #     for c, s in batch_texts:
    #         batch_indices.append(self.str_to_indexes(s))
    #         c = int(c) - 1
    #         classes.append(one_hot[c])
    #     return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.
        
        Args:
            s (str): String to be converted to indexes

        Returns:
            str2idx (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx
