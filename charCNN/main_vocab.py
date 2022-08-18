import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.models import load_model
from data_utils import Data

from models.char_cnn_kim import CharCNNKim

import tensorflow.compat.v1 as tf1
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
tf1.flags.DEFINE_string("model", "char_cnn_zhang", "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf1.flags.FLAGS
#FLAGS._parse_flags()
#FLAGS._parse()
import sys
FLAGS(sys.argv)
print("Model:",FLAGS.model)

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))
    # Load training data
    # training_data = Data(data_source=config["data"]["training_data_source"],
    #                      alphabet=config["data"]["alphabet"],
    #                      input_size=config["data"]["input_size"],
    #                      num_of_classes=config["data"]["num_of_classes"])
                         
    #training_data = Data(data_source=config["data"]["training_data_source"],data_labels=config["data"]["training_label_source"],
                         #alphabet=config["data"]["alphabet"],
                         #input_size=config["data"]["input_size"],
                         #num_of_classes=config["data"]["num_of_classes"])
                         
    # training_data = Data(data_source=config["data"]["training_data_source"],data_labels=config["data"]["training_label_source"],data_source2=config["data"]["validation_data_source"],data_labels2=config["data"]["validation_label_source"],alphabet_size=config["data"]"vocab_size"],input_size=config["data"]["input_size"],num_of_classes=config["data"]["num_of_classes"])
    
    training_data = Data(data_source=config["data"]["training_data_source"],data_labels=config["data"]["training_label_source"],
                        data_source1=config["data"]["validation_data_source"],data_labels1=config["data"]["validation_label_source"],data_source2=config["data"]["test_data_source"],data_labels2=config["data"]["test_label_source"],alphabet_size=config["data"]["vocab_size"],input_size=config["data"]["input_size"],num_of_classes=config["data"]["num_of_classes"],perform=True)
                         
                         
    training_data.load_data()
    training_data.convert_labels_TestB()
    training_inputs, training_labels,validation_inputs, validation_labels,test_inputs,alphabet_size,data_type,vocabs = training_data.get_all_data()
    #print("merges",merges)
    print("vocabs",vocabs)
    
    
    ##The code should try different vocab_sizes until the specified merge size is reached,increments + 100, then +1000 and so on
    
    # VISUALIZE
    # viz=True
    # if viz:
    #   plt.plot(vocabs,merges) 
    #   plt.title('Vocab_size Vs No. of BPE Merges')
    #   plt.xlabel('Vocab_size')
    #   plt.ylabel('BPE Merge')
    #   #show plot to user
    #   plt.show()
        
    
    #alphabet_size, starting vocab_size, another parameter the merge_size required as per 0.4*Vocab_size(here vocab_size the unique number
    #of words in training corpus), this is pre-computed for time being and rounded off.
    
    
    
    
    
    #
    #print(len(training_inputs),len(validation_inputs))
    # Load validation data
    #validation_data = Data(data_source=config["data"]["validation_data_source"],data_labels=config["data"]["validation_label_source"],
                           #alphabet=config["data"]["alphabet"],
                           #input_size=config["data"]["input_size"],
                           #num_of_classes=config["data"]["num_of_classes"])
    #validation_data.load_data()
    #validation_data.convert_labels()
    #validation_inputs, validation_labels,alphabet_size,data_type = validation_data.get_all_data()

    # Load model configurations and build model
    if FLAGS.model == "kim":
        acc=[]
        fscores_macro=[]
        fscores_micro=[]
        class_map={'algeria': 0, 'bahrain': 1, 'egypt': 2, 'iraq': 3, 'jordan': 4, 'ksa': 5, 'kuwait': 6, 'lebanon': 7, 'libya': 8, 'morocco': 9, 'oman': 10, 'palestine': 11, 'qatar': 12, 'sudan': 13, 'syria': 14, 'tunisia': 15, 'uae': 16, 'yemen': 17}
        c={}
        for i in class_map:
            c[class_map[i]]=i
        with tf.device("gpu:0"):
            for vocab in vocabs:
                
                alphabet_size=vocab
                print("Training %s vocab_size model"%(alphabet_size))
                training_data = Data(data_source=config["data"]["training_data_source"],data_labels=config["data"]["training_label_source"],data_source1=config["data"]["validation_data_source"],data_labels1=config["data"]["validation_label_source"],data_source2=config["data"]["test_data_source"],data_labels2=config["data"]["test_label_source"],alphabet_size=vocab,input_size=config["data"]["input_size"],num_of_classes=config["data"]["num_of_classes"],perform=False)
                         
                         
                training_data.load_data()
                #training_data.convert_labels_TestA()
                training_data.convert_labels_TestB()
                training_inputs, training_labels,validation_inputs, validation_labels,test_inputs,alphabet,data_type= training_data.get_all_data()
                
                model = CharCNNKim(input_size=config["data"]["input_size"],alphabet_size=alphabet_size,
                                   #alphabet_size=config["data"]["alphabet_size"],
                                   embedding_size=config["char_cnn_kim"]["embedding_size"],
                                   conv_layers=config["char_cnn_kim"]["conv_layers"],
                                   fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                                   num_of_classes=config["data"]["num_of_classes"],
                                   dropout_p=config["char_cnn_kim"]["dropout_p"],
                                   optimizer=config["char_cnn_kim"]["optimizer"],
                                   loss=config["char_cnn_kim"]["loss"])
                
                
                # Train model
                model_type="unigram"
                
                if data_type=="nadi":
                    path="models/nadi/charCNNKim%s_%s%s"%(config["training"]["epochs"],model_type,alphabet_size)
                print("path",path)
                if not os.path.exists(path):
                    print("Training the model and saving at %s"%(path))
                    
                    model.train(data_type,training_inputs=training_inputs,
                                training_labels=training_labels,
                                validation_inputs=validation_inputs,
                                validation_labels=validation_labels,
                                epochs=config["training"]["epochs"],
                                batch_size=config["training"]["batch_size"])
                    print("Training done!")
                    
                    model=load_model(path)
                    model.save(path)
                    print("model saved")
                    #print(model)
                    #print(model.summary())
                    #print("Model loaded")
                    y_pred=model.predict(test_inputs, batch_size=config["training"]["batch_size"], verbose=1)
                    pred_val=np.argmax(y_pred,axis=1)
                    print(pred_val)
                    #in actual scenario, we don't have the act_val for test data
                    #act_val=np.argmax(test_labels,axis=1)
                    #print(act_val)
                    p=[]
                    #a=[]
                    for i,k in enumerate(pred_val):
                        p.append(c[k])
                        #a.append(c[act_val[i]])
                    #path="results/nadi/charCNNKimTestA_%s%s.txt"%(model_type,alphabet_size)
                    path="official_results/nadi/NLP_DI_subtask1_testb_1.txt"
                    with open(path,"w") as f:
                        for r in p:
                            f.write(str(r))
                            f.write("\n")
                    
                    #cm=confusion_matrix(act_val,pred_val)
                    #print(cm)
                    #print(accuracy_score(act_val,pred_val))
                    #print(classification_report(act_val,pred_val))
                    #fscores_macro.append(f1_score(act_val,pred_val, average='macro'))
                    #fscores_micro.append(f1_score(act_val,pred_val, average='micro'))
                    #acc.append(accuracy_score(act_val,pred_val))
                    
                    
                    
                    
                if os.path.exists(path):
                    
                    print("Model exists: Loading..%s"%(path))
                    #print(model.summary())
                    model=load_model(path)
                    #print(model)
                    #print(model.summary())
                    print("Model loaded")
                    y_pred=model.predict(test_inputs, batch_size=config["training"]["batch_size"], verbose=1)
                    pred_val=np.argmax(y_pred,axis=1)
                    #print(pred_val)
                    #act_val=np.argmax(test_labels,axis=1)
                    #print(act_val)
                    p=[]
                    #a=[]
                    for i,k in enumerate(pred_val):
                        p.append(c[k])
                        #a.append(c[act_val[i]])
                    print("Writing the Results...")
                    #'NLP_DI_subtask1_testa_1.txt', then zip
                    #path="results/nadi/charCNNKimTestA_%s%s.txt"%(model_type,alphabet_size)
                    path="official_results/nadi/NLP_DI_subtask1_testb_1.txt"
                    with open(path,"w") as f:
                        for r in p:
                            f.write(str(r))
                            f.write("\n")
                    #path="results/nadi/gold.txt"
                    #with open(path,"w") as f:
                        #for r in a:
                            #f.write(str(r))
                            #f.write("\n")
                    #cm=confusion_matrix(act_val,pred_val)
                    #print(cm)
                    #print(accuracy_score(act_val,pred_val))
                    #print(classification_report(act_val,pred_val))
                    #fscores_macro.append(f1_score(act_val,pred_val, average='macro'))
                    #fscores_micro.append(f1_score(act_val,pred_val, average='micro'))
                    #acc.append(accuracy_score(act_val,pred_val))
                    
            #print("Fscores_Micro##",fscores_micro)
            #print("Fscores_Macro##",fscores_macro)
            #print("Accuracy##",acc)
            
    
            
        
        #model.evaluate(validation_inputs, validation_labels, batch_size=config["training"]["batch_size"], verbose=1)
    # print("Test Predictions")
    # #model.evaluate(validation_inputs, validation_labels, batch_size=config["training"]["batch_size"], verbose=1)
    # y_pred=model.predict(validation_inputs, batch_size=config["training"]["batch_size"], verbose=1)
    # pred_val=np.argmax(y_pred,axis=1)
    # print(pred_val)
    # act_val=np.argmax(validation_labels,axis=1)
    # print(act_val)
    # from sklearn.metrics import confusion_matrix
    # cm=confusion_matrix(act_val,pred_val)
    # print(cm)
    # print(accuracy_score(act_val,pred_val))
    # print(classification_report(act_val,pred_val))
    #model.test(testing_inputs=validation_inputs, testing_labels=validation_labels, batch_size=config["training"]["batch_size"])
