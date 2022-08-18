# NLP_DI_NADI

This repository contains the scripts for NADI shared task 2022 [https://nadi.dlnlp.ai/]. 

The folder *UnigramCNN* includes the scripts for training and testing the Unigram CNN model proposed by Kim et al. We adapted the code from [https://github.com/mhjabreel/CharCnn_Keras].  
Instead of character vocabulary we used unigram subword tokenizations. The validation part is not included. The dev data provided in the shared task was used to fine tune model: finding the number of epochs using early stopping and finding the optimal vocabulary sizes for the Unigram model.  

The colab scripts contains the training and testing with two pre-trained models:

*AraBERT*: [https://huggingface.co/aubmindlab/bert-base-arabert]

*Multi-dialect-Arabic-BERT*: [https://huggingface.co/bashar-talafha/multi-dialect-bert-base-arabic] <br />


**Requirements** <br />
Keras <br />
tensorflow <br />
scipy <br />
sentencepiece [https://github.com/google/sentencepiece] <br />
nltk <br />
transformers <br />

**Congfigurations** <br />
The model configurations and path files are specified in config file for UnigramCNN

**Execution** <br />
Run the command : python main_vocab.py --model kim



