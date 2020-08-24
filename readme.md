# Political Ideology Text Classification
## Introduction
This is the code I generated for my final project at University. The objective of the project was to use Machine Learning models and feature extraction techniques to predict the political ideology of texts.

## Scripts

There are several scripts with different functions:

**The following scripts are used to generate datasets. Those datasets might be needed by other scripts to work.**

**CreateDatasetFunctions.py**
	This file mixes ideology files of one congress with speeches to generate a document with the format DWNominate1|DWnominate2|Text.
	For this script, we need several paths as inputs:
		- path_speeches_map: path were the file X_SpeakerMap.txt is stored. X is the number of the congress for which we want to generate the dataset.
		- path_dwnominate: path were the file HS_X_members.csv is stored.
		- path_speeches: path were the file speeches_X.txt is stored.
		We also have to provide two paths that will be were the new dataset will be stored:
		- path_new_speeches_nonames: path were we want to store the new file with format DWNominate1|DWnominate2|Text. This text will contain no names, punctuation marks and digits.
		- path_new_speeches_nonames_aux: path were we want to store an auxiliar file. This file has the same format than previously. The difference between both files is that texts from this file contain names, punctuation marks and digits.
	To generate the dataset, call the function generateDataset, using all the paths specified before. When calling it, set the parameter "nonames" to true (default value).

**ExtractFeatures.py**
	Generates two new datasets containing stylistic features (avg word length, number of characters, etc). Previous to generating these documents, two datasets must have been created with CreateDatasetFunctions.py. To run it, the only thing that must to be done is to set the following paths:
	 - dataset_path_train: path to the training dataset (generated with CreateDatasetFunctions.py) (this should be the auxiliar file "path_new_speeches_nonames_aux" in CreateDatasetFunctions.py)
	 - dataset_path_test: path to the test set (generated with CreateDatasetFunctions.py)(this should be the auxiliar file "path_new_speeches_nonames_aux" in CreateDatasetFunctions.py)
	 - new_train: path for one of the  new datasets
	 - new_test: path for the other new dataset
	When those paths are specified, just run the script.

**ExtractPOS.py**
	Generates two new datasets containing POS tags frequency of texts. Previous to generating these documents, two datasets must have been created with CreateDatasetFunctions.py. To run it, the only thing that must to be done is to set the following paths:
	 - texts_train_path: path to the training dataset (generated with CreateDatasetFunctions.py)(this should be the auxiliar file "path_new_speeches_nonames_aux" in CreateDatasetFunctions.py)
	 - texts_test_path: path to the test set (generated with CreateDatasetFunctions.py)(this should be the auxiliar file "path_new_speeches_nonames_aux" in CreateDatasetFunctions.py)
	 - new_train_pos: path for one of the new datasets
	 - new_test_pos: path for the other new dataset
	When those paths are specified, just run the script.

**ExtractPOSngrams.py**
	Similar to ExtractPOS.py. In this case, POS ngram frequencies are generated. There are two more parameters:
	n: the "n" from "n-gram". If n = 2, bigrams are genreated. If n=3, trigrams, etc.
	nFeatures: The script will generate only the "nFeatures" most common ngrams.

**GenerateNGramsDataset.py**
	Same than ExtractPOSngrams.py. The difference is that, in this case, character n-grams are generated.


**The rest of the scripts were created for training, testing or analysis of the data.**

**LDA.py** 

Script used for the LDA analysis. Gensim library is required. 
	The first function used should be:
		- initialize: generates the variables dictionary, train_corpus and test_corpus (they are needed for other functions). This can be very slow due to the stemming in the "preprocess" function. To avoid this, substitute the line:
		
			result.append(stemmer.stem(w.lower()))
		with
			result.append(w.lower())
			
Once we have those variables, we can generate lda models by using:
- lda: this function returns an LDA model trained with (train_corpus) and with n_topics as number of topics. We also have to specify the dictionary (generated with "initialize"). If showWords == True, words belonging to that topic will be shown. If saveModel == True, the model will be saved in the path specified in the following code line:

		if saveModel:
        		lda_model.save('directory/lda_model_'+str(n_topics))

When we have stored models, we can load them by doing: model = gensim.models.LdaModel.load(path)

- getBestN: plots the results of "coherence" o "perplexity", depending on what is specified in "metrics". Coherence is much faster. In this case, the parameter n_topics is a list with all the topics we want to evaluate. n_topics = [10,12,14,16] is an example. Note that those models must be stored for this function to work.

- getCorpusTopics: outputs two lists with topics for both classes. This list are used as an input to plotTopics.

- plotTopics: plots the topic distribution for both classes. You also have to specify the number of topics used when training the model.

- plotTopicsNoClass: same than plotTopics but the distribution does not distinguish between classes.

- extractFeatures: generates a new dataset containing, for each text in the dataset, the probability of that text to contain each topic. You have to specify:
    		- corpus: train_corpus or test_corpus (generated with "initialize"). This is the congress from which you want to generate the dataset.
    		- model: an lda model.
    		- path: path were you want to store the new dataset.
    		- nTopics: number of topics the model was trained with.


**CreateMatrix.py:** 
Contains only one function with multiple inputs. This function will train a Naive Bayes and Logistic regression models and output: accuracy, precision, recall, most important words and Accuracy-dwnominate curve (for each of them). 
	It requires several paths:
		new_train, new_test: datasets generated with "ExtractFeatures.py".
		texts_train_path, texts_test_path: datasets generated with "CreateDatasetFunctions.py"
		train_pos, test_pos: datasets generated with ExtractPOS.py
		train_pos_gram, test_pos_gram: datasets generated with ExtractPOSngrams.py.
		new_train_ngrams, new_test_ngrams: dataset generated with GenerateNGramsDataset.py.
		train_lda, test_lda: datasets generated with LDA.extractFeatures.

Note that if those features are not going to be used (as it will be explained shortly), those files do not have to exist. You are not obliged to generate all datasets if you are not going to use them.
The other parameters of the function are:
	
thresholdPos = 0.2, thresholdNeg = -0.2, thresholdPosTest = 0.2, thresholdNegTest = -0.2, subsample=False, removeCenter=True, BoW = True, charNgrams = False, POS = False, features = False, POSgrams = False, tfidf = False, binary = False, lda = False


- removeCenter: if True, moderate texts will be removed from training.

- thresholdPos, thresholdNeg, thresholdPosTest, thresholdNegTest: only used if removeCenter == True. They specify the negative and positive thresholds for the training set and test set (thresholdPos and thresholdNeg for the train set and thresholdPosTest, thresholdNegTest for the test set). For example, if thresholdPos and thresholdNeg are set to 0.2 and -0.2, texts in that range will not be considered to train the model.

- subsample: allway should be set to False.

- BoW, charNgrams, POS, features, POSgrams, tfidf, lda: when True, those features will be added to the train and test matrices. At least one of the must be set to True.

- binary: if True, feature values will be set to 1 if they are different than 0.

**TrainingCombinations.py.** Used to train with many congresses at the same time and test on one congress. This uses only the term frequency BoW. The only parameters that must be set are:
	- train: list of congresses we want for training. Those datasets must have been created with CreateDatasetFunctions.
	- test_n: number of the test congress.

**Eval.py** provides several help functions.
	- readDataset: used to read datasets generated with CreateDatasetFunctions.
	- Accuracy, precision, recall: return the evaluation metrics using the predictions and actual labels.
	- histogram: plots the accuracy-dwnominate curve (called by CreateMatrix.TestModel)

