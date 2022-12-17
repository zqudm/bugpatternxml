#!/usr/bin/env python
# coding: utf-8



import json
from os import path
import pathlib
from pecos.utils.featurization.text.preprocess import Preprocessor
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory
from pecos.utils import smat_util
import pandas as pd 
class CustomPECOS:
    def __init__(self, preprocessor=None, xlinear_model=None, output_items=None):
        self.preprocessor = preprocessor
        self.xlinear_model = xlinear_model
        self.output_items = output_items
        
    @classmethod
    def train(cls, input_text_path, output_text_path):
        """Train a CustomPECOS model
        
        Args: 
            input_text_path (str): Text input file name.            
            output_text_path (str): The file path for output text items.
            vectorizer_config (str): Json_format string for vectorizer config (default None). e.g. {"type": "tfidf", "kwargs": {}}
            
        Returns:
            A CustomPECOS object
        """
        # Obtain X_text, Y
        parsed_result = Preprocessor.load_data_from_pickle(input_text_path, output_text_path)
        Y = parsed_result["label_matrix"]
        corpus = parsed_result["corpus"]

        # Train TF-IDF vectorizer
        preprocessor = Preprocessor.train(corpus, {"type": "tfidf", "kwargs":{}}) 
        X = preprocessor.predict(corpus)   
        
        # Train a XR-Linear model with TF-IDF features
        label_feat = LabelEmbeddingFactory.create(Y, X, method="pifa")
        cluster_chain = Indexer.gen(label_feat)
        xlinear_model = XLinearModel.train(X, Y, C=cluster_chain)
        
        # Load output items
        with open(output_text_path, "r", encoding="utf-8") as f:
            output_items = [q.strip() for q in f]
        
        return cls(preprocessor, xlinear_model, output_items)
    
    def predict(self, corpus):
        """Predict labels for given inputs
        
        Args:
            corpus (list of strings): input strings.
        Returns:
            csr_matrix: predicted label matrix (num_samples x num_labels)
        """
        X = self.preprocessor.predict(corpus)
        Y_pred = self.xlinear_model.predict(X)
        return smat_util.sorted_csr(Y_pred)

    def save(self, model_folder):
        """Save the CustomPECOS model

        Args:
            model_folder (str): folder name to save
        """
        self.preprocessor.save(f"{model_folder}/preprocessor")
        self.xlinear_model.save(f"{model_folder}/xlinear_model")
        with open(f"{model_folder}/output_items.json", "w", encoding="utf-8") as fp:
            json.dump(self.output_items, fp)

    @classmethod
    def load(cls, model_folder):
        """Load the CustomPECOS model

        Args:
            model_folder (str): folder name to load
        Returns:
            CustomPECOS
        """
        preprocessor = Preprocessor.load(f"{model_folder}/preprocessor")
        xlinear_model = XLinearModel.load(f"{model_folder}/xlinear_model")
        with open(f"{model_folder}/output_items.json", "r", encoding="utf-8") as fin:
            output_items = json.load(fin)
        return cls(preprocessor, xlinear_model, output_items)


# ### 4.2.3. Operating the Customized PECOS Model
# 
# With a well-declared model class, the customized PECOS model can be modularized and very convenient to use.

# In[20]:


# Declare the path for model serialization and preprocessor configuration.
model_folder = "./model/pecos-CustomPECOS-model"

# Train and save the trained model
code_train_path = "../dataset1/train.pickle"
code_label_path = "../dataset1/labels.txt"
model = CustomPECOS.train(code_train_path, code_label_path)
#model.save(model_folder)

# Load the trained model and predict
#model = model.load(model_folder)
code_test_path = "../dataset1/test.pickle"
parsed_result = Preprocessor.load_data_from_pickle(code_test_path, code_label_path)
Y_tst = parsed_result["label_matrix"]
corpus = parsed_result["corpus"]


Y_pred = model.predict(corpus)


# In[21]:

metrics_cost = smat_util.Metrics.generate(Y_tst, Y_pred, topk=10)
print("Evaluation Metrics with Cost-sensitive Learning")
print(metrics_cost)

