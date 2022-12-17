
# PECOS eXtreme Multi-label Classification: XR-Transformer
from pecos.utils import smat_util
# load training text features
from pecos.utils.featurization.text.preprocess import Preprocessor

from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xtransformer.module import MLProblemWithText


code_train_path = "../dataset1/train.pickle"
code_label_path = "../dataset1/labels.txt"
parsed_result = Preprocessor.load_data_from_pickle(code_train_path, code_label_path)
Y = parsed_result["label_matrix"]
corpus = parsed_result["corpus"]

        # Train TF-IDF vectorizer
preprocessor = Preprocessor.train(corpus, {"type": "tfidf", "kwargs":{}}) 
X = preprocessor.predict(corpus)   
 
#X = smat_util.load_matrix("test/tst-data/xmc/xtransformer/train_feat.npz")
#Y = smat_util.load_matrix("test/tst-data/xmc/xtransformer/train_label.npz")
#text = Preprocessor.load_data_from_file("test/tst-data/xmc/xtransformer/train.txt", text_pos=0)["corpus"]
prob = MLProblemWithText(corpus, Y, X_feat=X)
xtf = XTransformer.train(prob)

#xtf.save("model")
#xtf = XTransformer.load("model")

# P is a csr_matrix with shape=(N, L)
code_test_path = "../dataset1/test.pickle"
parsed_result = Preprocessor.load_data_from_pickle(code_test_path, code_label_path)
Y_tst = parsed_result["label_matrix"]
test_corpus = parsed_result["corpus"]
test_feat = preprocessor.predict(test_corpus)   

#
#
#Y_pred = model.predict(corpus)
#
P = xtf.predict(test_corpus, test_feat)

metric = smat_util.Metrics.generate(Y_tst, P, topk=10)
print(metric)

