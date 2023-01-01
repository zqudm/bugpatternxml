#!/usr/bin/env python3 -u
#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.

import argparse
import os
import pandas as pd
import numpy as np
import scipy.sparse as smat
from pecos.utils import smat_util
from pecos.utils.cli import SubCommand, str2bool
from pecos.utils.featurization.text.vectorizers import Vectorizer, vectorizer_dict



def load_data_from_file(
        label_text_path=None,
        label_path=None,
        output_label_path=None,
    ):
        """Parse a \n-separated text file to a CSR label matrix and a list of text strings.

        Text format for each line:
        <\n-separated label indices><TAB><space-separated text string>
        Example: l_1,..,l_k<TAB>w_1 w_2 ... w_t
            l_k can be one of two format:
                (1) the zero-based index for the t-th relevant label
                (2) double colon separated label index and label relevance
            w_t is the t-th token in the string

        Args:
            label_text_path (str): Path to the label text file.
                The main purpose is to obtain the number of labels. Default: None
            label_path (str): Path to the label file of train or test data.
            return_dict (bool, optional): if True, return the parsed results in a dictionary. Default True

        Returns:
            if return_dict:
                {
                    "label_matrix": (csr_matrix) label matrix with shape (N, L),
                }
            else:
                label_matrix
        """
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"cannot find input label file at {data_path}")
        with open(label_path, "r", encoding="utf-8") as fin:
            label_strings = []
            for line in fin:
                label_strings.append(line.strip())

        def parse_label_strings(label_strings, L):
            rows, cols, vals, rels = [], [], [], []

            # determine if relevance is provided
            has_rel = ":" in label_strings[0]

            for i, label in enumerate(label_strings):
                if has_rel:
                    label_tuples = [tp.split(":") for tp in label.split(",")]
                    label_list = list(map(int, [tp[0] for tp in label_tuples]))
                    # label values are currently not being used.
                    val_list = list(map(float, [tp[1] if tp[1] else 1.0 for tp in label_tuples]))
                    rel_list = list(map(float, [tp[2] for tp in label_tuples]))
                else:
                    label_list = list(map(int, label.split(",")))
                    val_list = [1.0] * len(label_list)
                    rel_list = []

                rows += [i] * len(label_list)
                cols += label_list
                vals += val_list
                rels += rel_list

            Y = smat.csr_matrix(
                (vals, (rows, cols)), shape=(len(label_strings), L), dtype=np.float32
            )
            if has_rel:
                R = smat.csr_matrix(
                    (rels, (rows, cols)), shape=(len(label_strings), L), dtype=np.float32
                )
            else:
                R = None

            return Y, R

        if label_text_path is not None:
            if not os.path.isfile(label_text_path):
                raise FileNotFoundError(f"cannot find label text file at: {label_text_path}")
            # this is used to obtain the total number of labels L to construct Y with a correct shape
            L = sum(1 for line in open(label_text_path, "r", encoding="utf-8") if line)
            label_matrix, label_relevance = parse_label_strings(label_strings, L)
        else:
            label_matrix = None
            label_relevance = None

        #if return_dict:
        #    return {
        #        "label_matrix": label_matrix,
        #    }
        #else:
        #    return label_matrix


        if output_label_path and label_matrix is not None:
            smat_util.save_matrix(output_label_path, label_matrix)


if __name__ == "__main__":

  
# label_text_path=None, label_path=None, output_label_path=None,

  parser = argparse.ArgumentParser()
  parser.add_argument('--lt', required=True)
  parser.add_argument("--lp",required=True)
  parser.add_argument("--output",required=True)

  args = parser.parse_args()

  print(args)
  print(args.lp)
  print(args.lt)
  print(args.output)
  load_data_from_file(args.lt,args.lp,args.output)
  
  
