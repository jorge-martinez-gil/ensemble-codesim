# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import os
from similarity import *

# Define the path to the IR-Plag-Dataset folder
dataset_path = 'datasets\IR-Plag'

# Loop through each subfolder in the dataset
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        # Find the Java file in the original folder
        original_path = os.path.join(folder_path, 'original')
        java_files = [f for f in os.listdir(original_path) if f.endswith('.java')]
        if len(java_files) == 1:
            java_file = java_files[0]
            with open(os.path.join(original_path, java_file), 'r') as f:
                code1 = f.read()
            # Loop through each subfolder in the plagiarized and non-plagiarized folders
            for subfolder_name in ['plagiarized', 'non-plagiarized']:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    # Loop through each Java file in the subfolder
                    for root, dirs, files in os.walk(subfolder_path):
                        for java_file in files:
                            if java_file.endswith('.java'):
                                with open(os.path.join(root, java_file), 'r') as f:
                                    code2 = f.read()
                                # Calculate the similarity ratio
                                s1 = ast.similarity (code1, code2)
                                s2 = bow.similarity (code1, code2)
                                s3 = codebert.similarity (code1, code2)
                                s4 = comments.similarity (code1, code2)
                                s5 = exe.similarity2 (code1, code2)
                                s6 = fcall.similarity (code1, code2)
                                s7 = fuzz.similarity (code1, code2)
                                s8 = graph.similarity (code1, code2)
                                s9 = hashing.similarity (code1, code2)
                                s10 = image.similarity (code1, code2)
                                s11 = jaccard.similarity (code1, code2)
                                s12 = lcs.similarity (code1, code2)
                                s13 = lev.similarity (code1, code2)
                                s14 = metrics.similarity (code1, code2)
                                s15 = ngrams.similarity (code1, code2)
                                s16 = pdg.similarity (code1, code2)
                                s17 = rk.similarity (code1, code2)
                                s18 = semclone.similarity (code1, code2)
                                s19 = semdiff.similarity (code1, code2)
                                s20 = tdf.similarity (code1, code2)
                                s21 = winn.similarity (code1, code2)

                                if subfolder_name == 'plagiarized':
                                    s0 = 1
                                elif subfolder_name == 'non-plagiarized':
                                    s0 = 0
                    
                                print ('[' + str(s1) + ',' + str(s2) + ',' + str(s3) + ',' + str(s4) + ',' + str(s5) + ',' + str(s6) + ',' + str(s7) + ',' + str(s8) + ',' + str(s9) + ',' + str(s10) + ',' + str(s11) + ',' + str(s12) + ',' + str(s13) + ',' + str(s14) + ',' + str(s15) + ',' + str(s16) + ',' + str(s17) + ',' + str(s18) + ',' + str(s19) + ',' + str(s20) + ',' + str(s21) + ',' + str(s0) + ']', file=open("output-karnalim.txt", "a"))
        else:
            print(f"Error: Found {len(java_files)} Java files in {original_path} for {folder_name}")