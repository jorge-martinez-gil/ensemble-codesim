# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import os
import subprocess
import os
import sys
import time
from difflib import SequenceMatcher
import re


# Compile and execute Java code
def execute_java_code(code, input_data):

    # print(code)

    pattern = r'class\s+(\w+)\s*{'

    # Search for the pattern in the Java code
    try:
        match = re.search(pattern, code)
        class_name = match.group(1)
    except:
        return 999

    java_filename = class_name + ".java"
    class_filename = class_name + ".class"
    with open(java_filename, "w") as java_file:
        java_file.write(code)
    
    subprocess.run(["javac", java_filename], check=True)
    
    process = subprocess.Popen(
        ["java", class_name],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate(input=input_data)
    
    os.remove(java_filename)  # Clean up the temporary Java file
    os.remove(class_filename)  # Clean up the temporary Java file
    
    return stdout.strip()

def similarity2(code1, code2):
    input_data = ""
    output1 = execute_java_code(code1, input_data)
    output2 = execute_java_code(code2, input_data)
    # Calculate the similarity ratio
    try:
        similarity_ratio = SequenceMatcher(None, output1, output2).ratio()
    except:
        similarity_ratio = 0
    return similarity_ratio

def similarity(code1, code2):
    return 0