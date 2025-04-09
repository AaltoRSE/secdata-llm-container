import torch
import numpy as np
import pandas as pd
from transformers import pipeline

def test_basic_environment():
    # Test PyTorch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    
    # Test NumPy
    arr = np.random.rand(3, 3)
    print("\nNumPy array:\n", arr)
    
    # Test Pandas
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    print("\nPandas DataFrame:\n", df)
    
    # Test Transformers
    classifier = pipeline('sentiment-analysis')
    result = classifier("I love using transformers!")
    print("\nTransformers sentiment analysis:", result)

if __name__ == "__main__":
    test_basic_environment() 