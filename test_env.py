import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

print("Ambiente configurado com sucesso!")

data = load_breast_cancer()
print("Dataset carregado com sucesso!")
print("Número de amostras:", data.data.shape[0])
print("Número de features:", data.data.shape[1])