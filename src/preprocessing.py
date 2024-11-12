import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import serialize_data
import numpy as np

def create_onehot_encoder(categories: list, path: str) -> OneHotEncoder:
    """
    Create and fit a OneHotEncoder with the specified categories, then save it to the given path.

    Parameters:
    - categories: A list of categories for which the encoder will be created.
    - path: The location on the disk where the encoder will be saved.

    Returns:
    - ohe: The fitted OneHotEncoder instance.
    """
    
    if not isinstance(categories, list):
        raise RuntimeError("Fungsi create_onehot_encoder: parameter categories haruslah bertipe list, berisi kategori yang akan dibuat encodernya.")

    if not isinstance(path, str):
        raise RuntimeError("Fungsi create_onehot_encoder: parameter path haruslah bertipe string, berisi lokasi pada disk komputer dimana encoder akan disimpan.")
    
    ohe = OneHotEncoder(sparse_output=False)  

    ohe.fit(np.array(categories).reshape(-1, 1))

    serialize_data(ohe, path)

    print(f"Kategori yang telah dipelajari adalah {ohe.categories_[0].tolist()}")

    return ohe