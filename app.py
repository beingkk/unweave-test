"""
Simple pipeline that loads text data, calculates sentence embeddings and saves the embedding array
"""
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import Iterator
import numpy as np
from numpy.typing import ArrayLike
import typer
from toolz.functoolz import pipe
import logging
logging.getLogger().setLevel(logging.INFO)
from pathlib import Path

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
OUTPUTS_PATH = Path("outputs/")
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
# Default run mode (True for faster test mode)
DEFAULT_TEST = True

def load_tripadvisor_data(test: bool = DEFAULT_TEST) -> Iterator[str]:
    """
    Loads test data, sourced from kaggle:
    https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
    """
    nrows = None if not test else 50
    filepath = 'tripadvisor_hotel_reviews.csv'
    logging.info(f'Loading data from {filepath}')    
    return pd.read_csv(filepath, nrows=nrows)["Review"].to_list()


def calculate_embeddings(sentences: Iterator[str]) -> ArrayLike:
    """Uses sentence transformers to calculated emebeddings"""
    model = SentenceTransformer(MODEL_NAME)
    logging.info(f'Calculating {len(sentences)} embeddings')
    return model.encode(sentences)


def save_outputs(output_array: ArrayLike, output_filename: str =  'embeddings.npy'):
    """Saves a numpy array to a local binary file"""    
    logging.info(f'Saving embeddings to {output_filename}')
    np.save(OUTPUTS_PATH / output_filename, output_array)

    
def main(test: bool = DEFAULT_TEST):
    """Complete pipeline that loads text data, calculates sentence embeddings and saves the array"""
    pipe(
        test,
        load_tripadvisor_data,
        calculate_embeddings,
        save_outputs
    )

if __name__ == '__main__':
    typer.run(main)
