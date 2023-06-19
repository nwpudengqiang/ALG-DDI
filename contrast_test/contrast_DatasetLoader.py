from functools import lru_cache
from chemicalx.data import DatasetLoader
from typing import Dict


import json
import numpy as np
import pandas as pd

from chemicalx.data.contextfeatureset import ContextFeatureSet
from chemicalx.data.drugfeatureset import DrugFeatureSet
from chemicalx.data.labeledtriples import LabeledTriples
import torch


class myDatasetLoader(DatasetLoader):
    """A dataset loader for local data."""
    # drugset_name: ClassVar[str] = "drug_set.json"
    # contexts_name: ClassVar[str] = "context_set.json"
    # labels_name: ClassVar[str] = "labeled_triples.csv"

    def __init__(self, directory):
        """Instantiate the dataset loader.

        """
        self.directory = directory
        # self.drug_set = self.directory.joinpath("drug_set.json")
        # self.contexts_path = self.directory.joinpath("context_set.json")
        # self.labels_path = self.directory.joinpath("labeled_triples.csv")

    def generate_path(self, file_name: str) -> str:
        """Generate a complete url for a dataset file.

        :param file_name: Name of the data file.
        :returns: The complete url to the dataset.
        """
        data_path = "/".join([self.directory, file_name])
        return data_path

    def load_raw_json_data(self, path: str) -> Dict:
        """Load a raw JSON dataset at the given path.

        :param path: The path to the JSON file.
        :returns: A dictionary with the data.
        """
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return raw_data

    def load_raw_csv_data(self, path: str) -> pd.DataFrame:
        """Load a CSV dataset at the given path.

        :param path: The path to the triples CSV file.
        :returns: A pandas DataFrame with the data.
        """

        types = {"drug_1": str, "drug_2": str, "context": str, "label": float}
        raw_data = pd.read_csv(path, encoding="utf8", sep=",", dtype=types)
        return raw_data

    @lru_cache(maxsize=1)  # noqa: B019
    def get_context_features(self) -> ContextFeatureSet:
        """Get the context feature set."""
        path = self.generate_path("context_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {k: torch.FloatTensor(np.array(v).reshape(1, -1)) for k, v in raw_data.items()}
        return ContextFeatureSet(raw_data)

    @lru_cache(maxsize=1)  # noqa: B019
    def get_drug_features(self) -> DrugFeatureSet:
        """Get the drug feature set."""
        path = self.generate_path("drug_set.json")
        raw_data = self.load_raw_json_data(path)
        raw_data = {
            key: {"smiles": value["smiles"], "features": np.array(value["features"]).reshape(1, -1)}
            for key, value in raw_data.items()
        }
        return DrugFeatureSet.from_dict(raw_data)

    @lru_cache(maxsize=1)  # noqa: B019
    def get_labeled_triples(self) -> LabeledTriples:
        """Get the labeled triples file from the storage."""
        path = self.generate_path("labeled_triples.csv")
        df = self.load_raw_csv_data(path)
        return LabeledTriples(df)