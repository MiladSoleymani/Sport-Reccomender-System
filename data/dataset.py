from data.helper_funcs import (
    set_digit_ratio,
    merge_parameters,
    merge_labels,
    merge,
    clean,
)

import pandas as pd
import numpy as np


class ATHLXDataset(object):
    def __init__(
        self,
        parameters_data_path: str,
        labels_data_path: str,
        num_top_spotrts: int,
        type_use: str = "train",
        transforms=None,
    ) -> None:
        self.parameters_data_path = parameters_data_path
        self.labels_data_path = labels_data_path
        self.num_top_spotrts = num_top_spotrts
        self.type_use = type_use
        self.transforms = transforms

        self.data = self.prepare_data()

    def prepare_data(self) -> pd.DataFrame:
        # merge .json filed into one .csv
        parameters_data = merge_parameters(folder_path=self.parameters_data_path)
        labels_data = merge_labels(
            folder_path=self.labels_data_path, num_top_spotrts=self.num_top_spotrts
        )
        self.labels_data_keys = list(labels_data.keys())
        self.labels_data_keys.remove("id")
        self.labels_data_keys = [
            key for key in self.labels_data_keys if key not in ["id"]
        ]

        # Drop columns containing NaN values
        parameters_data = clean(parameters_data)
        self.parameters_data_keys = list(parameters_data.keys())
        self.parameters_data_keys = [
            key
            for key in self.parameters_data_keys
            if key not in ["id", "Athlete", "Date_created"]
        ]

        # 'inner' means it will keep only the common rows
        data = merge(parameters_data, labels_data)

        # Replace 'E' with '3' in the 'Digit_Ratio' column
        data = set_digit_ratio(data)

        return data

    def get_parameters(self) -> str:
        return "\n\t\t\t".join(list(map(str, self.data.keys())))

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self):
        x = self.data[self.parameters_data_keys]
        y = self.data[self.labels_data_keys]

        x = np.array(x.values)
        y = np.array(y.values)

        # if self.transforms is not None:
        #     data = self.transforms(data)

        return x, y

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset              : {self.__class__.__name__}\n"
            + f"# num_data           :  {self.__len__()}\n"
            + f"# parameters         :  {self.get_parameters()}\n"
            + "#############################################################\n"
        )
