from typing import TypedDict, List
import json

class DataFrameLike(TypedDict):
    columns: List[str]
    index: List[str]
    data: List[List[float]]

def load_data() -> DataFrameLike:
    with open("data.json", "r") as file:
        data: DataFrameLike = json.load(file)

    return data