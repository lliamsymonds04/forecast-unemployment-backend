import datetime

from DataLoader import DataFrameLike

def str_to_datetime(d: str):
    return datetime.datetime.strptime(d, "%Y-%m-%d")


def find_date_index(df: DataFrameLike, date: datetime):
    left = 0
    right = len(df["index"]) - 1
    while left <= right:
        p = (left + right) // 2
        d = str_to_datetime(df["index"][p])
        if d == date:
            return p
        elif d > date:
            right = p - 1
        else:
            left = p + 1

    return -1


def get_date_ranges(df: DataFrameLike, lower: str, upper: str| None = None):
    lower_date = str_to_datetime(lower)
    lower_index = find_date_index(df, lower_date)

    if upper is None:
        return df["index"][lower_index:-1]
    else:
        upper_date = str_to_datetime(upper)
        upper_index = find_date_index(df, upper_date)

    return df["index"][lower_index:upper_index+1]


def get_data_in_range(df: DataFrameLike, lower: str, upper: str | None = None):
    lower_date = str_to_datetime(lower)
    lower_index = find_date_index(df, lower_date)

    if upper is None:
        return df["data"][lower_index:-1]
    else:
        upper_date = str_to_datetime(upper)
        upper_index = find_date_index(df, upper_date)

    return df["data"][lower_index:upper_index+1]