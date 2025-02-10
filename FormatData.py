import pandas as pd



def format_data() -> pd.DataFrame:
    unemployment_data = pd.read_excel(
        io="data/Unemployment.xlsx",
        sheet_name="Data1",
        skiprows=10,
        usecols=[0, 66],
        parse_dates=[0],
        index_col=0
    )

    inflation_data = pd.read_excel(
        io="data/Inflation.xlsx",
        sheet_name="Data1",
        skiprows=10,
        usecols=[0, 9],
        parse_dates=[0],
        index_col=0,
    )

    interest_rate_data = pd.read_excel(
        io="data/InterestRate.xlsx",
        sheet_name="Data",
        skiprows=265,
        usecols=[0, 1],
        parse_dates=[0],
        index_col=0,
    )

    df = pd.concat([unemployment_data, inflation_data, interest_rate_data], axis=1)
    df.columns = ["Unemployment", "Inflation", "InterestRate"]
    df = df[df.index >= "1990-08-01"]

    df_resampled = df.resample('MS').mean()
    df_resampled = df_resampled.interpolate(method='linear')

    df_resampled = df_resampled.ffill().bfill()

    return df_resampled

# formatted_data = format_data()

# def get_dataset() -> pd.DataFrame:
#     return formatted_data