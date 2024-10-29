"""
Utility functions for the Montreal fire tasks

"""


def calculate_yearly_sum_stats_for_months(df, months=None, cutoff_year=None):
    """
    Calculates the sum over some months, each year, and then returns the average.

    Parameters:
    df (pd.DataFrame): The DataFrame with a DateTimeIndex.
    months (list): List of months (1-12) to filter the DataFrame.
    cutoff_year: int
        Will only consider data older than this year

    Returns:
    float: The average yearly sum for the specified months.
    """
    if months is None:
        months = list(range(1, 13))

    filtered_df = df.loc[
        (df.index.month.isin(months))
        & (df.index.year < (cutoff_year if cutoff_year else 99999))
    ]
    yearly_sums = filtered_df.groupby(filtered_df.index.year).sum()
    return {
        "min": yearly_sums.min(),
        "max": yearly_sums.max(),
        "mean": yearly_sums.mean(),
        "sum": yearly_sums.sum(),
        "values": {year: yearly_sums.loc[year] for year in yearly_sums.index},
        "n": len(yearly_sums),
    }
