import os
import pandas as pd


class FNSPIDUnivariateDataLoader:
    """
    A class to handle the FNSPID data for the benchmarking project.
    Parameters:
    ----------
    news_data_dir : str
        Source: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
        The directory containing the news data, should have subdirectories for each
        stock symbol, with each subdirectory containing per-day CSV files of news data.
        Each CSV file should have the following columns:
        - Date: The publication date of the news article.
        - Article_title: The title of the news article.
        - Url: The URL of the news article.
        - Article: The full text of the news article.
        - Lsa_summary: The Latent Semantic Analysis (LSA) summary of the news article.
        - Luhn_summary: The Luhn summary of the news article.
        - Textrank_summary: The TextRank summary of the news article.
        - Lexrank_summary: The LexRank summary of the news article.
    price_data_dir : str
        Source: https://github.com/Zdong104/FNSPID_Financial_News_Dataset
        The directory containing the price data. Should have a single CSV file for each
        stock symbol containing the price data.
        Each CSV file should have the following columns:
        - date: The date of the price data.
        - volume: The trading volume.
        - open: The opening price.
        - high: The highest price.
        - low: The lowest price.
        - close: The closing price.
        - adj close: The adjusted closing price.
    """

    def __init__(
        self,
        news_data_dir="/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/fnspid/news_by_stock_by_date",
        price_data_dir="/home/toolkit/starcaster/research-starcaster/benchmark/benchmark/data/fnspid/full_history",
    ):
        self.news_data_dir = news_data_dir
        self.price_data_dir = price_data_dir

    def load_news_data(self, stock_symbol, start_date, end_date):
        """
        Load news data for a specific stock symbol and date range.
        Parameters:
        ----------
        stock_symbol : str
            The stock symbol to load news data for.
        start_date : str
            The start date of the date range (format: 'YYYY-MM-DD').
        end_date : str
            The end date of the date range (format: 'YYYY-MM-DD').
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the news data for the stock symbol and date range.
            Columns:
            - Date: The publication date of the news article.
            - Article_title: The title of the news article.
            - Url: The URL of the news article.
            - Article: The full text of the news article.
            - Lsa_summary: The Latent Semantic Analysis (LSA) summary of the news article.
            - Luhn_summary: The Luhn summary of the news article.
            - Textrank_summary: The TextRank summary of the news article.
            - Lexrank_summary: The LexRank summary of the news article.
        """
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        # Directory for the specific stock symbol
        stock_dir = os.path.join(self.news_data_dir, stock_symbol)

        if not os.path.exists(stock_dir):
            raise ValueError(
                f"Directory for stock symbol '{stock_symbol}' does not exist."
            )

        # Collect all relevant CSV files
        data_frames = []
        for file_name in os.listdir(stock_dir):
            file_date = pd.to_datetime(file_name.split(".csv")[0]).date()
            if start_date <= file_date <= end_date:
                file_path = os.path.join(stock_dir, file_name)
                df = pd.read_csv(file_path, parse_dates=["Date"])
                data_frames.append(df)

        # Concatenate all collected DataFrames
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
        else:
            combined_df = (
                pd.DataFrame()
            )  # Return an empty DataFrame if no files are found

        return combined_df

    def load_price_data(self, stock_symbol, start_date, end_date):
        """
        Load price data for a specific stock symbol and date range.
        Parameters:
        ----------
        stock_symbol : str
            The stock symbol to load news data for.
        start_date : str
            The start date of the date range (format: 'YYYY-MM-DD').
        end_date : str
            The end date of the date range (format: 'YYYY-MM-DD').
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the price data for the specified stock symbol and date
            range.
            Columns:
            - date: The date of the price data.
            - volume: The trading volume.
            - open: The opening price.
            - high: The highest price.
            - low: The lowest price.
            - close: The closing price.
            - adj close: The adjusted closing price.
        """
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()

        # Load the price data CSV file
        file_path = os.path.join(self.price_data_dir, f"{stock_symbol}.csv")
        df = pd.read_csv(file_path, parse_dates=["date"])

        # Filter the data based on the date range
        mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
        filtered_df = df[mask]

        return filtered_df
