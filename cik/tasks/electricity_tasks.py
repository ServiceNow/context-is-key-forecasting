from tactis.gluon.dataset import get_dataset
from gluonts.dataset.util import to_pandas
import numpy as np

from ..base import UnivariateCRPSTask
from ..config import DATA_STORAGE_PATH
from ..utils import get_random_window_univar, datetime_to_str
from ..memorization_mitigation import add_realistic_noise
from . import WeightCluster


class ElectricityIncreaseInPredictionTask(UnivariateCRPSTask):
    """
    A task where the consumption of electricity spikes in prediction part,
    due to a heat wave and people using a lot of air conditioning.
    The spikes should be deducted from the context and reflected in the forecast.
    TODO: A multivariate extension of this task, where weather is another time series

    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.3"  # Modification will trigger re-caching

    def random_instance(self):
        datasets = ["electricity_hourly"]

        # Select a random dataset
        dataset_name = self.random.choice(datasets)
        dataset = get_dataset(dataset_name, regenerate=False, path=DATA_STORAGE_PATH)

        assert len(dataset.train) == len(
            dataset.test
        ), "Train and test sets must contain the same number of time series"

        # Get the dataset metadata
        metadata = dataset.metadata

        # Select a random time series
        ts_index = self.random.choice(len(dataset.train))
        full_series = to_pandas(list(dataset.test)[ts_index])

        # Select a random window
        window = get_random_window_univar(
            full_series,
            prediction_length=metadata.prediction_length,
            history_factor=self.random.randint(3, 7),
            random=self.random,
        )

        # Extract the history and future series
        history_series = window.iloc[: -metadata.prediction_length]
        future_series = window.iloc[-metadata.prediction_length :]

        if dataset_name == "electricity_hourly":
            # Sample a starting point in the first half of the prediction
            future_series.index = future_series.index.to_timestamp()
            # Arbitrary way to select a start date: sort the values of future_series (excluding the last 4 points), pick it from the largest 5 values
            spike_start_point = self.random.choice(
                np.argsort(future_series.values[:-4])[-5:][::-1]
            )
            spike_start_date = future_series.index[spike_start_point]
            spike_duration = self.random.choice(
                [1, 2, 3]
            )  # Arbitrarily picked from 1,2,3
            spike_magnitude = self.random.choice(
                [3, 4, 5]
            )  # Arbitrarily set to twice or thrice the max value in the time series
            # Add spike to the data
            future_series.iloc[
                spike_start_point : spike_start_point + spike_duration
            ] = (spike_magnitude * future_series.iloc[spike_start_point])

            # Convert future index to timestamp for consistency
            history_series.index = history_series.index.to_timestamp()

            # Transform
            history_series = add_realistic_noise(history_series, self.random)
            future_series = add_realistic_noise(future_series, self.random)

            background = f"This is the electricity consumption recorded in Kilowatt (kW) in city A."
            scenario = self.get_scenario(
                spike_start_date, spike_duration, spike_magnitude
            )

        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

        # Instantiate the class variables
        self.past_time = history_series.to_frame()
        self.future_time = future_series.to_frame()
        self.constraints = None
        self.background = background
        self.scenario = scenario

        # ROI metric parameters
        self.region_of_interest = slice(
            spike_start_point, spike_start_point + spike_duration
        )

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        return f"Suppose that there is a heat wave in city A from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'} in city A, leading to excessive use of air conditioning, and {spike_magnitude} times the usual electricity being consumed."


class ElectricityIncreaseInPredictionWithDistractorText(
    ElectricityIncreaseInPredictionTask
):
    """
    ElectricityIncreaseInPredictionTask with 3 different distractors in the context. The model would have to retrieve the right context to succeed in this task.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        relevant_context = f"Suppose that there is a heat wave in city A from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, leading to excessive use of air conditioning, and {spike_magnitude} times the usual electricity being consumed."

        distractor_types = [1, 2, 3]
        distractor_type = self.random.choice(distractor_types)
        if distractor_type == 1:
            distractor_factors = [3, 4, 5, 6, 7, 8]
            distractor_factor = self.random.choice(distractor_factors)
            distractor_text = f"There was a festival in neighbouring cities B and C that resulted in {spike_magnitude+distractor_factor} times the usual electricity being consumed there. But this did not affect electricity consumption in city A."
        elif distractor_type == 2:
            spike_month = spike_start_date.month
            distractor_factors = [3, 4, 5, 6, 7, 8]
            distractor_factor = self.random.choice(distractor_factors)
            distractor_text = f"Historically, over the past 3 years, there have been patterns of increased electricity usage due to extreme cold weather in city A in the month of {spike_month}, decreasing electricity consumption by {spike_magnitude+distractor_factor} times the usual electricity being consumed there. But this year, the cold wave is not expected to happen."  # One concern with this is that both the history and the scenario probably belong to the same month, so this text may not affect the model
        elif distractor_type == 3:
            dip_percentages = [75, 85, 95]
            dip_percentage = self.random.choice(dip_percentages)
            distractor_text = f"A brief technical issue in the electricity grid caused a major dip of {dip_percentage}% in electricity consumption 2 weeks ago. This issue is not expected to happen again this week."

        distractor_context_order = self.random.choice(
            [1, 2]
        )  # Put relevant context before or after the distractor
        if distractor_context_order == 1:
            return " ".join([distractor_text, relevant_context])
        elif distractor_context_order == 2:
            return " ".join([relevant_context, distractor_text])


class ElectricityIncreaseInPredictionWithDistractorWithDates(
    ElectricityIncreaseInPredictionTask
):
    """
    ElectricityIncreaseInPredictionTask with a distractor with the same dates in the context. The model would have to retrieve the right context to succeed in this task.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        distractor_types = [1, 2]
        distractor_type = self.random.choice(distractor_types)

        if distractor_type == 1:
            distractor_factors = [3, 4, 5, 6, 7, 8]
            distractor_factor = self.random.choice(distractor_factors)
            distractor_text = f"There was a festival in neighbouring cities B and C that resulted in {spike_magnitude+distractor_factor} times the usual electricity being consumed there from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}. But this did not affect electricity consumption in city A."
        elif distractor_type == 2:
            dip_percentages = [75, 85, 95]
            dip_percentage = self.random.choice(dip_percentages)
            distractor_text = f"A brief technical issue in the electricity grid in a nearby city caused a major dip of {dip_percentage}% from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}. This issue has affected many nearby cities, but not this city."
        return (
            distractor_text
            + f"Suppose that there is a heat wave in city A from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, leading to excessive use of air conditioning, and {spike_magnitude} times the usual electricity being consumed."
        )


class ElectricityIncreaseInPredictionWithSplitContext(
    ElectricityIncreaseInPredictionTask
):
    """
    ElectricityIncreaseInPredictionTask with a context providing the wrong magnitude of the spike, but correcting it later, providing the wrong magnitude of the spike.
    The model would need to just follow instructions, but it would have to link instructions together to succeed.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        distractor_factors = [3, 4, 5, 6, 7, 8]
        distractor_factor = self.random.choice(distractor_factors)
        return f"Suppose that there is a heat wave in city A from {datetime_to_str(spike_start_date)} for {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, which would typically lead to excessive use of air conditioning, and {spike_magnitude+distractor_factor} times the usual electricity being consumed. But in this case, residents sought to conserve energy and used lesser air conditioning, resulting in excessive usage of only {spike_magnitude} times the usual electricity."


class ShortNewsElectricityIncreaseInPredictionTask(ElectricityIncreaseInPredictionTask):
    """
    A version of the ElectricityIncreaseInPredictionTask where the relevent
    information must be retrieved from within a short news article provided in context.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        # This news article was generated with the assistance of Claude
        scenario = f"A heatwave struck the city, which began on {datetime_to_str(spike_start_date)} and lasted for approximately {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, saw temperatures soar to unprecedented levels. According to the city's electricity provider, power consumption during the peak of the heatwave reached approximately {spike_magnitude} times the typical usage for this time of year."
        return scenario


class MediumNewsElectricityIncreaseInPredictionTask(
    ElectricityIncreaseInPredictionTask
):
    """
    A version of the ElectricityIncreaseInPredictionTask where the relevent
    information must be retrieved from within a medium length news article provided in context.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        # This news article was generated with the assistance of Claude
        scenario = f"A sudden and intense heatwave struck the city, causing a dramatic surge in electricity consumption as residents sought refuge from the scorching temperatures. The extreme weather event, which began on {datetime_to_str(spike_start_date)} and lasted for approximately {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, saw temperatures soar to unprecedented levels. In response, citizens across the metropolitan area turned to their air conditioning units en masse, leading to a significant strain on the local power grid. According to the city's electricity provider, power consumption during the peak of the heatwave reached approximately {spike_magnitude} times the typical usage for this time of year. \nFor now, citizens are encouraged to stay hydrated, check on vulnerable neighbors, and use air conditioning responsibly as the community works together to beat the heat."
        return scenario


class LongNewsElectricityIncreaseInPredictionTask(ElectricityIncreaseInPredictionTask):
    """
    A version of the ElectricityIncreaseInPredictionTask where the relevent
    information must be retrieved from within a long news article provided in context.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov", "c_f"]
    _skills = UnivariateCRPSTask._skills + [
        "instruction following",
        "retrieval: context",
    ]
    __version__ = "0.0.2"  # Modification will trigger re-caching

    def get_scenario(self, spike_start_date, spike_duration, spike_magnitude):
        # This news article was generated with the assistance of Claude
        scenario = f"A sudden and intense heatwave struck the city, causing a dramatic surge in electricity consumption as residents sought refuge from the scorching temperatures. The extreme weather event, which began on {datetime_to_str(spike_start_date)} and lasted for approximately {spike_duration} {'hour' if spike_duration == 1 else 'hours'}, saw temperatures soar to unprecedented levels. In response, citizens across the metropolitan area turned to their air conditioning units en masse, leading to a significant strain on the local power grid.According to the city's electricity provider, power consumption during the peak of the heatwave reached approximately {spike_magnitude} times the typical usage for this time of year. \"We've never seen anything quite like this,\" said Jane Smith, spokesperson for PowerCity Utilities. \"The sudden spike in demand pushed our systems to their limits.\" \nAs the city recovers from this unprecedented power surge, experts are already discussing long-term solutions to manage similar situations in the future. These may include upgrades to the power grid, incentives for energy-efficient appliances, and the development of more robust emergency response protocols. \nFor now, citizens are encouraged to stay hydrated, check on vulnerable neighbors, and use air conditioning responsibly as the community works together to beat the heat."
        return scenario


__TASKS__ = [
    ElectricityIncreaseInPredictionTask,
    ElectricityIncreaseInPredictionWithDistractorText,
    ElectricityIncreaseInPredictionWithDistractorWithDates,
    ElectricityIncreaseInPredictionWithSplitContext,
    ShortNewsElectricityIncreaseInPredictionTask,
    MediumNewsElectricityIncreaseInPredictionTask,
    LongNewsElectricityIncreaseInPredictionTask,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            ElectricityIncreaseInPredictionTask,
        ],
    ),
    WeightCluster(
        weight=1,
        tasks=[
            ElectricityIncreaseInPredictionWithDistractorText,
            ElectricityIncreaseInPredictionWithDistractorWithDates,
            ElectricityIncreaseInPredictionWithSplitContext,
        ],
    ),
    WeightCluster(
        weight=1,
        tasks=[
            ShortNewsElectricityIncreaseInPredictionTask,
            MediumNewsElectricityIncreaseInPredictionTask,
            LongNewsElectricityIncreaseInPredictionTask,
        ],
    ),
]
