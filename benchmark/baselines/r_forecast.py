"""
This file contains various baselines which use the R package "forecast".

This requires R to be installed to run:
conda install r-essentials r-forecast r-unix
"""

import pandas as pd
import numpy as np
import time

_rpy2_initialized = False
try:
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2 import rinterface, robjects

    _rpy2_initialized = True
except:
    pass

from .base import Baseline
from ..base import BaseTask


class R_ETS(Baseline):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
        model: str = "ZZZ",
    ):
        """
        Get predictions from the ETS model.

        Parameters:
        -----------
        model: str
            Which model to use, a 3 letters code for error, trend, and seasonality.
            The letters can be Z (automatic), A (additive), M (multiplicative) or N (none)
        """
        if not _rpy2_initialized:
            raise RuntimeError("The rpy2 package has not been successfully imported.")

        super().__init__()

        self.model = model

        # Required R packages:
        self._stats_pkg = None  # "stats"
        self._forecast_pkg = None  # "forecast"

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        starting_time = time.time()
        samples = self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=task_instance.seasonal_period,
            n_samples=n_samples,
        )
        extra_info = {
            "total_time": time.time() - starting_time,
        }
        return samples, extra_info

    def forecast(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        seasonal_periods: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarly meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.

        Note: If seasonal_periods is <= 0, then we set the period to 1, which skips it if the model uses "Z" as its seasonal component.
              If the model uses "A" or "M", then it will fail.
        """
        # Lazy initialization, since otherwise the parallel dispatching using pickle will not work
        if self._stats_pkg is None:
            self._stats_pkg = rpackages.importr("stats")
        if self._forecast_pkg is None:
            self._forecast_pkg = rpackages.importr("forecast")

        history = pandas2ri.py2rpy(past_time[past_time.columns[-1]])
        ts = self._stats_pkg.ts(history, frequency=max(1, seasonal_periods))
        fit = self._forecast_pkg.ets(ts, model=self.model)

        samples = np.stack(
            [
                numpy2ri.rpy2py(
                    self._forecast_pkg.simulate_ets(fit, nsim=len(future_time))
                )
                for _ in range(n_samples)
            ]
        )
        samples = samples[:, :, None]

        if np.isnan(samples).any():
            # If the model fails, then switch off the trend component of the model, then rerun it.
            no_trend_model = self.model[0] + "N" + self.model[2]

            fit = self._forecast_pkg.ets(ts, model=no_trend_model)

            samples = np.stack(
                [
                    numpy2ri.rpy2py(
                        self._forecast_pkg.simulate_ets(fit, nsim=len(future_time))
                    )
                    for _ in range(n_samples)
                ]
            )
            samples = samples[:, :, None]

        return samples

    @property
    def cache_name(self) -> str:
        args_to_include = ["model"]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the R packages
        state["_stats_pkg"] = None
        state["_forecast_pkg"] = None
        return state


_R_ARIMA_CODE = r"""
require(parallel)
# require(pbmcapply)      # mcapply with progress bars
require(forecast)
require(unix)           # eval_safe

# Compute an affine transform to be applied to y before fitting the
# ARIMA model, of the form (y - A) / B + C.
# The last term (C) might be 1.0 for positive data, and corresponds to
# an additional bias that makes fitting ARIMA models more stable.
arima_scale_factors <- function(y, eps = 1.0, C = 0.0) {
    A <- 0
    B <- 1
    sdy <- sd(y)
    if (sdy > 0 && min(y) >= 0) {
        miny <- min(y)
        #maxy <- max(y)
        A <- miny - eps
        B <- sdy             # Better behaved than the max for ARIMA fitting
        if (is.null(C))
            C <- min(1.0, miny)
    }
    else if (sdy > 0 && max(y) < 0) {
        #miny <- min(y)
        maxy <- max(y)
        A <- maxy + eps
        B <- sdy
        if (is.null(C))
            C <- max(-1.0, maxy)
    }
    stopifnot(B > 0)
    list(A = A, B = B, C = C)
}


fit_arima_and_simulate <- function(y, prediction_length, num_samples,
                                   max.p = 3, max.q = 3, max.P = 2, max.Q = 2,
                                   stepwise=FALSE, num.cores=2,
                                   timeout = 1800, ...)
{
    # Filter simulation results (by column) that either:
    # - are NA
    # - are Infinite
    # - Are more than 1000x (in absolute value) away from train values
    filter_sim <- function(y, sim, thresh=1000) {
        maxy <- max(abs(y))
        keepsim <- is.finite(sim) & abs(sim) <= thresh * maxy
        ii <- which(apply(keepsim, 2, all))
        sim[, ii]
    }

    # Internal function that can be called a few times with ever simpler
    # specifications if the fitting doesn't work.
    fitsim <- function(y, max.p = 3, max.q = 3, max.P = 2, max.Q = 2,
                       stepwise, num.cores, timeout,
                       print_elapsed=FALSE, ...) {
        tryCatch({
            eval_safe({
                tau <- system.time({
                    fit <- auto.arima(y,
                                    max.p = max.p, max.q = max.q,
                                    max.P = max.Q, max.Q = max.Q,
                                    lambda=NULL, stepwise=stepwise,
                                    num.cores=num.cores, ...)
                }, gcFirst=FALSE)
                if (print_elapsed) {
                    message("ARIMA fitting time: ",
                            tau["elapsed"])
                    flush.console()
                }

                # Now simulate. Replicate 5x more than the nominal num_samples,
                # and keep num_samples post filtering
                sim <- replicate(5*num_samples, simulate(fit, nsim=prediction_length))
                finitesim <- filter_sim(y, sim)
                return(list(fit=fit, fitting_time=tau["elapsed"],
                            sim=finitesim[, seq_len(min(num_samples,
                                                        ncol(finitesim)))]))
            }, timeout=timeout)
        },
        error=function(e) {
            message("Error in fit_arima_and_simulate: ", str(e))
            flush.console()
            return(NULL)
        } )
    }

    simplified <- 0                     # Number of times we've simplified
    sc <- arima_scale_factors(y)
    y <- (y - sc$A) / sc$B + sc$C

    # Try fitting with the specified parameters, and if that fails, try again with a
    # simpler model
    fit <- fitsim(y,
                  max.p = max.p, max.q = max.q,
                  max.P = max.Q, max.Q = max.Q,
                  stepwise=stepwise,
                  num.cores=num.cores, timeout=timeout, ...)
    if (!is.null(fit) && ncol(fit$sim) == num_samples) {
        sim <- ((fit$sim - sc$C) * sc$B) + sc$A
        fitted <- fitted.values(fit$fit)
        fitted <- ((fitted - sc$C) * sc$B) + sc$A

        if (all(is.finite(fitted)) && all(is.finite(sim)))
            return(list(fit=fit$fit, sim=sim, fitted=fitted,
                        scaled_y=y, scale=sc, simplified=simplified,
                        fitting_time=fit$fitting_time))
    }

    # If we get here, the fit failed. Try again with a much simpler model.
    # Voluntarily don't include ellipsis here.
    simplified <- simplified + 1
    fit <- fitsim(y, max.p=1, max.q=1, seasonal=FALSE,
                  stepwise=stepwise, num.cores=num.cores, timeout=timeout)
    if (!is.null(fit) && ncol(fit$sim) == num_samples) {
        sim <- ((fit$sim - sc$C) * sc$B) + sc$A
        fitted <- fitted.values(fit$fit)
        fitted <- ((fitted - sc$C) * sc$B) + sc$A

        if (all(is.finite(fitted)) && all(is.finite(sim)))
            return(list(fit=fit$fit, sim=sim, fitted=fitted,
                        scaled_y=y, scale=sc, simplified=simplified,
                        fitting_time=fit$fitting_time))
    }

    message("Returning NULL")
    flush.console()

    return(NULL)
}
"""


class R_Arima(Baseline):

    __version__ = "0.0.1"  # Modification will trigger re-caching

    def __init__(
        self,
    ):
        """
        Get predictions from the Arima model.
        """
        if not _rpy2_initialized:
            raise RuntimeError("The rpy2 package has not been successfully imported.")

        super().__init__()

        # The R function to call:
        self._fit_and_simulate = None
        # Required R packages:
        self._stats_pkg = None  # "stats"

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        starting_time = time.time()
        samples = self.forecast(
            past_time=task_instance.past_time,
            future_time=task_instance.future_time,
            seasonal_periods=task_instance.seasonal_period,
            n_samples=n_samples,
        )
        extra_info = {
            "total_time": time.time() - starting_time,
        }
        return samples, extra_info

    def forecast(
        self,
        past_time: pd.DataFrame,
        future_time: pd.DataFrame,
        seasonal_periods: int,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarly meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.

        Note: If seasonal_periods is <= 0 (aka: no periods), then we set the period to 1.
        """
        # Lazy initialization, since otherwise the parallel dispatching using pickle will not work
        if self._stats_pkg is None:
            self._stats_pkg = rpackages.importr("stats")
        if self._fit_and_simulate is None:
            rinterface.initr()
            robjects.r(_R_ARIMA_CODE)
            self._fit_and_simulate = robjects.r("fit_arima_and_simulate")

        history = pandas2ri.py2rpy(past_time[past_time.columns[-1]])
        ts = self._stats_pkg.ts(history, frequency=max(1, seasonal_periods))
        prediction_length = len(future_time)

        result = self._fit_and_simulate(
            ts,
            prediction_length=prediction_length,
            num_samples=n_samples,
        )

        result_dict = dict(zip(result.names, list(result)))
        samples = numpy2ri.rpy2py(result_dict["sim"]).T
        samples = samples[:, :, None]

        return samples

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the R method and packages
        state["_fit_and_simulate"] = None
        state["_stats_pkg"] = None
        return state
