"""
Normalization utilities for time series data.
"""

import logging
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

NormalizationMethod = Literal["standard", "minmax", "robust", "log1p"]
EPSILON = 1e-8


def normalize(
    data: np.ndarray,
    method: NormalizationMethod = "standard",
    axis: int = 0,
    custom_params: Union[Dict[str, Any], None] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalizes time series data using the specified method.

    Args:
        data (np.ndarray): The input time series data. Must be 1D or 2D.
                           If 2D, normalization occurs along the specified axis.
        method (NormalizationMethod): The normalization method to use.
            - "standard": Standard scaling (Z-score normalization). (X - mean) / std.
            - "minmax": Min-max scaling. (X - min) / (max - min). Scales to [0, 1].
            - "robust": Robust scaling using median and IQR. (X - median) / IQR.
            - "log1p": Log transformation (log(1 + X)). Useful for positive data
                       with skewed distributions. Does not use `axis` as it's element-wise.
        axis (int): The axis along which to compute statistics for normalization.
                    Typically 0 for column-wise (features) or 1 for row-wise.
                    Ignored for "log1p".
        custom_params (Dict[str, Any], optional): Pre-computed parameters to use for normalization.
            If provided, these parameters will be used instead of calculating them from `data`.
            This is useful for applying a learned transformation to new data (e.g., test set).
            The structure of this dict depends on the method:
            - "standard": {"mean": np.ndarray, "std": np.ndarray}
            - "minmax": {"min": np.ndarray, "max": np.ndarray}
            - "robust": {"median": np.ndarray, "iqr": np.ndarray}
            - "log1p": Not applicable as it's stateless in terms of fitted params.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - The normalized data.
            - A dictionary containing the parameters used for normalization,
              which are needed for denormalization. Includes 'method' and 'axis'.

    Raises:
        ValueError: If the input data is not a NumPy array, has unsupported dimensions,
                    or an invalid method is specified.
        ValueError: If `custom_params` are provided for "log1p" or are missing required keys.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    if data.ndim not in [1, 2]:
        raise ValueError("Input data must be 1D or 2D.")

    # If data is 1D, ensure axis=0 and calculations behave as if it's a single column/feature
    original_ndim = data.ndim
    if original_ndim == 1:
        # reshape 1D array to 2D column vector for consistent axis handling except for log1p which is element-wise
        if method != "log1p":
            data = data.reshape(-1, 1) if axis == 0 else data.reshape(1, -1)
            if axis != 0 and method != "log1p":  # log1p is element-wise
                # For 1D data, axis must be 0 or it's ambiguous for these scalers
                raise ValueError("For 1D data, 'axis' must be 0 for scaling methods.")

    params: Dict[str, Any] = {"method": method, "axis": axis}
    normalized_data: np.ndarray

    if method == "standard":
        if custom_params:
            if "mean" not in custom_params or "std" not in custom_params:
                raise ValueError("Custom params for 'standard' must include 'mean' and 'std'.")
            mean = custom_params["mean"]
            std = custom_params["std"]
        else:
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)

        params.update({"mean": mean, "std": std})
        # Avoid division by zero for constant features
        normalized_data = (data - mean) / np.maximum(std, EPSILON)

    elif method == "minmax":
        if custom_params:
            if "min" not in custom_params or "max" not in custom_params:
                raise ValueError("Custom params for 'minmax' must include 'min' and 'max'.")
            data_min = custom_params["min"]
            data_max = custom_params["max"]
        else:
            data_min = np.min(data, axis=axis, keepdims=True)
            data_max = np.max(data, axis=axis, keepdims=True)

        params.update({"min": data_min, "max": data_max})
        scale = data_max - data_min
        # Avoid division by zero for constant features; result will be 0
        normalized_data = (data - data_min) / np.maximum(scale, EPSILON)
        # Handle case where min == max: scale is 0, so normalized_data becomes 0.
        # If scale is 0 (min == max), (data - data_min) is also 0. 0 / EPSILON is 0. Correct.

    elif method == "robust":
        if custom_params:
            if "median" not in custom_params or "iqr" not in custom_params:
                raise ValueError("Custom params for 'robust' must include 'median' and 'iqr'.")
            median = custom_params["median"]
            iqr = custom_params["iqr"]
        else:
            # For 1D data, axis=0 means percentile over all elements.
            # For 2D data, axis=0 means percentile per column.
            q1 = np.percentile(data, 25, axis=axis, keepdims=True)
            median = np.median(data, axis=axis, keepdims=True)  # q2 or 50th percentile
            q3 = np.percentile(data, 75, axis=axis, keepdims=True)
            iqr = q3 - q1

        params.update({"median": median, "iqr": iqr})
        # Avoid division by zero for constant features (IQR=0)
        normalized_data = (data - median) / np.maximum(iqr, EPSILON)

    elif method == "log1p":
        if custom_params:
            raise ValueError("'log1p' does not support custom_params as it's stateless.")
        if np.any(data < -1):  # log1p(x) = log(1+x), so 1+x must be > 0
            # Warning: log1p is typically used for non-negative data.
            # If data can be < -1, this will produce NaNs or errors.
            # Consider adding a small constant if data can be slightly negative
            # or ensure data is appropriate for log1p.
            pass  # np.log1p will handle it by returning NaN, which might be desired.
        normalized_data = np.log1p(data)
        # No numerical parameters to store beyond the method itself for log1p
        # axis is stored but not used by log1p itself for denormalization.
    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Supported methods are: 'standard', 'minmax', 'robust', 'log1p'."
        )

    # If original data was 1D, revert normalized_data to 1D
    if original_ndim == 1 and method != "log1p":
        # If axis=0, data was (-1,1), result is (-1,1), squeeze to (-1,)
        # If axis=1, data was (1,-1), result is (1,-1), squeeze to (-1,)
        # Squeeze will remove all single-dimensional entries from the shape.
        normalized_data = np.squeeze(normalized_data)
    elif original_ndim == 1 and method == "log1p":  # log1p is element-wise, preserves 1D
        pass

    return normalized_data, params


def denormalize(normalized_data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Denormalizes time series data using parameters from a previous normalization.

    Args:
        normalized_data (np.ndarray): The normalized input data.
        params (Dict[str, Any]): The parameters dictionary returned by the
                                 `normalize` function. Must contain 'method'
                                 and other method-specific parameters.

    Returns:
        np.ndarray: The denormalized (original scale) data.

    Raises:
        ValueError: If `params` is missing required keys or an invalid method is found.
    """
    if not isinstance(normalized_data, np.ndarray):
        raise ValueError("Input normalized_data must be a NumPy array.")
    if "method" not in params:
        raise ValueError("Parameters dictionary must contain 'method' key.")

    method = params["method"]
    original_data: np.ndarray

    # If normalized_data is 1D and the method (not log1p) used axis operations,
    # it might need reshaping to ensure broadcasting with stored params (which have keepdims=True).
    # The params (mean, std, etc.) would have shape like (1, M) or (M,) if data was (N, M)
    # or (1,) if data was 1D.
    # Squeezing in normalize for 1D input means params like mean (1,) will broadcast fine with 1D normalized_data.

    if method == "standard":
        if "mean" not in params or "std" not in params:
            raise ValueError("Params for 'standard' denormalization missing 'mean' or 'std'.")
        mean = params["mean"]
        std = params["std"]
        # Use the stored std. If it was 0, normalized_data should be 0 (due to EPSILON),
        # so 0 * 0 + mean = mean, which is correct.
        original_data = normalized_data * std + mean

    elif method == "minmax":
        if "min" not in params or "max" not in params:
            raise ValueError("Params for 'minmax' denormalization missing 'min' or 'max'.")
        data_min = params["min"]
        data_max = params["max"]
        scale = data_max - data_min
        # If scale was 0, normalized_data should be 0, so 0 * 0 + min = min. Correct.
        original_data = normalized_data * scale + data_min

    elif method == "robust":
        if "median" not in params or "iqr" not in params:
            raise ValueError("Params for 'robust' denormalization missing 'median' or 'iqr'.")
        median = params["median"]
        iqr = params["iqr"]
        # If iqr was 0, normalized_data should be 0, so 0 * 0 + median = median. Correct.
        original_data = normalized_data * iqr + median

    elif method == "log1p":
        original_data = np.expm1(normalized_data)  # exp(y) - 1

    else:
        raise ValueError(f"Unknown normalization method in params: {method}. " f"Cannot denormalize.")

    return original_data
