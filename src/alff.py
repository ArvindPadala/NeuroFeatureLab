import numpy as np


def compute_alff(
    timeseries: np.ndarray,
    tr: float = 2.0,
    low_freq: float = 0.01,
    high_freq: float = 0.10
) -> np.ndarray:
    """
    Compute ALFF-style low-frequency amplitude features from fMRI-like time series.

    ALFF stands for Amplitude of Low-Frequency Fluctuation. In real resting-state
    fMRI analysis, ALFF summarizes the strength of low-frequency BOLD signal
    fluctuations in a specific frequency band.

    This implementation is a simplified research-methods demo.

    Parameters
    ----------
    timeseries : np.ndarray
        Array of shape (n_timepoints, n_regions).
    tr : float
        Repetition time in seconds.
    low_freq : float
        Lower frequency cutoff.
    high_freq : float
        Upper frequency cutoff.

    Returns
    -------
    np.ndarray
        ALFF-style value for each region.
    """

    if not isinstance(timeseries, np.ndarray):
        timeseries = np.asarray(timeseries)

    if timeseries.ndim != 2:
        raise ValueError("timeseries must be a 2D array of shape (n_timepoints, n_regions).")

    if tr <= 0:
        raise ValueError("tr must be positive.")

    if low_freq >= high_freq:
        raise ValueError("low_freq must be smaller than high_freq.")

    n_timepoints = timeseries.shape[0]

    # Remove mean signal per region
    demeaned = timeseries - np.mean(timeseries, axis=0)

    # Frequency values
    freqs = np.fft.rfftfreq(n_timepoints, d=tr)

    # Fast Fourier Transform amplitude
    fft_amplitude = np.abs(np.fft.rfft(demeaned, axis=0))

    # Select low-frequency band
    band_mask = (freqs >= low_freq) & (freqs <= high_freq)

    if not np.any(band_mask):
        raise ValueError("No frequency bins found in the requested low-frequency band.")

    # Average amplitude in low-frequency band for each region
    alff_values = fft_amplitude[band_mask].mean(axis=0)

    return alff_values


def create_alff_feature_names(region_names):
    """
    Create readable ALFF-style feature names.

    Example:
        temporal_alff
    """

    return [f"{region}_alff" for region in region_names]