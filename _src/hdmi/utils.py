import numpy as np
import scipy.signal as signal

from pathlib import Path
from typing import Literal
from PIL import Image
from dataclasses import dataclass

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"


def load_iq(file_path: Path) -> np.ndarray:
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.float32).view(np.complex64)
    return data

@dataclass
class VESAConfig:
    """
    VESA 640x480 @ 60Hz parameters and SDR settings.
    """

    # Constants
    pixel_clock: float = 25.175e6
    h_total: int = 800
    v_total: int = 525
    sample_rate: float = 64e6
    
    @property
    def frame_dims(self) -> tuple[int, int]:
        return (self.h_total, self.v_total)
    
    @property
    def h_freq(self) -> float:
        return self.pixel_clock / self.h_total
        
    @property
    def v_freq(self) -> float:
        return self.h_freq / self.v_total

def compute_autocorrelation(x: np.ndarray, mode: str = "coherent") -> np.ndarray:
    # Apply: Coherent or Incoherent
    if mode == "incoherent":
        x = np.abs(x)
        
    # Remove the mean
    x_centered = x - np.mean(x)
    
    # Correlate using FFT
    xcorr = signal.correlate(x_centered, x_centered, mode="full", method="fft")
    
    # Unbiased Normalisation
    n = len(x)
    overlaps = np.concatenate([np.arange(1, n), [n], np.arange(n - 1, 0, -1)])
    
    # Avoid division by zero at extreme tails (though usually we crop them)
    with np.errstate(divide='ignore', invalid='ignore'):
        xcorr_norm = xcorr / overlaps
    
    # Keep only positive lags for autocorrelation plot
    mid_point = xcorr_norm.size // 2
    xcorr_norm = xcorr_norm[mid_point:]
    
    return np.abs(xcorr_norm)

def refine_peaks(y, x):
    if x <= 0 or x >= len(y) - 1:
        return float(x)

    left = y[x - 1]
    center = y[x]
    right = y[x + 1]
    
    denominator = (left - 2 * center + right)
    if denominator == 0:
        return float(x)
        
    delta = 0.5 * (left - right) / denominator
    return float(x) + delta

def compute_drift(
    data: np.ndarray, 
    frame_size_samples: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the drift between the first and last frame using cross-correlation.
    
    Returns:
        lags: The lag indices.
        xcorr_drift: The cross-correlation result.
        drift_lag: The estimated drift in samples.
    """
    num_frames_available = len(data) // frame_size_samples
    
    # Extract first and last
    frame_first = data[:frame_size_samples]
    frame_last_idx = num_frames_available - 1
    start_idx_last = frame_last_idx * frame_size_samples
    frame_last = data[start_idx_last : start_idx_last + frame_size_samples]
    
    # Cross Correlate
    # Remove DC component
    frame_last_centered = frame_last - np.mean(frame_last)
    frame_first_centered = frame_first - np.mean(frame_first)
    
    # xcorr of (last, first) -> drift
    xcorr_drift = np.abs(signal.correlate(frame_last_centered, frame_first_centered, mode="full"))
    lags = np.arange(-frame_size_samples + 1, frame_size_samples)
    drift_peak_idx = np.argmax(xcorr_drift)
    
    # Sub-sample refinement
    refined_idx = refine_peaks(xcorr_drift, drift_peak_idx)
    drift_lag = lags[0] + refined_idx
    
    return lags, xcorr_drift, drift_lag

def contrast_enhance(
    image: np.ndarray,
    method: Literal["quantile", "std"],
    **kwargs
) -> np.ndarray:
    """
    Apply contrast enhancement (normalisation) to an image.
    """
    return _norm_gray(image, method, **kwargs)

def _norm_gray(arr: np.ndarray, method: str, **kwargs) -> np.ndarray:
    if method == "quantile":
        return _stretch_quantile(arr, **kwargs)
    elif method == "std":
        return _stretch_std(arr)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

def _stretch_quantile(arr: np.ndarray, low_quantile=0.02, high_quantile=0.98) -> np.ndarray:

    low = np.quantile(arr, low_quantile)
    high = np.quantile(arr, high_quantile)
    clipped = np.clip(arr, low, high)

    return _stretch_minmax(clipped)

def _stretch_std(arr: np.ndarray, sigma=3.0) -> np.ndarray:
    mean = np.mean(arr)
    std = np.std(arr)
    lo = mean - sigma * std
    hi = mean + sigma * std

    clipped = np.clip(arr, lo, hi)

    return _stretch_minmax(clipped)

def _stretch_minmax(arr: np.ndarray, axis=None) -> np.ndarray:
    mn = np.min(arr, axis=axis, keepdims=True)
    mx = np.max(arr, axis=axis, keepdims=True)    
    out = np.zeros_like(arr)
    diff = mx - mn
    diff[diff == 0] = 1.0 
    out = (arr - mn) / diff
    return np.clip(out, 0, 1)
