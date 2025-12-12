"""
Initial Scratchpad for auto-correlation and cross-correlation analysis
"""
# %%

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import scipy.signal

from hdmi import HDMIReconstructor, DATA_DIR, load_iq
from hdmi.utils import VESAConfig, compute_autocorrelation, refine_peaks, compute_drift, contrast_enhance
from hdmi.plotting import Plotter

# %%
CONFIG = VESAConfig()

# Load Data (400 MHz)
DATA_PATH_400 = DATA_DIR / "scene3-640x480-60-400M-64M-40M.dat"
iq_400 = load_iq(DATA_PATH_400)

# %%

SEARCH_TOLERANCE = 0.02
ANALYSIS_DURATION = 0.2

upper_limit = int(CONFIG.sample_rate * ANALYSIS_DURATION)

# Auto-correlation
coh_xcorr = compute_autocorrelation(iq_400[:upper_limit], mode="coherent")
incoh_xcorr = compute_autocorrelation(iq_400[:upper_limit], mode="incoherent")

# Normalise
coh_xcorr_norm = coh_xcorr / np.max(coh_xcorr)
incoh_xcorr_norm = incoh_xcorr / np.max(incoh_xcorr)

# Plot the entire view
Plotter.plot_autocorrelation_comparison(coh_xcorr_norm, incoh_xcorr_norm)

# Plot a zoomed in plot
approx_frame_lag = int(CONFIG.sample_rate / CONFIG.v_freq)
ZOOM_TOLERANCE = 0.005
zoom_width = int(approx_frame_lag * ZOOM_TOLERANCE) 
zoom_start = approx_frame_lag - zoom_width
zoom_end = approx_frame_lag + zoom_width

# Plot zoomed view
Plotter.plot_zoomed_peak(coh_xcorr_norm, incoh_xcorr_norm, zoom_start, zoom_end)

# %%

peak_distance_approx = int(CONFIG.sample_rate / (CONFIG.v_freq * (1 + SEARCH_TOLERANCE)))
peaks, _ = scipy.signal.find_peaks(coh_xcorr, distance=peak_distance_approx, height=np.mean(coh_xcorr))

# Guess an initial period
expected_period = int(CONFIG.sample_rate / CONFIG.v_freq)
search_window = int(expected_period * SEARCH_TOLERANCE) # Explicit tolerance window

search_slice = coh_xcorr[expected_period-search_window : expected_period+search_window]
local_max = np.argmax(search_slice)
initial_period_int = (expected_period - search_window) + local_max
initial_period = refine_peaks(coh_xcorr, initial_period_int)

fv_estimated = CONFIG.sample_rate / initial_period
print(f"Initial Fv Estimate: {fv_estimated:.4f} Hz")

# Perform Long-Baseline Drift Estimation
frame_size_samples = int(initial_period)
lags, xcorr_drift, drift_lag = compute_drift(iq_400, frame_size_samples)

print(f"Drift Lag: {drift_lag:.2f} samples")

# Plot Period Estimation
Plotter.plot_period_estimation(lags, xcorr_drift, drift_lag)

# Update Global Constants
num_frames_available = len(iq_400) // frame_size_samples
frame_last_idx = num_frames_available - 1
start_idx_last = frame_last_idx * frame_size_samples

true_total_duration = start_idx_last + drift_lag

FV_AUTOMATED = CONFIG.sample_rate / (true_total_duration / frame_last_idx)
FP_AUTOMATED = FV_AUTOMATED * CONFIG.h_freq / CONFIG.v_freq

print(f"Final Refined Fv: {FV_AUTOMATED:.6f} Hz")
print(f"Final Refined Fp: {FP_AUTOMATED:.6f} Hz")

# %%
logger.info("Generating Figure 3.3: Split-Screen Accuracy Test...")
cc_img = HDMIReconstructor.factory_reconstruct(
    iq=iq_400,
    frame_size=CONFIG.frame_dims,
    sample_rate=CONFIG.sample_rate,
    refresh_rate=FV_AUTOMATED,
    num_frames=30,
    offset=(50, 0),
    stack_method="average",
    is_coherent=False
)
Plotter.plot_report_figure(cc_img)

# %%

logger.info("Generating Figure 3.3: Split-Screen Accuracy Test...")
frames_stack = HDMIReconstructor.factory_reconstruct(
    iq=iq_400,
    frame_size=CONFIG.frame_dims,
    sample_rate=CONFIG.sample_rate,
    refresh_rate=FV_AUTOMATED,
    num_frames=60,
    offset=0,
    stack_method=None,
)

first_frame = frames_stack[0]
last_frame = frames_stack[-1]

Plotter.plot_split_image(first_frame, last_frame)

# %%
"""
RGB Composite
"""

logger.info("Generating RGB Composite from Multi-Frequency Data...")

# Define paths
DATA_PATH_250 = DATA_DIR / "scene3-640x480-60-275M-64M-40M.dat"
DATA_PATH_450 = DATA_DIR / "scene3-640x480-60-475M-64M-40M.dat"

# Offsets
offset_250 = (33, 47)
offset_400 = (50, 0)
offset_450 = (325, 90)

def reconstruct_channel(path, offset_val):
    if not path.exists():
        logger.warning(f"File not found: {path} - Skipping channel")
        return np.zeros((CONFIG.frame_dims[1], CONFIG.frame_dims[0]))

    iq_data = load_iq(path)
    
    # Use capture_frame_image helper which wraps the new API
    # Equivalent to: core.reconstruct_hdmi_from_iq(..., coherent=False, correct_fp=True)
    img = HDMIReconstructor.factory_reconstruct(
        iq=iq_data,
        frame_size=CONFIG.frame_dims,
        sample_rate=CONFIG.sample_rate,
        refresh_rate=FV_AUTOMATED,
        num_frames=30,
        offset=offset_val,
        stack_method="average",
        is_coherent=False,
        interpolation_method="sinc"
    )

    return img 

img_250 = reconstruct_channel(DATA_PATH_250, offset_250)
img_400 = reconstruct_channel(DATA_PATH_400, offset_400)
img_450 = reconstruct_channel(DATA_PATH_450, offset_450)

logger.info(f"Displaying 250 MHz image with offset: {offset_250}")
Plotter.plot_report_figure(img_250)

logger.info(f"Displaying 400 MHz image with offset: {offset_400}")
Plotter.plot_report_figure(img_400)

logger.info(f"Displaying 450 MHz image with offset: {offset_450}")
Plotter.plot_report_figure(img_450)

# Combine the channels
logger.info(f"Displaying RGB ")
rgb_stack = np.stack([img_250, img_400, img_450], axis=-1)
rgb_stack = contrast_enhance(rgb_stack, method="quantile")
Plotter.plot_report_figure(rgb_stack)

# %%
"""
FFT Analysis and Reconstruction Experiments
(todo: needs to be integrated into the main pipepline)
"""

# %%

import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

# %%

pixel_clock = FV_AUTOMATED * CONFIG.h_total * CONFIG.v_total

# %%

from scipy.signal import resample, butter, sosfiltfilt

# This is based on my existing code.
# It should be ported at some point.
def manual_reconstruction(
    iq_data: np.ndarray, 
    frame_dims: tuple[int, int], 
    sample_rate: float, 
    max_frames: int = 5, 
    pixel_offset: int = 0, 
    pixel_clock: float | None = None
) -> np.ndarray:    
    duration = len(iq_data) / sample_rate
    target_samples = int(duration * pixel_clock)

    # Sinc interpolation
    resampled_signal = resample(iq_data, target_samples)
    
    frames_flat = slice_signal(
        resampled_signal, 
        frame_dims, 
        oversampling_factor=1, 
        start_offset=pixel_offset
    )
    
    # Reshape and limit number of frames
    num_pixels = frame_dims[0] * frame_dims[1]
    num_frames_available = len(frames_flat) // num_pixels
    
    if max_frames > 0:
        num_frames_available = min(num_frames_available, max_frames)
        
    frames_flat = frames_flat[:num_frames_available * num_pixels]
    
    if len(frames_flat) == 0:
        return np.array([])
        
    final_frames = frames_flat.reshape((num_frames_available, frame_dims[1], frame_dims[0]))
        
    return final_frames

def slice_signal(
    iq_data: np.ndarray,
    frame_dims: tuple[int, int],
    oversampling_factor: int,
    start_offset: int | tuple[int, int],
) -> np.ndarray:
    width, height = frame_dims
    pixels_per_frame = width * height
    
    if isinstance(start_offset, tuple):
        px, line = start_offset
        sample_offset = (line * width + px) * oversampling_factor
    else:
        sample_offset = start_offset
        
    sliced_signal = iq_data[sample_offset:]
    
    remainder = len(sliced_signal) % (pixels_per_frame * oversampling_factor)
    if remainder != 0:
        sliced_signal = sliced_signal[:-remainder]
        
    return sliced_signal

def compute_phase_diff(
    iq_data: np.ndarray,
    sample_rate: float,
    lpf_cutoff: float = 1e4
) -> np.ndarray:
    filtered_signal = sosfiltfilt(
        butter(N=8, Wn=lpf_cutoff, btype="low", output="sos", fs=sample_rate), 
        iq_data)
    
    phase_diff = np.angle(filtered_signal[1:] * np.conj(filtered_signal[:-1]))
    phase_diff = np.pad(phase_diff, (1, 0), mode="edge")
    
    return phase_diff

# %%

frames_uncorrected = manual_reconstruction(iq_400, CONFIG.frame_dims, CONFIG.sample_rate, max_frames=30, pixel_offset=offset_400, pixel_clock=pixel_clock)
# %%
Plotter.plot_rg_visual(frames_uncorrected[0], "Figure X (Left): Uncorrected Frame (Real/Green)")
# %%
Plotter.plot_hsv_visual(frames_uncorrected[0], "Figure X (Right): Uncorrected Frame (HSV)")
# %%

FC = 400e6
FP_STANDARD = 25.175e6

# %%
Plotter.plot_spectrum_analysis(iq_400, CONFIG.sample_rate, FC, FP_STANDARD)

# %%
logger.info("Performing Frequency Correction (Static Shift)...")
fft_vals = fftshift(fft(iq_400))
freqs = fftshift(fftfreq(len(iq_400), d=1/CONFIG.sample_rate))
freq_shift = freqs[np.argmax(np.abs(fft_vals))]

logger.info(f"Shift Frequency detected: {freq_shift:.2f} Hz")

# Manually applying shift for demonstration
t = np.arange(len(iq_400)) / CONFIG.sample_rate
phasor = np.exp(-1j * 2 * np.pi * freq_shift * t)
signal_shifted = iq_400 * phasor
# %%

phase_diff = compute_phase_diff(signal_shifted, CONFIG.sample_rate)
freq_profile_hz = phase_diff * CONFIG.sample_rate / (2 * np.pi)
logger.info("Plotting Frequency Deviation Profile")
plt.figure(figsize=(10, 6))
time_ms = np.arange(len(freq_profile_hz)) / CONFIG.sample_rate * 1000
plt.plot(time_ms, freq_profile_hz, linewidth=1)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency Deviation (Hz)")
plt.tight_layout()
plt.show()

# %%

logger.info("Plotting Dynamic Phase Correction")
pad_len = len(signal_shifted) - len(freq_profile_hz)
correction_freq = np.pad(freq_profile_hz, (0, pad_len), 'edge') if pad_len > 0 else freq_profile_hz[:len(signal_shifted)]

cumulative_phase_correction = np.cumsum(correction_freq) * (2 * np.pi / CONFIG.sample_rate)
dynamic_phasor = np.exp(-1j * cumulative_phase_correction)
signal_corrected = signal_shifted * dynamic_phasor

logger.info("Plotting Spectrum Comparison")
Plotter.plot_peak_comparison(signal_shifted, signal_corrected, CONFIG.sample_rate)

# %%
frames_corrected = manual_reconstruction(signal_corrected, CONFIG.frame_dims, CONFIG.sample_rate, max_frames=30, pixel_offset=offset_400, pixel_clock=pixel_clock)
coherent_avg = np.mean(frames_corrected, axis=0)

# Real/Green Visual
logger.info("Plotting Corrected RG")
Plotter.plot_rg_visual(frames_corrected[0], "Figure 7: Reconstructed Frame (Real=Red, Imag=Green)", 0)

logger.info("Plotting Corrected and Corrent HSV")
Plotter.plot_hsv_visual(coherent_avg, "Figure 6: Coherent Average (Noise Reduced)")
