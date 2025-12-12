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
