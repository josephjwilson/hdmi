"""
Initial Scratchpad for testing
This has been heavily cleaned and ported to make it more readable.
"""

# %%
import numpy as np
import logging 

from hdmi import DATA_DIR, HDMIReconstructor, load_iq
from hdmi.utils import VESAConfig
from hdmi.plotting import Plotter

logger = logging.getLogger(__name__)

# %%
CONFIG = VESAConfig()

# Load Data (400 MHz)
data_file = DATA_DIR / "scene3-640x480-60-400M-64M-40M.dat"
id_400 = load_iq(data_file)

# %%

img = HDMIReconstructor.factory_reconstruct(
    iq=id_400,
    frame_size=CONFIG.frame_dims,
    sample_rate=CONFIG.sample_rate,
    refresh_rate=CONFIG.v_freq,
    num_frames=1,
    stack_method="average",
    is_coherent=False,
    interpolation_method="sinc" 
)

# %%
Plotter.plot_report_figure(img)

# %%
logger.info("Frequency & Interpolation Comparison...")

FREQUENCIES = {
    400: DATA_DIR / "scene3-640x480-60-400M-64M-40M.dat",
    425: DATA_DIR / "scene3-640x480-60-425M-64M-40M.dat",
    450: DATA_DIR / "scene3-640x480-60-450M-64M-40M.dat"
}

methods_to_test = ["linear", "sinc"]

for freq, file_path in FREQUENCIES.items():        
    logger.info(f"Processing {freq} MHz:")
    iq_data = load_iq(file_path)
    
    for method in methods_to_test:
        logger.info(f"Method: {method}:")
        
        img = HDMIReconstructor.factory_reconstruct(
            iq=iq_data,
            frame_size=CONFIG.frame_dims,
            sample_rate=CONFIG.sample_rate,
            refresh_rate=CONFIG.v_freq,
            num_frames=1,
            stack_method="average",
            is_coherent=False,
            interpolation_method=method
        )
        
        Plotter.plot_report_figure(img)

# %%

logger.info("Manual Sweep")

percentages = np.linspace(-0.5, 0.5, 6) 

for pct in percentages:
    offset_hz = CONFIG.pixel_clock * (pct / 100.0)
    current_pixel_clock = CONFIG.pixel_clock + offset_hz
    sweep_v_freq = current_pixel_clock / (CONFIG.h_total * CONFIG.v_total)
    
    logger.info(f"Sweeping {pct:+.1f}% ({offset_hz:+.1f} Hz)")
    
    img_sweep = HDMIReconstructor.factory_reconstruct(
        iq=id_400,
        frame_size=CONFIG.frame_dims,
        sample_rate=CONFIG.sample_rate,
        refresh_rate=sweep_v_freq,
        num_frames=1,
        stack_method="average",
        is_coherent=False,
        interpolation_method="sinc" 
    )
    
    Plotter.plot_report_figure(img_sweep)
