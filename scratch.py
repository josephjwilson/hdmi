"""
Initial Scratchpad for testing
This has been heavily cleaned and ported to make it more readable.
"""

# %%
from hdmi import DATA_DIR, HDMIReconstructor, load_iq
from hdmi.utils import VESAConfig
from hdmi.plotting import Plotter

# %%
CONFIG = VESAConfig()

# Load Data (400 MHz)
data_file = DATA_DIR / "scene3-640x480-60-425M-64M-40M.dat"
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
