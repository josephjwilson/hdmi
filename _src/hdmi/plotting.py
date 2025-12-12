import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

class Plotter:    
    @staticmethod
    def plot_report_figure(image: np.ndarray, cmap: str = "grey"):
        plt.figure(figsize=(10, 6))
        
        if image.ndim == 3 and image.shape[0] == 1:
            img_to_plot = image[0]
        else:
            img_to_plot = image

        if img_to_plot.ndim == 3:
             plt.imshow(img_to_plot)
        else:
             plt.imshow(img_to_plot, cmap=cmap)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # The rest of the plotters are a bit more niche for the specific figures.
    @staticmethod
    def plot_autocorrelation_comparison(
        coh_xcorr_norm: np.ndarray, 
        incoh_xcorr_norm: np.ndarray,
        title: Optional[str] = None
    ):
        """
        Plot 1: Full View Coherent vs Incoherent Autocorrelation.
        """
        plt.figure(figsize=(10, 4))
        t_axis = np.arange(len(coh_xcorr_norm))
        plt.plot(t_axis, coh_xcorr_norm, label="Coherent", linewidth=1.5)
        plt.plot(t_axis, incoh_xcorr_norm, label="Incoherent", alpha=0.7, linewidth=1)
        
        if title:
            plt.title(title)
            
        plt.ylabel("Normalised Amplitude")
        plt.xlabel("Samples")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_zoomed_peak(
        coh_xcorr_norm: np.ndarray, 
        incoh_xcorr_norm: np.ndarray,
        zoom_start: int,
        zoom_end: int,
        max_scale_factor: float = 1.5
    ):
        """
        Plot 2: Zoomed View (Focus on first frame peak).
        """
        t_axis = np.arange(len(coh_xcorr_norm))
        
        # Determine local max for scaling
        zoom_slice = coh_xcorr_norm[zoom_start:zoom_end]
        local_max_val = np.max(zoom_slice)
        
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, coh_xcorr_norm, label="Coherent")
        plt.plot(t_axis, incoh_xcorr_norm, label="Incoherent", alpha=0.7)
        plt.xlim(zoom_start, zoom_end)
        
        # Scale y-axis to focus on the signal
        plt.ylim(0, local_max_val * max_scale_factor)
        
        plt.xlabel("Samples")
        plt.ylabel("Normalised Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_period_estimation(
        lags: np.ndarray, 
        xcorr_drift: np.ndarray, 
        period: float,
        zoom_range: int = 5000,
        title: Optional[str] = None
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(lags, xcorr_drift, label="Cross-Correlation (Last vs First)")
        plt.axvline(0, color='gray', linestyle='--')
        plt.axvline(period, color='red', linestyle=':', label=f"Period: {period:.2f} samples")
        
        plt.xlim(period - zoom_range, period + zoom_range) # Zoom near the peak
        
        if title:
            plt.title(title)
            
        plt.xlabel("Samples")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_split_image(
        first_frame: np.ndarray,
        last_frame: np.ndarray,
        title: Optional[str] = None
    ):
        if first_frame.shape != last_frame.shape:
             raise ValueError("Frames must have same shape")

        split_point = first_frame.shape[0] // 2
        split_image = np.zeros_like(first_frame)
        split_image[:split_point, :] = first_frame[:split_point, :]
        split_image[split_point:, :] = last_frame[split_point:, :]

        plt.figure(figsize=(12, 10))
        plt.imshow(split_image, cmap='grey', vmin=0, vmax=np.quantile(split_image, 0.99))
        plt.axhline(split_point, color='red', linestyle='--', linewidth=1, label="Stitch Line (Frame 0 vs Frame 59)")
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
