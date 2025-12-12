import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from scipy.fft import fft, fftshift, fftfreq
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
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rg_visual(complex_frame: np.ndarray):
        real_part = np.real(complex_frame)
        real_p2, real_p98 = np.percentile(real_part, [2, 98])
        real_range = real_p98 - real_p2
        
        real_scale = real_range if real_range > 1e-6 else 1.0        
        red_channel = np.clip((real_part - real_p2) / real_scale, 0, 1)

        imag_part = np.imag(complex_frame)
        imag_p2, imag_p98 = np.percentile(imag_part, [2, 98])
        imag_range = imag_p98 - imag_p2
        
        imag_scale = imag_range if imag_range > 1e-6 else 1.0
    
        green_channel = np.clip((imag_part - imag_p2) / imag_scale, 0, 1)    
        blue_channel = np.zeros_like(red_channel)
        
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_hsv_visual(complex_frame: np.ndarray):
        magnitude = np.abs(complex_frame)
        mag_p2, mag_p98 = np.percentile(magnitude, [2, 98])
        mag_range = mag_p98 - mag_p2
        
        mag_scale = mag_range if mag_range > 1e-9 else 1.0
        
        value_channel = np.clip((magnitude - mag_p2) / mag_scale, 0, 1)
        
        phase_angle = np.angle(complex_frame)
        hue_channel = (phase_angle + np.pi) / (2 * np.pi)
        saturation_channel = np.ones_like(hue_channel) * 0.85 
        
        hsv_image = np.stack([hue_channel, saturation_channel, value_channel], axis=-1)
        rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrum_analysis(
        signal_data: np.ndarray, 
        sample_rate: float, 
        centre_freq: float, 
        pixel_clock: float
    ):
        num_samples = len(signal_data)
        fft_spectrum = fftshift(fft(signal_data))
        freqs = fftshift(fftfreq(num_samples, d=1/sample_rate))
        
        magnitude = np.abs(fft_spectrum)
        max_magnitude_val = np.max(magnitude)
        magnitude_norm = magnitude / max_magnitude_val
        
        freqs_mhz = (freqs + centre_freq) / 1e6
        pixel_clock_mhz = pixel_clock / 1e6
        
        plt.figure(figsize=(12, 6))
        plt.plot(freqs_mhz, magnitude_norm, alpha=0.9, label='Signal Spectrum')
        
        idx_max = np.argmax(magnitude_norm)
        freq_at_max = freqs_mhz[idx_max]
        mag_at_max = magnitude_norm[idx_max]
        
        plt.plot(freq_at_max, mag_at_max, 'r+', markersize=12, markeredgewidth=2, label='Carrier Peak')
        plt.annotate(f"Peak: {freq_at_max:.2f} MHz", 
                     xy=(freq_at_max, mag_at_max), 
                     xytext=(freq_at_max, mag_at_max + 0.05),
                     ha='center', va='bottom',
                     arrowprops=dict(arrowstyle='->', facecolor='black'))
        
        f_start = freqs_mhz[0]
        f_end = freqs_mhz[-1]
        n_min = int(np.ceil(f_start / pixel_clock_mhz))
        n_max = int(np.floor(f_end / pixel_clock_mhz))
        
        labeled_harmonic = False
        for harmonic_idx in range(n_min, n_max + 1):
            harmonic_freq = harmonic_idx * pixel_clock_mhz
            is_carrier = np.abs(harmonic_freq - freq_at_max) < 1.0
            
            style = '--' if not is_carrier else '-'
            alpha = 0.4 if not is_carrier else 0.8
            
            label_text = 'Pixel Clock Harmonics ($n \\cdot F_p$)' if (not labeled_harmonic and not is_carrier) else None
            plt.axvline(x=harmonic_freq, linestyle=style, alpha=alpha, label=label_text, color='green')
            plt.text(harmonic_freq, 1.02, f"{harmonic_idx}", ha='center', va='bottom', fontsize=8, transform=plt.gca().get_xaxis_transform())
            
            if label_text: labeled_harmonic = True

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Normalised Magnitude")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.ylim(0, 1.15)
        plt.show()

    @staticmethod
    def plot_peak_comparison(
        original_signal: np.ndarray, 
        corrected_signal: np.ndarray, 
        sample_rate: float
    ):
        num_samples = len(original_signal)
        frequencies = fftshift(fftfreq(num_samples, d=1/sample_rate))
        
        magnitude_original = np.abs(fftshift(fft(original_signal)))
        magnitude_corrected = np.abs(fftshift(fft(corrected_signal)))
        
        centre_index = num_samples // 2
        zoom_span = 2000
        
        frequencies_zoomed = frequencies[centre_index - zoom_span : centre_index + zoom_span]
        mag_orig_zoomed = magnitude_original[centre_index - zoom_span : centre_index + zoom_span]
        mag_corr_zoomed = magnitude_corrected[centre_index - zoom_span : centre_index + zoom_span]
        
        norm_factor_orig = np.max(mag_orig_zoomed) if np.max(mag_orig_zoomed) > 0 else 1.0
        norm_factor_corr = np.max(mag_corr_zoomed) if np.max(mag_corr_zoomed) > 0 else 1.0
        
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies_zoomed, mag_orig_zoomed / norm_factor_orig, label="Static Shift", alpha=0.6)
        plt.plot(frequencies_zoomed, mag_corr_zoomed / norm_factor_corr, label="Dynamic Correction", alpha=0.9)

        plt.xlabel("Frequency Offset (Hz)")
        plt.ylabel("Normalised Magnitude")
        plt.legend()
        plt.tight_layout()
        plt.show()
