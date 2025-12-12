import numpy as np
from scipy.signal import resample

from dataclasses import dataclass
from typing import Literal

@dataclass
class ReconstructionConfig:
    width: int = 800
    height: int = 525
    sample_rate: float = 64e6
    refresh_rate: float = 60.0    
    is_coherent: bool = True
    interpolation_method: str = "sinc"
    
    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.width, self.height)
        
    @property
    def pixels_per_frame(self) -> int:
        return self.width * self.height

class HDMIReconstructor:
    def __init__(self, config: ReconstructionConfig):
        self.config = config
        self._clock_freq = None
        self._vertical_freq = config.refresh_rate
        
    @classmethod
    def factory_reconstruct(
        cls,
        iq: np.ndarray,
        frame_size: tuple[int, int],
        sample_rate: float,
        refresh_rate: float = 60.0,
        num_frames: int = 1,
        offset: int | tuple[int, int] = 0,
        stack_method: Literal["am", "split"] | None = "am",
        is_coherent: bool = True,
        interpolation_method: str = "sinc"
    ) -> np.ndarray:
        """
        Factory helper to quickly reconstruct an image from IQ data.
        """
        cfg = ReconstructionConfig(
            width=frame_size[0],
            height=frame_size[1],
            sample_rate=sample_rate,
            refresh_rate=refresh_rate,
            is_coherent=is_coherent,
            interpolation_method=interpolation_method
        )

        engine = cls(cfg)

        return engine._reconstruct(
            iq, 
            frames=num_frames, 
            offset=offset,
            stack_mode=stack_method,
        )

    def _reconstruct(
        self, 
        iq: np.ndarray, 
        frames: int = -1, 
        offset: int | tuple[int, int] = 0,
        stack_mode: Literal["average", "split"] | None = None,
    ) -> np.ndarray:
        clock_frequency = self.config.refresh_rate * self.config.pixels_per_frame
        duration = len(iq) / self.config.sample_rate
        target_sample_count = int(duration * clock_frequency)
        
        resampled = self._interpolate_iq(
            iq, 
            target_sample_count, 
            interpolation_method=self.config.interpolation_method)
        
        if frames > 0:
            limit = frames * self.config.pixels_per_frame
            resampled = resampled[:limit]

        framed_linear = self._offset_iq(
            resampled, 
            self.config.frame_size, 
            1, 
            offset
        )
        
        frame_len = self.config.pixels_per_frame
        n_frames = len(framed_linear) // frame_len

        batch = framed_linear[:n_frames * frame_len].reshape(
            (n_frames, self.config.height, self.config.width)
        )

        if self.config.is_coherent:
            agg = self._aggregate_frames(batch, stack_mode)
            img = self._demodulate(agg)
        else:
            demodualted_iq = self._demodulate(batch)
            img = self._aggregate_frames(demodualted_iq, stack_mode)

        return img

    def _interpolate_iq(
        self,
        data: np.ndarray,
        target_sample_count: int,
        interpolation_method: str = "sinc",
    ) -> np.ndarray:
        if interpolation_method == "sinc":
            return resample(data, target_sample_count)
        
        if interpolation_method == "linear":
            old_indices = np.arange(len(data))
            new_indices = np.linspace(0, len(data) - 1, target_sample_count)
            return np.interp(new_indices, old_indices, data)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}.")

    def _offset_iq(
        self,
        signal_data: np.ndarray,
        frame_dims: tuple[int, int],
        oversampling: int,
        start_offset: int | tuple[int, int],
    ) -> np.ndarray:
        width, height = frame_dims
        frame_len = width * height * oversampling
        
        if isinstance(start_offset, tuple):
            x, y = start_offset
            start = (y * width + x) * oversampling
        else:
            start = start_offset

        # Truncate to whole frames
        n_samples = len(signal_data) - start
        n_samples -= n_samples % frame_len
        
        return signal_data[start : start + n_samples]

    def _aggregate_frames(
        self,
        frames: np.ndarray,
        method: Literal["split", "average", "median"] | None = None
    ) -> np.ndarray:        
        if method is None:
            return frames
        if method == "average":
            return np.mean(frames, axis=0)
        if method == "median":
            return np.median(frames, axis=0)
        if method == "split":
            if len(frames) < 2:
                return frames[0]

            h = frames.shape[1]
            mid = h // 2

            split = np.zeros_like(frames[0])
            split[:mid] = frames[0, :mid]
            split[mid:] = frames[-1, mid:]
            return split

        raise ValueError(f"Unknown method:{method}")

    def _reshape_to_frames(self, signal_stream: np.ndarray) -> np.ndarray:
        """Reshapes 1D signal into (N, Height, Width) video batch."""
        width = self.config.width
        height = self.config.height
        frame_len = width * height
        
        n_frames = len(signal_stream) // frame_len
        
        limit = n_frames * frame_len
        
        return signal_stream[:limit].reshape((n_frames, height, width))

    def _demodulate(self, iq_data: np.ndarray) -> np.ndarray:
        return np.abs(iq_data)
