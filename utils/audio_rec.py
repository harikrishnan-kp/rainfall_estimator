import subprocess
from typing import List, Optional

import numpy as np

from utils.helper import config
from utils.helper import time_stamp_fnamer


class AudioBuffer:
    """
    A sliding window audio buffer for managing audio data.

    This class handles the accumulation of audio chunks and provides overlapping windows
    suitable for signal processing tasks.

    Attributes:
        sampling_rate (int): The sample rate of the audio data.
        window_samples (int): Total number of samples in one processing window.
        buffer (np.ndarray): Internal buffer for accumulating samples.

    Example:
        >>> buffer = AudioBuffer(sampling_rate=16000, process_window_sec=1.0, hop_size_sec=0.5)
        >>> buffer.add(np.random.randn(8000))
        >>> windows = buffer.get_windows()
    """

    def __init__(self, sampling_rate: int, process_window_sec: float):
        """
        Initializes the AudioBuffer.

        Args:
            sampling_rate (int): Sample rate of the audio (e.g., 16000).
            process_window_sec (float): The length of the window to return in seconds.
        """
        self.sampling_rate = sampling_rate
        self.window_samples = int(sampling_rate * process_window_sec)

        self.buffer = np.array([], dtype=np.float32)

    def add(self, chunk: Optional[np.ndarray]) -> None:
        """
        Adds a new chunk of audio samples to the internal buffer.

        Args:
            chunk (np.ndarray | None): 1D array of audio samples or a list that can be converted.
        """
        if chunk is None or len(chunk) == 0:
            return

        chunk = np.asarray(chunk, dtype=np.float32).ravel()
        self.buffer = np.concatenate([self.buffer, chunk])

    def get_windows(self) -> List[np.ndarray]:
        """
        Generates all available full windows from the current buffer.

        Returns:
            (List[np.ndarray]): A list of 1D arrays, each of size `window_samples`.
        """
        windows = []

        while len(self.buffer) >= self.window_samples:
            window = self.buffer[:self.window_samples]
            windows.append(window)

            # Slide buffer
            self.buffer = self.buffer[self.window_samples:]

        return windows

    def clear(self) -> None:
        """Resets the internal buffer."""
        self.buffer = np.array([], dtype=np.float32)


class AudioManager(AudioBuffer):
    def __init__()
        self.deployment_status = config["field_deployed"]
        self.sampling_rate = config["sampling_rate"]
        self.resolution = config["resolution"]
        self.file_format = config["file_format"]
        self.duration = config["sample_duration_sec"]
        super.__init__(
            sampling_rate=self.sampling_rate,
            process_window_sec=1.0
        )

    def record(self):
        """
        Record audio and return it as a NumPy array.
        """
        cmd = [
            "arecord",
            "-q",
            f"--duration={self.duration}",
            "-t",
            "wav",
            "-f",
            str(self.resolution),
            "-r",
            str(self.sampling_rate),
            "-",
        ]

        wav_bytes = subprocess.check_output(cmd)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            audio = np.frombuffer(
                wav_file.readframes(wav_file.getnframes()),
                dtype=np.int16,
            )

        return audio

    def save(self, audio, output_path):
        """
        Save the recorded audio.
        """
        with open(output_path, "wb") as f:
            f.write(audio)

    def is_ready(self):
        """
        Check required number of samples are ready
        """
        return len(self.buffer) >= self.window_samples

    def run(self):
        """
        Record audio continuously
        """
        while True:
            dt_now = datetime.now()
            file_path = time_stamp_fnamer(dt_now) + ".wav"

            audio = self.record_audio(file_path)

            # add audio to buffer
            self.buffer.append(audio)
            
            # save audio if the setup is for training
            if self.deployment_status:
                self.save_audio(audio, file_path)
