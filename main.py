from os import path, listdir
from datetime import datetime, timedelta
import threading

import pandas as pd

from plugins import BatteryMonitor, MoistureSensor
from core.model import RainfallEstimator
from core.mech_raingauge import DavisRainGauge
from core.connectivity import send_data
from utils.helper import time_stamp_fnamer, delete_files, config
from utils.logging import initialize_logging, log_time_remaining, write_rain_data_to_csv
from utils.dir import get_data_dir
from utils.audio_rec import record_audio, AudioManager


class AcousticRaingauge:

    def __init__(self):
        self.DB_write_interval = config["DB_writing_interval_min"]
        self.num_subsamples = config["infer_inetrval_sec"] // config["sample_duration_sec"]
        self.record_hours = config["record_hours"]
        self.end_time = datetime.now() + timedelta(hours=self.record_hours)
        self.deployed = config["field_deployed"]
        self.min_threshold = config["min_threshold"]
        self.moisture_threshold = config["moisture_threshold"]

        self.acoustic_model = RainfallEstimator()
        self.battery = BatteryMonitor()
        self.moisture_sensor = MoistureSensor()
        self.audio_recorder = AudioManager()

    def run(self):
        db_counter, rain = 0, 0
        locations = []
        result_data = []
        data_dir = get_data_dir()
        
        # audio recording thread
        rain_thread = threading.Thread(target=self.audio_recorder.run, daemon=True).start()

        try:
            if self.deployed:
                while True:
                    if self.audio_recorder.is_ready():
                        audio_samples = self.audio_sample_buffer.get_window()
                        rain_mm = self.acoustic_model.estimate_rainfall(audio_samples) # inference
                        print("Estimated rainfall: ", rain_mm)

                        rain += rain_mm
                        db_counter += 1

                        # reading moisture sensor
                        moisture = moisture_sensor.get_data()

                        # reading battery parameters
                        # solar_V, battery_V, solar_I, battery_I = 17.2, 15.2, 1.5, 2.2
                        solar_V, battery_V, solar_I, battery_I = (self.battery.get_dataframe())

                        # sending data to DB
                        if db_counter == self.DB_write_interval:
                            if moisture and moisture < self.moisture_threshold and rain >= self.min_threshold:
                                send_data(config, mm_hat, solar_V, battery_V, solar_I, battery_I)
                            else:
                                send_data(config, 0.0, solar_V, battery_V, solar_I, battery_I)
                            rain, db_counter = 0, 0

            else:
                # run mechanical raingauge in new thread
                mech_raingauge = DavisRainGauge()
                rain_thread = threading.Thread(target=mech_raingauge.run, daemon=True).start()

                logger = initialize_logging(
                    config["audio_log_filename"],
                    datetime.now(),
                    int(self.record_hours * (3600 / config["sample_duration_sec"])),
                )
                for i in range(1, int(self.record_hours * (3600 / config["sample_duration_sec"])) + 1):
                    dt_now = datetime.now()
                    logger.info(f"Recording sample number {i} on {dt_now}")
                    audio_fname = time_stamp_fnamer(dt_now) + ".wav"
                    location = path.join(data_dir, audio_fname)
                    self.audio_recorder.record_audio(location)
                    locations.append(location)

                    if i % self.num_subsamples == 0: # estimating rainfall
                        mm_hat = self.acoustic_model.estimate_rainfall(locations)
                        # logger.info("Estimated rainfall: ", mm_hat)
                        locations.clear()
                        moisture = moisture_sensor.get_data() # reading moisture sensor
                        result_data.append(
                            {
                                "time_stamp": dt_now,
                                "rainfall_estimate": mm_hat,
                                "moisture": moisture,
                            }
                        )
                        write_rain_data_to_csv(result_data, config["rain_log_filename"])
                        rain += mm_hat
                        db_counter += 1

                        # reading battery parameters
                        # solar_V, battery_V, solar_I, battery_I = 17.2, 15.2, 1.5, 2.2
                        solar_V, battery_V, solar_I, battery_I = (self.battery.get_dataframe())
                        
                        # sending data to DB
                        if db_counter == self.DB_write_interval:
                            if moisture and moisture < self.moisture_threshold and rain >= self.min_threshold:
                                send_data(config, mm_hat, solar_V, battery_V, solar_I, battery_I)

                            else:
                                send_data(config, 0.0, solar_V, battery_V, solar_I, battery_I)
                            rain, db_counter = 0, 0
                    log_time_remaining(logger, self.end_time)
                logger.info(f"Finished data logging at {datetime.now()}\n")

        except KeyboardInterrupt:
            print("Execution interrupted by user")
        finally:
            pass


if __name__ == "__main__":
    AcousticRaingauge.run()
