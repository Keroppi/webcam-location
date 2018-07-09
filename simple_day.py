import numpy as np, sys, PIL, math, time, datetime, random, statistics

class SimpleDay():
    def __init__(self, place, lat, lng, mali_solar_noon, time_offset, sunrise, sunset, sunrise_in_frames, sunset_in_frames, interval_min, season):
        self.place = place  # Name of the location

        self.lat = lat # Latitude
        self.lng = lng # Longitude

        self.mali_solar_noon = mali_solar_noon # Solar noon in Mali that day at 0 degrees longitude.

        self.time_offset = time_offset # Time zone and daylight savings offset in seconds.
        self.sunrise = sunrise # local time
        self.sunset = sunset # local time

        # Note: Can get UTC time by computing
        # sunrise - datetime.timedelta(seconds=days[d_idx].time_offset)
        # sunset - datetime.timedelta(seconds=days[d_idx].time_offset)

        self.sunrise_in_frames = sunrise_in_frames # True if sunrise is visible in all the original frames for the day (not just the 32 we pick).
        self.sunset_in_frames = sunset_in_frames

        self.interval_min = interval_min # Average interval (in minutes) between pictures (using all images from the day).

        self.season = season

        # Broken up into 3 month seasons:
        # For latitudes > 0: Dec - Feb is 'winter', Mar - May is 'spring', Jun - Aug is 'summer', Sep - Nov is 'fall'
        # For latitudes < 0: Dec - Feb is 'summer', Mar - May is 'fall', Jun - Aug is 'winter', Sep - Nov is 'spring'











