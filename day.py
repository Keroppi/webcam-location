import constants

class Day():
    def get_sun_idx(self, times, sunrise, sunset):
        sunrise_idx = 0
        sunset_idx = 0

        max_sunrise_idx = None
        max_sunset_idx = None
        for time in times:
            if sunrise < time:
                max_sunrise_idx = sunrise_idx
                break

            sunrise_idx += 1
            sunset_idx += 1

        for time in times:
            if sunset < time:
                max_sunset_idx = sunset_idx
                break

            sunrise_idx += 1
            sunset_idx +=1

        if max_sunrise_idx is None:
            sunrise_idx = constants.IMAGES_PER_DAY # Past the last image.
        elif max_sunrise_idx == 0:
            sunrise_idx = -1
        else:
            remainder = (sunrise - times[max_sunrise_idx - 1]) / (times[max_sunrise_idx] - times[max_sunrise_idx - 1])
            sunrise_idx = remainder + max_sunrise_idx - 1

        if max_sunset_idx is None:
            sunset_idx = constants.IMAGES_PER_DAY
        elif max_sunset_idx == 0:
            sunset_idx = -1
        else:
            remainder = (sunset - times[max_sunset_idx - 1]) / (times[max_sunset_idx] - times[max_sunset_idx - 1])
            sunset_idx = remainder + max_sunset_idx - 1

        return (sunrise_idx, sunset_idx)

    def __init__(self, times, img_stack, sunrise, sunset):
        self.date = times[0].date()
        self.img_stack = img_stack
        self.sunrise = sunrise
        self.sunset = sunset

        # Randomly select IMAGES_PER_DAY images from times.


        self.sunrise_idx, self.sunset_idx = self.get_sun_idx(times, sunrise, sunset)






