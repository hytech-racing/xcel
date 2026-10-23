"""Small matplotlib window with sliders for the odometry noise. Polled from the sim loop,
so it runs on the main thread without its own event loop."""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .odom import OdomNoise

# (field, label, min, max)
SLIDERS = [
    ("vel_std", "vel noise [m/s]", 0.0, 1.0),
    ("yaw_rate_std", "yaw rate noise [rad/s]", 0.0, 0.5),
    ("vel_scale_err", "vel scale error [-]", -0.2, 0.2),
    ("gyro_bias", "gyro bias [rad/s]", -0.2, 0.2),
]


class OdomNoiseWindow:
    def __init__(self, noise: OdomNoise) -> None:
        self._noise = noise
        self._fig = plt.figure("odom noise", figsize=(5, 0.5 * len(SLIDERS) + 0.4))
        self._sliders = []  # keep references, widgets stop responding once garbage collected
        for i, (field, label, lo, hi) in enumerate(SLIDERS):
            ax = self._fig.add_axes((0.4, 1 - (i + 1) / (len(SLIDERS) + 0.5), 0.45, 0.5 / (len(SLIDERS) + 0.5)))
            slider = Slider(ax, label, lo, hi, valinit=getattr(noise, field))
            slider.on_changed(lambda value, field=field: setattr(self._noise, field, value))
            self._sliders.append(slider)
        self._fig.show()

    def poll(self) -> None:
        """Process pending GUI events; a closed window keeps the last values."""
        if plt.fignum_exists(self._fig.number):
            self._fig.canvas.flush_events()
