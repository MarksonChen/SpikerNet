class DBS:
    """DBS base class"""
    def __init__(self, dt: float = 1e-3, stim_amp: float = 1.0,
                 width: float = 60.0, tstart: float = 0.0):
        self.dt, self.stim_amp, self.width, self.tstart = dt, stim_amp, width, tstart

    @property
    def charge(self) -> float:           # mC
        return max(0.0, self.stim_amp * self.width * 1e-6)


class cDBS(DBS):
    """Continuous DBS pulse train."""
    def __init__(self, f: float, **kwargs):
        super().__init__(**kwargs)
        self.f = f                                           # Hz
        self._steps_per_pulse = max(1, round(1 / (self.f * self.dt)))
        self._pulse_counter   = 0
        self._i               = 0
        self._n_start         = round(self.tstart / self.dt)

    def advance(self) -> float:
        """Advance by one dt and return delivered charge (mC)."""
        self._i += 1
        if self._i <= self._n_start:
            return 0.0

        out = self.charge if self._pulse_counter == 0 else 0.0
        self._pulse_counter = (self._pulse_counter + 1) % self._steps_per_pulse
        return out


