class LabelMapper:
    def __init__(self, mode: str, num_classes: int, threshold_low=3, threshold_mid=5, threshold_high=6):
        """
        mode: 'A', 'V', or 'VAD'
        num_classes: 2, 3, or 8
        """
        self.mode = mode.upper()
        self.num_classes = num_classes
        self.threshold_low = threshold_low
        self.threshold_mid = threshold_mid
        self.threshold_high = threshold_high

        self._validate()

    def _validate(self):
        if self.mode in ['A', 'V'] and self.num_classes not in [2, 3]:
            raise ValueError("For A or V, num_classes must be 2 or 3")

        if self.mode == 'VAD' and self.num_classes != 8:
            raise ValueError("For VAD, num_classes must be 8")


    def __call__(self, valence=None, arousal=None, dominance=None):

        if self.mode == 'VAD':
            if valence is None or arousal is None or dominance is None:
                raise ValueError(
                    f"VAD mode requires valence, arousal, dominance. "
                    f"Got: V={valence}, A={arousal}, D={dominance}"
                )
            return self._map_vad(valence, arousal, dominance)

        elif self.mode == 'A':
            if arousal is None:
                raise ValueError("A mode requires arousal")
            return self._map_binary_or_ternary(arousal)

        elif self.mode == 'V':
            if valence is None:
                raise ValueError("V mode requires valence")
            return self._map_binary_or_ternary(valence)
    # ------------------------------
    # Arousal / Valence mapping
    # ------------------------------
    def _map_binary_or_ternary(self, value):
        if self.num_classes == 2:
            return 0 if value > self.threshold_mid else 1

        elif self.num_classes == 3:
            if value <= self.threshold_low:
                return 0
            elif value <= self.threshold_high:
                return 1
            else:
                return 2

    # ------------------------------
    # VAD 8-class mapping
    # ------------------------------
    def _map_vad(self, valence, arousal, dominance):
        v = 0 if valence > self.threshold_mid else 1
        a = 0 if arousal > self.threshold_mid else 1
        d = 0 if dominance > self.threshold_mid else 1

        # Binary encoding trick
        return v * 4 + a * 2 + d
