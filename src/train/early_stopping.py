class EarlyStopping:
    def __init__(
        self,
        patience=20,
        min_delta=0.0,
        mode="min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best = None
        self.counter = 0

    def step(self, metric):

        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience