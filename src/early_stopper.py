class EarlyStopper:
    def __init__(self, patience: int = 3) -> None:
        self.patience: int = patience
        self.best_score: float = float('inf')
        self.counter: int = 0
        self.save_model = False

    def check(self, validation_score: float) -> bool:
        self.save_model = False
        if validation_score > self.best_score:
            self.counter += 1
            if self.counter == self.patience:
                return True
        else:
            self.best_score = validation_score
            self.counter = 0
            self.save_model = True
        return False