class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def save(self):
        pass

    def act(self, _: int):
        return self.action_space.sample()

    def store(self, _: dict):
        pass

    def update(self) -> float:
        return 0.0
