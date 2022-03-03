class LazyFrames(object):
    """Array-like object that lazily concat multiple frames.

    This object ensures that common frames between the observations are only
    stored once.  It exists purely to optimize memory usage which can be huge
    for DQN's 1M frames replay buffers.

    This object should only be converted to numpy array before being passed to
    the model.

    You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames, stack_axis=2):
        self.stack_axis = stack_axis
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=self.stack_axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out
