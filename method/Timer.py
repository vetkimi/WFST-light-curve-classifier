class Timer:  #@save
    """Record time consumption"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in your list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Returns the average time"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Returns the sum of the times."""
        return sum(self.times)

    def cumsum(self):
        """Returns the accumulated time"""
        return np.array(self.times).cumsum().tolist()