import sys

class Logger:
    """Redirects print statements to both a log file and the console."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        # Use 'w' to overwrite the file for each new run
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()