class BaseController:
    def __init__(self):
        self.status = "Initialized"
    
    def log(self, message):
        """ Log message to the console or file. """
        print(f"[LOG]: {message}")

    def handle_error(self, error_message):
        """ Handle errors in a standardized way. """
        print(f"[ERROR]: {error_message}")
    
    def start(self):
        """ Placeholder method to be overridden by subclasses. """
        raise NotImplementedError("Subclasses should implement this method.")
