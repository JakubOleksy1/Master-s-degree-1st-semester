class Result:
    def __init__(self, message, value=None):
        self.isSuccess = None
        self.message = message
        self.value = value

    def __str__(self):
        return f"Result: {self.isSuccess}, Message: {self.message}, Value: {self.value}"

class Ok(Result):
    def __init__(self, message, value=None):
        super().__init__(message, value)
        self.isSuccess = True

class Error(Result):
    def __init__(self, message, value=None):
        super().__init__(message, value)
        self.isSuccess = False
