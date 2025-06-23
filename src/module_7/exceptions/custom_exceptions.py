# exceptions/custom_exceptions.py
class InvalidRequestException(Exception):
    """Exception for invalid requests"""
    pass

class ServiceUnavailableException(Exception):
    """Exception when the service is unavailable"""
    pass

class UserNotFoundException(Exception):
    """Exception when a user is not found in the feature store"""
    pass

class PredictionException(Exception):
    """Exception for errors during prediction"""
    pass