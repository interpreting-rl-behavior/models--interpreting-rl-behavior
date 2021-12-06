class Namespace:
    """
    Because they're nicer to work with than dictionaries
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)