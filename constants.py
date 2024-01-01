# K for intra-layer edges:
K = 5

# Threshold for intra-layer edges:
THRESHOLD = 0.05


LR = 1e-3  # Learning rate for optimizer
DECAY = 5e-4  # Weight-decay for optimizer

EPOCHS = 201  # Training epochs


def timeit(func):
    """
    Decorator for timing functions.
    """
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        print(f'{func.__name__} took {end} seconds')
        return result
    return wrapper
