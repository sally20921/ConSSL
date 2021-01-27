import re

def underscore(word):
    """
    Make an underscored, lowercase form from the expression in the string.
    Example::
        >>> underscore("DeviceType")
        'device_type'
    As a rule of thumb you can think of :func:`underscore` as the inverse of
    :func:`camelize`, though there are cases where that does not hold::
        >>> camelize(underscore("IOError"))
        'IoError'
    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    print(word.lower())

if __name__ == "__main__":
    data = "KNNMonitor"
    data2 = "SimCLRTransform"
    data3 = "MNISTDataset"
    data4 = "STL10Dataset"
    data5 = "CIFAR100Dataset"
    print(underscore(data5))
    print(underscore(data3))
    print(underscore(data4))
    print(underscore(data2))
    print(underscore(data))

