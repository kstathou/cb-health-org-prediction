import sys
import pickle
from utils import split_str, flatten_lists

def predict_health_cb(data, vectoriser, classifier):
    """Predict health labels for CB.

    Args:
        data (:obj:`list` of :obj:`tuple`): Crunchbase IDs and list of
            categories.
    Return:
        output(:obj:`list` of :obj:`dict`): Crunchbase IDS and bool.

    """
    with open(vectoriser, 'rb') as h:
        vec = pickle.load(h)

    with open(classifier, 'rb') as h:
        clf = pickle.load(h)

    # Store index.
    data_idx = [tup[0] for tup in data]
    labels = gs.predict(vec.transform(flatten_lists([tup[1] for tup in data])))

    return [{'id':id_, 'is_health':pred}
                for id_, pred in zip(data_idx, labels)]
