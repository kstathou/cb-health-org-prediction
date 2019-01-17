import sys
import pickle
from utils import split_str

with open(sys.argv[1], 'rb') as h:
    data = pickle.load(h)

with open('../models/vectoriser.pickle', 'rb') as h:
    vec = pickle.load(h)

with open('../models/clf.pickle', 'rb') as h:
    clf = pickle.load(h)

labels = clf.predict(vec.transform(data))

with open('../data/output/labels.pickle', 'wb') as h:
    pickle.dump(labels, h)
