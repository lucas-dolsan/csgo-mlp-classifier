import pickle
import os

def serialize(filepath, object):

    with open(f'{filepath}.pkl', 'wb') as fid:
        pickle.dump(object, fid)


def deserialize(filepath):
    file_exists = os.path.exists(f'{filepath}.pkl')

    if not file_exists:
        return None

    with open(f'{filepath}.pkl', 'rb') as fid:
        return pickle.load(fid)
