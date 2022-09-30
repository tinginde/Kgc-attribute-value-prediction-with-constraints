import pickle

def read_pickle(filename):
    with open(filename,'rb') as fr:
        df = pickle.load(fr)
        return df
    
#print(read_pickle('dict_a2ev.pickle'))
print(read_pickle('saved_all2idx.pkl')['P1279'])
