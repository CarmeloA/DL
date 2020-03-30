import _pickle as pickle

f = open('/home/xt/mmdetection/result.pkl','rb')
info = pickle.load(f)
print(info)