# parse arguments ([ConfigArgParse](https://github.com/bw2/ConfigArgParse))
from config import config
from data import data
from model import model
import os, torch, numpy as np
args = config.parse()



# seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)



# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)



# load data
dataset, train, test = data.load(args)
print("length of train is", len(train))



# # initialise model
baseline = model.initialise(dataset, args)
base_GCN = model.train(baseline, dataset, train, args)
acc = model.test(base_GCN, dataset, test, args)
print("The GCN base line accuracy:", float(acc))

args.mediators = True
args.fast = False
dataset, train, test = data.load(args)
GCN_medi = model.initialise(dataset, args)
mediator_GCN = model.train(GCN_medi, dataset, train, args)
acc_medi = model.test(mediator_GCN, dataset, test, args)
print("The GCN with mediator model accuracy:", float(acc_medi))

args.mediators = True
args.fast = True
dataset_for_fast, train_for_fast, test_for_fast = data.load(args)
GCN_fast = model.initialise(dataset_for_fast, args)
FastGCN = model.train(GCN_fast, dataset_for_fast, train_for_fast, args)
acc_fast = model.test(FastGCN, dataset_for_fast, test_for_fast, args)
print("The fast GCN  model accuracy:", float(acc_fast))

