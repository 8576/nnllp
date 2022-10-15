import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from models.FastText import Config, Model
from dp import Createds
import logging
import logging.config

# logging 
logging.config.fileConfig("logconfig.ini")
log = logging.getLogger("info")

def ft(msg):
    return "=" * 10 + "{}".format(msg) + "=" * 10


config = Config("", "")
model = Model(config=config)
model = model.to(config.device)
model.load_state_dict(config.save_path)
model.eval()

test_data = DataLoader(Createds(data_file="testset.csv", config=config), batch_size=128)
predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)
with torch.no_grad():
    for label, text in test_data:
        outputs = model(text)
        loss = F.cross_entropy(outputs, labels)
        labels = labels.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

acc = metrics.accuracy_score(labels_all, predict_all)
report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
confusion = metrics.confusion_matrix(labels_all, predict_all)


print(ft("acc"), acc, sep="\n")
print(ft("report"), report, sep="\n")
print(ft("confusion"), confusion, sep="\n")



