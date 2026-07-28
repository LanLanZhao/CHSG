import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

from DrawHyper import get_class_detail


class HSIEvaluation(object):
    def __init__(self, name) -> None:
        self.target_names = get_class_detail(name)
        self.res = {}

    @staticmethod
    def AA_andEachClassAccuracy(confusion_matrix):
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc

    def eval(self, y_test, y_pred_test):
        labels = list(range(len(self.target_names)))
        classification = classification_report(y_test, y_pred_test,
                                               labels=labels, digits=4, target_names=self.target_names,
                                               zero_division=0)
        oa = accuracy_score(y_test, y_pred_test)
        confusion = confusion_matrix(y_test, y_pred_test)
        each_acc, aa = self.AA_andEachClassAccuracy(confusion)
        kappa = cohen_kappa_score(y_test, y_pred_test)

        self.res['classification'] = str(classification)
        self.res['oa'] = oa * 100
        self.res['confusion'] = str(confusion)
        self.res['each_acc'] = str(each_acc * 100)
        self.res['aa'] = aa * 100
        self.res['kappa'] = kappa * 100
        return str(classification), oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100
