import torch
from torchmetrics.classification import MulticlassAveragePrecision #area under precision recall curve
from torchmetrics.classification import MulticlassPrecision # Precision
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassAUROC
import warnings

# Filter out UserWarning related to torch.metrics
warnings.filterwarnings("ignore", ".*nan.*") #category=UserWarning, module="torchmetrics.classification")
warnings.filterwarnings("ignore", ".*positive samples.*") 


# aggiungere logits e targets

class ClassificationMetricsEpoch:
    def __init__(self, num_classes,device):
        self.num_classes = num_classes
        self.device = 'cpu'
        #self.metric = MulticlassConfusionMatrix(num_classes=num_classes)
        #self.additional_token_GT = []
        #self.additional_token_predicted = []
        self.reset()

    def reset(self):
        self.metric = MulticlassConfusionMatrix(num_classes=self.num_classes).to(self.device)
        self.accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro').to(self.device)
        self.precision =  MulticlassPrecision(num_classes=self.num_classes, average='macro').to(self.device)
        self.recall = MulticlassRecall(num_classes=self.num_classes, average='macro').to(self.device)
        self.auc_roc = MulticlassAUROC(num_classes=self.num_classes, average='macro').to(self.device)
        self.MAP = MulticlassAveragePrecision(num_classes=self.num_classes, average='macro').to(self.device)
        self.pred = torch.tensor([])
        self.GT = torch.tensor([],dtype= torch.int16)
        #self.accuracy_batch = []
        #self.precision_batch = []
        #self.recall_batch = []
        #self.f1_score_batch = []
        #self.auc_roc_batch = []
        #self.MAP_batch = []

        

    def update_batch(self, logits, targets):
        GT_indexes= (torch.argmax(targets, dim=1)-1).to(self.device) #? togliere -1 
        probabilities = torch.softmax(logits, dim=1).to(self.device)
        predicted_indexes = torch.argmax(probabilities, dim=1).to(self.device)
        self.pred = torch.cat((self.pred, probabilities), dim=0).to(self.device)
        self.GT = torch.cat((self.GT, GT_indexes), dim=0).to(self.device)
        #Confusion Matrix
        self.metric.update(predicted_indexes, GT_indexes)
        '''
        #Accuracy
        accuracy = self.accuracy(predicted_indexes, GT_indexes).item()
        self.accuracy_batch.append(accuracy)
        #Precision
        precision = self.precision(predicted_indexes, GT_indexes).item()
        self.precision_batch.append(precision)
        #Recall
        recall = self.recall(predicted_indexes, GT_indexes).item()
        self.recall_batch.append(recall)
        #F1_score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall)!=0 else 0
        self.f1_score_batch.append(f1_score)
        #AUC ROC
        auc_roc = self.auc_roc(probabilities, GT_indexes).item()
        self.auc_roc_batch.append(auc_roc)
        #MAP
        MAP = self.MAP(probabilities, GT_indexes).item()
        self.MAP_batch.append(MAP)
        '''

        return #accuracy, precision, recall, f1_score, auc_roc, MAP
        # MAP
        #mean_average_precision = self.multiclass_average_precision(probabilities.cpu(), GT_indexes.cpu())
        #self.MAP_batch.append(mean_average_precision)

        # AVERAGE PRECISION
        #average_precision = self.average_precision(predicted_indexes.cpu(), GT_indexes.cpu())
        #self.AP_batch.append(average_precision)

        # AVERAGE RECALL
        #average_recall = self.average_recall(predicted_indexes.cpu(), GT_indexes.cpu())
        #self.Recall_batch.append(average_recall)
        #print( 'average_precision: ', average_precision.cpu(), 'average_recall: ', average_recall.cpu())
        return  #mean_average_precision, average_precision, average_recall
        # Confusion_Matrix = self.metric(predicted_indexes, GT_indexes, threshold = 0.5)
        # calculate averege first statistics
        # TP, FP, FN, TN = self.average_first_statistics(Confusion_Matrix)
        # # update statistics
        # self.TP += TP
        # self.FP += FP
        # self.FN += FN
        # self.TN += TN
        # return self.TP, self.FP, self.FN, self.TN
        #calculate averege first statistics

    def statistics_epoch(self):
        #calculate average stats for the epoch
        #accuracy_epoch = sum(self.accuracy_batch)/len(self.accuracy_batch)
        #precision_epoch = sum(self.precision_batch)/len(self.precision_batch)
        #recall_epoch = sum(self.recall_batch)/len(self.recall_batch)
        #f1_score_epoch = sum(self.f1_score_batch)/len(self.f1_score_batch)
        #auc_roc_epoch = sum(self.auc_roc_batch)/len(self.auc_roc_batch)
        #MAP_epoch = sum(self.MAP_batch)/len(self.MAP_batch)
        CM = self.metric.compute()
        accuracy_epoch, precision_epoch, recall_epoch, average_specificity, f1_score_epoch = self.calculate_average_stats_from_confusion_matrix(CM)
        #accuracy_epoch = self.accuracy(self.pred, self.GT).item()
        #precision_epoch = self.precision(self.pred, self.GT).item()
        #recall_epoch = self.recall(self.pred, self.GT).item()
        #f1_score_epoch = 2 * (precision_epoch * recall_epoch) / (precision_epoch + recall_epoch) if (precision_epoch + recall_epoch)!=0 else 0
        auc_roc_epoch = self.auc_roc(self.pred, self.GT).item()
        MAP_epoch = self.MAP(self.pred, self.GT).item()
        self.reset()
        #self.AP_epoch = sum(self.AP_batch)/len(self.AP_batch)
        #self.Recall_epoch = sum(self.Recall_batch)/len(self.Recall_batch)
        #
        return round(accuracy_epoch,4), round(precision_epoch,4), round(recall_epoch,4), round(f1_score_epoch,4), round(auc_roc_epoch,4), round(MAP_epoch,4)
        
        
    def calculate_average_stats_from_confusion_matrix(self, confusion_matrix):
        accuracy = []
        precision = []
        recall = []
        specificity = []
        f1_score = []

        for index in torch.unique (self.GT):#range(self.num_classes):
            TP = (confusion_matrix[index][index]).item()
            FP = (confusion_matrix.sum(dim=0)[index] - TP).item()
            FN = (confusion_matrix.sum(dim=1)[index] - TP).item()
            TN = (confusion_matrix.sum() - (FP + FN + TP)).item()
            # calculate metrics
            accuracy_elem = 1 if (TP + FP + FN + TN)==0 else (TP + TN) / (TP + FP + FN + TN)
            precision_elem = 1 if (TP + FP)==0 else TP / (TP + FP)
            recall_elem = 1 if (TP + FN)==0 else TP / (TP + FN)
            specificity_elem = 1 if (TN+FP)==0 else TN / (TN + FP)
            f1_score_elem =0 if (precision_elem + recall_elem)==0 else (2 * precision_elem * recall_elem) / (precision_elem + recall_elem)
            accuracy.append(accuracy_elem)
            precision.append(precision_elem)
            recall.append(recall_elem)
            specificity.append(specificity_elem)
            f1_score.append(f1_score_elem)

        avg_accuracy = round(sum(accuracy)/len(accuracy),4)
        avg_precision = round(sum(precision)/len(precision),4)
        avg_recall = round(sum(recall)/len(recall),4)
        avg_specificity = round(sum(specificity)/len(specificity),4)
        avg_f1_score = round(sum(f1_score)/len(f1_score),4)
        return avg_accuracy, avg_precision, avg_recall, avg_specificity, avg_f1_score

    #def statistics_test_additional_token_batch(self, logits, targets):
        #probabilities = torch.softmax(logits, dim=1)
        #GT_indexes= torch.argmax(targets, dim=1)
        #predicted_indexes = torch.argmax(probabilities, dim=1)

        #self.additional_token_GT.append(GT_indexes)
        #self.additional_token_predicted.append(predicted_indexes)
        #self.metric.update(predicted_indexes, GT_indexes)
    
    #def statistics_test_additional_token_epoch(self):


    def start_epoch(self):
        self.reset()


    # def average_first_statistics (confusion_matrix):
    #     TP = torch.diag(confusion_matrix)
    #     FP = confusion_matrix.sum(dim=0) - TP
    #     FN = confusion_matrix.sum(dim=1) - TP
    #     TN = confusion_matrix.sum() - (FP + FN + TP)
    #     return TP, FP, FN, TN

    # def second_statistics (TP, FP, FN, TN):
    #     # calculate metrics
    #     accuracy = 1 if (TP + FP + FN + TN)==0 else (TP + TN) / (TP + FP + FN + TN)
    #     precision = 1 if (TP + FP)==0 else TP / (TP + FP)
    #     recall = 1 if (TP + FN)==0 else TP / (TP + FN)
    #     specificity = 1 if (TN+FP)==0 else TN / (TN + FP)
    #     f1_score = (2 * precision * recall) / (precision + recall)
    #     return accuracy, precision, recall, specificity, f1_score

    

    

    
