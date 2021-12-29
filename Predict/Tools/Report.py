from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def MakeReport(type, y_test, y_pred, argument, *ypred):
    if len(ypred) == 1:
        AUC_value = str(metrics.roc_auc_score(y_test, ypred[0]))[:7]
    elif len(ypred) == 0:
        AUC_value = "NULL"
    else:
        AUC_value = "False"

    ACC_value = str(metrics.accuracy_score(y_test, y_pred))[:7]
    Recall_value = str(metrics.recall_score(y_test, y_pred))[:7]
    F1_Score_value = str(metrics.f1_score(y_test, y_pred))[:7]
    Precesion_value = str(metrics.precision_score(y_test, y_pred))[:7]
    ConfusionMatrix = str(metrics.confusion_matrix(y_test, y_pred))
    Report = classification_report(y_test, y_pred)
    str_argument = ""
    for key, value in argument.items():
        str_argument += f"{key} : {value}\n"

    Report_txt = f"{'='*50}\ntype : {type}\n{str_argument}{'-'*50}\nAUC : {AUC_value}\nACC_value : {ACC_value}\nRecall_value : {Recall_value}\nF1_Score_value : {F1_Score_value}\nPrecesion_value : {Precesion_value}\nConfusionMatrix : \n{ConfusionMatrix}\n{'-'*50}\nReport : \n{Report}\n{'='*50}\n"

    print(Report_txt)

    print("f1 is :", f1_score(y_test, y_pred, average="binary"))
    # None：返回每一类各自的f1_score，得到一个array。
    # "binary":返回由pos_label指定的类的f1_score。
    # p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_test,y_pred,labels=[0,1])
    # print(p_class)
    # print(r_class)
    # print(f_class)
    # print(support_micro)

    with open("./Predict/Log/Log.txt", "a+") as f:
        f.write(Report_txt)

    return (
        AUC_value,
        ACC_value,
        Recall_value,
        F1_Score_value,
        Precesion_value,
        ConfusionMatrix,
        Report_txt
    )
