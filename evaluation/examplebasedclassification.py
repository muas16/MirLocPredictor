def subsetAccuracy(y_test, predictions):
    """
    The subset accuracy evaluates the fraction of correctly classified examples

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    subsetaccuracy : float
        Subset Accuracy of our model
    """
    subsetaccuracy = 0.0

    for i in range(len(y_test)):
        same = True
        for j in range(len(y_test[1])):
            if y_test[i][ j] != predictions[i][ j]:
                same = False
                break
        if same:
            subsetaccuracy += 1.0

    return subsetaccuracy / len(y_test)


def hammingLoss(y_test, predictions):
    """
    The hamming loss evaluates the fraction of misclassified instance-label pairs

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    hammingloss : float
        Hamming Loss of our model
    """
    hammingloss = 0.0
    for i in range(len(y_test)):
        aux = 0.0
        for j in range(len(y_test[1])):
            if int(y_test[i][ j]) != int(predictions[i][ j]):
                aux = aux + 1.0
        aux = aux / len(y_test[1])
        hammingloss = hammingloss + aux

    return hammingloss / len(y_test)


def accuracy(y_test, predictions):
    """
    Accuracy of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracy : float
        Accuracy of our model
    """
    accuracy = 0.0

    for i in range(len(y_test)):
        intersection = 0.0
        union = 0.0
        for j in range(len(y_test[1])):
            # a = int(y_test[i][ j])
            # b = int(predictions[i][j])
            if int(y_test[i][ j]) == 1 or int(predictions[i][ j]) == 1:
                union += 1
            if int(y_test[i][ j]) == 1 and int(predictions[i][j]) == 1:
                intersection += 1

        if union != 0:
            accuracy = accuracy + float(intersection / union)

    accuracy = float(accuracy / len(y_test))

    return accuracy


def precision(y_test, predictions):
    """
    Precision of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precision : float
        Precision of our model
    """
    precision = 0.0

    for i in range(len(y_test)):
        intersection = 0.0
        hXi = 0.0
        for j in range(len(y_test[1])):
            hXi = hXi + int(predictions[i][ j])
            if int(y_test[i][ j]) == 1 and int(predictions[i][ j]) == 1:
                intersection += 1
        # print("hXi = ",hXi)
        if hXi != 0:
            precision = precision + float(intersection / hXi)

    precision = float(precision / len(y_test))

    return precision


def recall(y_test, predictions):
    """
    Recall of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recall : float
        recall of our model
    """
    recall = 0.0

    for i in range(len(y_test)):
        intersection = 0.0
        Yi = 0.0
        for j in range(len(y_test[1])):
            Yi = Yi + int(y_test[i][ j])

            if int(y_test[i][ j]) == 1 and int(predictions[i][ j]) == 1:
                intersection = intersection + 1
        # print("hXi = ",Yi)
        if Yi != 0:
            recall = recall + float(intersection / Yi)

    recall = recall / len(y_test)
    return recall


def fbeta(y_test, predictions, beta=1):
    """
    FBeta of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbeta : float
        fbeta of our model
    """
    pr = precision(y_test, predictions)
    re = recall(y_test, predictions)

    num = float((1 + pow(beta, 2)) * pr * re)
    den = float(pow(beta, 2) * pr + re)

    if den != 0:
        fbeta = num / den
    else:
        fbeta = 0.0
    return fbeta


def f1_scor(y_test, predictions):
    """
    FBeta of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbeta : float
        fbeta of our model
    """
    # pr = precision(y_test, predictions)
    # re = recall(y_test, predictions)
    #
    # num = (2 * float(pr) / float(re))

    # return num / len(y_test)






    #============================= junaid_iqbal_code ========================================
    f1 = 0.0
    for i in range(len(y_test)):
        intersection = 0.0
        hXi = 0.0
        Yi = 0.0
        for j in range(len(y_test[1])):
            hXi = hXi + int(predictions[i][ j])
            Yi = Yi + int(y_test[i][ j])
            if int(y_test[i][ j]) == 1 and int(predictions[i][ j]) == 1:
                intersection += 1
            d = hXi + Yi
        if d!=0:
            f1+= float(2*(intersection))/float(d)
    f1_s = f1/len(y_test)
    return f1_s