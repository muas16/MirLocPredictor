import pandas as pd
import simplejson as json
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, KFold
import os


stride=1
n_folds=10


def seq2ngram(seq, ngram1,ngram2=0, stride=2):
    """
    This function convert a given sequence into the specified ngrams... If you want to combine any two ngram you can explicitly
     set ngram2 to the specified ngram val. Otherwise it will convert the sequence according to ngram1

    :param seq: a string of sequence "CTGACGTACTTA"
    :param ngram1: specify the integer value for ngram conversion
    :param ngram2: specify the integer value for ngram conversion [optional]
    :param stride: window step slided over the sequence
    :return: ngrams of sequence "CTG TGA GAC CGT TAC ACT CTT TTA"
    """
    seq_ngram = " ".join([seq[i:i + ngram1] for i in range(0, len(seq),stride) if len(seq[i:i + ngram1]) == ngram1])
    if ngram2 !=0:
        seq_ngram2 = " ".join([seq[i:i + ngram2] for i in range(0, len(seq),stride) if len(seq[i:i + ngram2]) == ngram2])
        seq_ngram=seq_ngram+" "+seq_ngram2
    return seq_ngram


df1 = pd.read_csv("circSLNN_data.csv")
print(len(df1))
kf=KFold(n_splits=n_folds, random_state=42, shuffle=True)

for ngram1 in range(2, 11):
    counter = 0

    for train_index, test_index in kf.split(df1):

        x_train, x_test = df1["seq"].values[train_index], df1["seq"].values[test_index]
        Y_train1, y_test1 = df1.drop(['seq'], axis=1).values[train_index], df1.drop(['seq'], axis=1).values[test_index]

        x_train, x_valid, Y_train1, y_valid1 = train_test_split(x_train,
                                                                Y_train1, test_size=0.1,
                                                                random_state=42)

        Y_train1 = Y_train1.tolist()
        y_valid1 = y_valid1.tolist()
        y_test1 = y_test1.tolist()

        y_train = []
        y_test = []
        y_valid = []

        col_names = ['Exosome', 'Cytoplasm', 'Mitochondrion', 'Microvesicle', 'Circulating', 'Nucleus']
        for item in Y_train1:

            item_values = []
            indexes = []
            column_names = []
            item_values.append(item[-6])
            item_values.append(item[-5])
            item_values.append(item[-4])
            item_values.append(item[-3])
            item_values.append(item[-2])
            item_values.append(item[-1])
            for i in range(0, len(item_values)):
                if item_values[i] == 1:
                    indexes.append(i)

            for value in indexes:
                column_names.append(col_names[value])
            y_train.append(column_names)

        for item in y_valid1:

            item_values = []
            indexes = []
            column_names = []

            item_values.append(item[-6])
            item_values.append(item[-5])
            item_values.append(item[-4])
            item_values.append(item[-3])
            item_values.append(item[-2])
            item_values.append(item[-1])
            for i in range(0, len(item_values)):
                if item_values[i] == 1:
                    indexes.append(i)
            for value in indexes:
                column_names.append(col_names[value])
            y_valid.append(column_names)

        for item in y_test1:

            item_values = []
            indexes = []
            column_names = []

            item_values.append(item[-6])
            item_values.append(item[-5])
            item_values.append(item[-4])
            item_values.append(item[-3])
            item_values.append(item[-2])
            item_values.append(item[-1])

            for i in range(0, len(item_values)):
                if item_values[i] == 1:
                    indexes.append(i)
            for value in indexes:
                column_names.append(col_names[value])
            y_test.append(column_names)

        print("x_train ", len(x_train))
        print("y_train ", len(y_train))
        print("x_test  ", len(x_test))
        print("y_test  ", len(y_test))
        print("x_valid ", len(x_valid))
        print("y_valid ", len(y_valid))

        out_dir = "t/" +str(ngram1)+"/"+ str(counter)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for item in range(0, len(y_train)):
            jsondict = dict()
            lb = list(y_train[item])
            jsondict["doc_label"] = lb
            jsondict["doc_token"] = seq2ngram(x_train[item], ngram1=ngram1, stride=stride).split()


            with open(os.path.join(out_dir, 'train.json'), 'a', encoding='utf-8',
                      errors='ignore') as json_file:
                json.dump(jsondict, json_file, ensure_ascii=False)
                json_file.write("\n")

        for item in range(0, len(y_valid)):
            # print(item)
            jsondict = dict()
            lb = list(y_valid[item])
            jsondict["doc_label"] = lb
            jsondict["doc_token"] = seq2ngram(x_valid[item], ngram1=ngram1, stride=stride).split()

            with open(os.path.join(out_dir, 'valid.json'), 'a', encoding='utf-8',
                      errors='ignore') as json_file:
                json.dump(jsondict, json_file, ensure_ascii=False)
                json_file.write("\n")

        for item in range(0, len(y_test)):
            jsondict = dict()
            lb = list(y_test[item])
            jsondict["doc_label"] = lb
            jsondict["doc_token"] = seq2ngram(x_test[item], ngram1=ngram1, stride=stride).split()

            with open(os.path.join(out_dir, 'test.json'), 'a', encoding='utf-8',
                      errors='ignore') as json_file:
                json.dump(jsondict, json_file, ensure_ascii=False)
                json_file.write("\n")

        counter += 1

# In[ ]:




