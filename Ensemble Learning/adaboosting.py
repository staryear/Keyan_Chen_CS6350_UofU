import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def get_sub_tree(train_data, label_name, label, feature,max_depth):
    attribute_tree_dict = {}
    value_count_feat = train_data[feature].value_counts(sort=False)
    #print(value_count_feat)
    for i,count in value_count_feat.items():
        sub_feature_data = train_data[train_data[feature] == i]

        is_data_pure = False
        max_type_count = -100
        max_type = None
        for type in label:
            type_count = sub_feature_data[sub_feature_data[label_name] == type].shape[0]
            type_weight = sum(sub_feature_data[sub_feature_data[label_name] == type]["weight"])
            if type_weight > max_type_count:
                max_type_count = type_weight
                max_type = type
            if type_count == count:
                is_data_pure = True
                attribute_tree_dict[i] = type
                train_data = train_data[train_data[feature] != i]
        if not is_data_pure:
            if max_depth > 1:
                attribute_tree_dict[i] = "multi"
            else:
                attribute_tree_dict[i] = max_type
        #print(i, attribute_tree_dict[i])
    return train_data, attribute_tree_dict

def get_tree(train_data, label_name, label, pre_feature, node, max_depth, method):
    if train_data.shape[0] > 0 and max_depth > 0:
        best_feature = get_best_feature(train_data, label_name, label, method)
        #print("get tree best feature" + best_feature + "\t" + str(max_depth))
        train_data, tree_dict = get_sub_tree(train_data, label_name, label, best_feature, max_depth)
        next_node = None

        if pre_feature == None:
            node[best_feature] = tree_dict
            next_node = node[best_feature]
        else:
            node[pre_feature] = dict()
            node[pre_feature][best_feature] = tree_dict
            next_node = node[pre_feature][best_feature]

        for i,type in list(next_node.items()):
            if type == "multi":
                sub_feature_data = train_data[train_data[best_feature] == i]
                get_tree(sub_feature_data, label_name, label, i, next_node, max_depth-1, method)

def get_id3_tree(train_data, label_name, label,max_depth, method):
    tree_dict = {}
    get_tree(train_data, label_name, label, None, tree_dict, max_depth, method)
    return tree_dict
def get_GI(train_data, label_name, label):
    gini_index = 0
    total_size = train_data.shape[0]
    for i in label:
        label_count = train_data[train_data[label_name] == i].shape[0]
        label_gini_index = pow(label_count/total_size,2)
        gini_index += label_gini_index
        #print(i,label_count,label_gini_index,gini_index)
    return 1-gini_index

def get_ME(train_data, label_name, label):
    total_size = train_data.shape[0]
    label_ME = 0
    for i in label:
        label_count = train_data[train_data[label_name] == i].shape[0]
        label_ME = max(label_count/total_size, label_ME)
        #print(i,label_count,label_ME)
    ME = 1-label_ME
    return ME

def get_entropy(train_data, label_name, label):
    entropy = 0
    total_size = sum(train_data["weight"])
    for i in label:
        label_entropy = 0
        label_count = sum(train_data[train_data[label_name] == i]["weight"])
        if label_count != 0:
            label_entropy = -(label_count/total_size)*math.log2(label_count/total_size)
        entropy += label_entropy
        #print(i,label_count,label_entropy,entropy)
    return entropy

def get_info_gain(train_data, label_name, label, feature, method):
    sub_data = train_data[feature].unique()
    total_size = sum(train_data["weight"])
    #print(sub_data)
    sum_feat_infogain = 0
    total_cal = 0
    if method == "entropy":
        total_cal = get_entropy(train_data,label_name, label)
    elif method == "gini":
        total_cal = get_GI(train_data, label_name, label)
    elif method == "me":
        total_cal = get_ME(train_data, label_name, label)
    #print(total_cal)

    for i in sub_data:
        filtered_data = train_data[train_data[feature] == i]
        filtered_data_count = sum(filtered_data["weight"])
        filtered_data_cal = 0
        if method == "entropy":
            filtered_data_cal = get_entropy(filtered_data,label_name, label)
        elif method == "gini":
            filtered_data_cal = get_GI(filtered_data, label_name, label)
        elif method == "me":
            filtered_data_cal = get_ME(filtered_data, label_name, label)
        sum_feat_infogain += filtered_data_count/total_size*filtered_data_cal

    #print(sum_feat_infogain)
    info_gain = total_cal-sum_feat_infogain
    #print(info_gain)
    return info_gain  # Press Ctrl+F8 to toggle the breakpoint.

def get_best_feature(train_data, label_name, label,method):

    feature_only = train_data.columns.drop([label_name,"weight","prediction"])
    #print(feature_only)
    max_info_gain = -1
    best_feature = None
    for i in feature_only:
        feature_gain = get_info_gain(train_data,label_name,label, i, method)
        if max_info_gain < feature_gain:
            max_info_gain = feature_gain
            best_feature = i
    #print(best_feature + "\tget_best_feature")
    return best_feature

def predict(id3_tree, instance):
    if not isinstance(id3_tree, dict):
        return id3_tree
    else:
        node = next(iter(id3_tree))
        feature = instance[node]
        if feature in id3_tree[node]:
            return predict(id3_tree[node][feature], instance)
        else:
            return None

def get_error_rate(tree, test_data, label_name):
    wrong_count = 0
    correct_count = 0
    error = 0
    predictions = []
    for index, row in test_data.iterrows():
        result = predict(tree, test_data.iloc[index])
        if result is None:
            result = test_data[label_name].iloc[index]
        if result == test_data[label_name].iloc[index]:
            correct_count += 1
        else:
            wrong_count += 1
            error += test_data["weight"].iloc[index]
        predictions.append(result)
        if result != "1" and result != "-1":
            print(index,result)
    error_rate = wrong_count / (correct_count + wrong_count)
    test_data["prediction"] = predictions
    return error


def convert_num_feature(data_set, num_feature_list):
    for num_feature in num_feature_list:
        cur_num_column = data_set.get(num_feature)
        temp_column = cur_num_column.copy(deep=True)
        mid_val = np.median(data_set.get(num_feature).to_numpy())
        temp_column.where(cur_num_column <= mid_val, "neg", inplace=True)
        temp_column.where(cur_num_column > mid_val, "pos", inplace=True)
        data_set[num_feature] = temp_column

    cur_class_column = data_set.get("class")
    temp_class = cur_class_column.copy(deep=True)
    temp_class.where(cur_class_column == "yes", "1", inplace=True)
    temp_class.where(cur_class_column == "no", "-1", inplace=True)
    data_set["class"] = temp_class
    data_set["prediction"] = np.zeros(data_set.shape[0])
    data_set["weight"] = [1/len(data_set)] * len(data_set)
    return data_set

def convert_unk_feature(data_set, feature_list):
    for feature in feature_list:
        if (data_set[feature].unique().__contains__('unknown')):
            cur_column = data_set.get(feature)
            temp_column = cur_column.copy(deep=True)
            sub_data = data_set[feature].unique()
            max_count = 0
            max_feature = None
            for i in sub_data:
                if i != "unknown":
                    filtered_data = data_set[data_set[feature] == i]
                    filtered_data_count = filtered_data.shape[0]
                    if filtered_data_count > max_count:
                        max_count = filtered_data_count
                        max_feature = i
            temp_column.where(cur_column != 'unknown', max_feature, inplace=True)
            data_set[feature] = temp_column
    return data_set

def adaboost(train_data, test_data, label_list):
    errors_of_stump = []
    errors_of_stump_test = []
    adaboost_errors = []
    adaboost_errors_test = []
    adaboost_sum = np.zeros(len(train_data))
    adaboost_sum_test = np.zeros(len(test_data))

    weights = [1/len(train_data)] * len(train_data)
    train_error = 0
    for t in range(0, 500):
        train_data["weight"] = weights
        test_data["weight"] = weights
        #print(train_data["weight"])
        weighted_decision_tree = get_id3_tree(train_data, "class", label_list, 2, "entropy")
        train_error = get_error_rate(weighted_decision_tree, train_data, 'class')
        test_error = get_error_rate(weighted_decision_tree, test_data, 'class')
        print(train_error)
        errors_of_stump.append(train_error)
        errors_of_stump_test.append(test_error)
        if train_error == 0:
            print("train error is 0")
            break
        alpha = 0.5 * np.log((1 - train_error) / train_error)
        #print("alpha "+ str(alpha))
        #print(train_data["prediction"])
        #print(train_data["class"].astype(int) * train_data["prediction"].astype(int))
        new_weights = weights * np.exp(-alpha * (train_data["class"].astype(int) * train_data["prediction"].astype(int)))
        norm = sum(new_weights)
        weights = new_weights / norm
        #print(new_weights)

        adaboost_sum = alpha * train_data["prediction"].astype(int) + adaboost_sum
        adaboost_error = (np.sign(adaboost_sum) - train_data["class"].astype(int)) / 2
        adaboost_error_sum_weights = np.sum(adaboost_error.replace(-1, 1)) / len(train_data)
        adaboost_errors.append(adaboost_error_sum_weights)
        #print(adaboost_error_sum_weights)

        adaboost_sum_test = alpha * test_data["prediction"].astype(int) + adaboost_sum_test
        adaboost_error_test = (np.sign(adaboost_sum_test) - test_data["class"].astype(int)) / 2
        adaboost_error_test_sum = np.sum(adaboost_error_test.replace(-1, 1)) / len(test_data)
        adaboost_errors_test.append(adaboost_error_test_sum)

    ploting(adaboost_errors, adaboost_errors_test, "Error", "Test Error",
                        "Adaboost Error by Count Tree")
    ploting(errors_of_stump, errors_of_stump_test, "Error", "Test Error",
                        "Stump Error by Count Tree")

def ploting(line1, line2, label1, label2, title):
    plt.plot(np.asarray(line1), label=label1)
    plt.plot(np.asarray(line2), label=label2)
    plt.title(title)
    plt.xlabel("T")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    car_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    train_data_car = pd.read_csv("./car/train.csv", names=car_columns, header=None)
    test_data_car = pd.read_csv("./car/test.csv", names=car_columns, header=None)
    label_car = ["unacc", "acc", "good", "vgood"]
    method_list = ["entropy", "gini", "me"]
    #get_entropy(train_data_car,"acc", label_car)
    #print(train_data_car.head(5))
    #get_info_gain(train_data_car,"class", label_car,"safety", "entropy")
    #get_beat_feature(train_data_car,"class", label_car)
    #get_sub_tree(train_data_car,"class", label_car,"safety", 2)
    #decision_tree = get_id3_tree(train_data_car, "class", label_car, 4, "me")

    
    for method in method_list:
        for depth in range(1, 7):
            decision_tree = get_id3_tree(train_data_car, "class", label_car, depth, method)
            #print(decision_tree)
            test_error = get_error_rate(decision_tree, test_data_car, 'class')
            train_error = get_error_rate(decision_tree, train_data_car, 'class')

            print("test_error" + "\t" + method + "\t" + str(depth) + "\t" + str(test_error))
            print("train_error" + "\t" + method + "\t" + str(depth) + "\t" + str(train_error))
    """
    bank_columns = ["age", "job", "marital", "education", "default", "balance", "housing",
                    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
                    "previous", "poutcome", "class"]

    bank_num_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    label_bank = ["1", "-1"]
    train_data_bank = pd.read_csv("./bank/train.csv", names=bank_columns, header=None)
    test_data_bank = pd.read_csv("./bank/test.csv", names=bank_columns, header=None)

    train_data_bank = convert_num_feature(train_data_bank, bank_num_columns)
    test_data_bank = convert_num_feature(test_data_bank, bank_num_columns)
    print(train_data_bank.head(5))
    print("bank1")
    adaboost(train_data_bank, test_data_bank, label_bank)
    """
    for method in method_list:
        for depth in range(1, 17):
            decision_tree = get_id3_tree(train_data_bank, "class", label_bank , depth, method)
            # print(decision_tree)
            test_error = get_error_rate(decision_tree, test_data_bank, 'class')
            train_error = get_error_rate(decision_tree, train_data_bank, 'class')

            print("test_error" + "\t" + method + "\t" + str(depth) + "\t" + str(test_error))
            print("train_error" + "\t" + method + "\t" + str(depth) + "\t" + str(train_error))

    train_data_bank_no_unk = convert_unk_feature(train_data_bank, bank_columns)
    test_data_bank_no_unk = convert_unk_feature(test_data_bank, bank_columns)

    print(train_data_bank_no_unk.head(5))
    print("bank2")

    for method in method_list:
        for depth in range(1, 17):
            decision_tree = get_id3_tree(train_data_bank_no_unk, "class", label_bank , depth, method)
            # print(decision_tree)
            test_error = get_error_rate(decision_tree, test_data_bank_no_unk, 'class')
            train_error = get_error_rate(decision_tree, train_data_bank_no_unk, 'class')

            print("test_error" + "\t" + method + "\t" + str(depth) + "\t" + str(test_error))
            print("train_error" + "\t" + method + "\t" + str(depth) + "\t" + str(train_error))
    """