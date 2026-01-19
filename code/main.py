from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from model import create_multi_label_model
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
GPU_NO = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NO


def multi_evaluate(y_pred, y_test, M):
    N = len(y_test)
    count_Aiming = 0
    count_Coverage = 0
    count_Accuracy = 0
    Aiming = 0
    Coverage = 0
    Accuracy = 0
    Absolute_True = 0
    Absolute_False = 0
    for i in range(N):
        union_set_len = np.sum(np.maximum(y_pred[i], y_test[i]))
        inter_set_len = np.sum(np.minimum(y_pred[i], y_test[i]))
        y_pred_len = np.sum(y_pred[i])
        y_test_len = np.sum(y_test[i])
        if y_pred_len > 0:
            Aiming += inter_set_len / y_pred_len
            count_Aiming = count_Aiming + 1
        if y_test_len > 0:
            Coverage += inter_set_len / y_test_len
            count_Coverage = count_Coverage + 1
        if union_set_len > 0:
            Accuracy += inter_set_len / union_set_len
            count_Accuracy = count_Accuracy + 1
        Absolute_True += int(np.array_equal(y_pred[i], y_test[i]))
        Absolute_False += (union_set_len - inter_set_len) / M
    Aiming = Aiming / count_Aiming
    Coverage = Coverage / count_Coverage
    Accuracy = Accuracy / count_Accuracy
    Absolute_True = Absolute_True / N
    Absolute_False = Absolute_False / N
    return Aiming, Coverage, Accuracy, Absolute_True, Absolute_False



PATH_SEQ = "../feature/miRNA_seq_feature_64_plus_stats8.csv"
PATH_MRNA_COLOC = "../feature/miRNA_mRNA_co-localization_feature_plus10.csv"
PATH_MRNA_NET = "../feature/gate_feature_mRNA_0.8_128_0.01.csv"
PATH_DRUG = "../feature/gate_feature_drug_0.8_128_0.01.csv"
PATH_DISEASE = "../feature/gate_feature_disease_0.8_128_0.01.csv"
PATH_LABEL = "../dataset/miRNA_localization.csv"
PATH_LABEL_INDEX = "../dataset/miRNA_have_loc_information_index.txt"

NUM_CLASSES = 7
CLASS_NAMES = [
    "Cytoplasm", "Exosome", "Nucleolus", "Nucleus",
    "Extracellular vesicle", "Microvesicle", "Mitochondrion"
]


N_SPLITS = 10
FOLD_ID_TO_RUN = 0
RANDOM_SEED = 42
INPUT_DIM = 466
EPOCHS = 10
BATCH_SIZE = 32
THRESHOLD = 0.5

def load_all_arrays():
    seq_df = pd.read_csv(PATH_SEQ, index_col="Index")
    mrna_coloc_df = pd.read_csv(PATH_MRNA_COLOC, header=None)
    mrna_net_df = pd.read_csv(PATH_MRNA_NET, header=None)
    drug_df = pd.read_csv(PATH_DRUG, header=None)
    disease_df = pd.read_csv(PATH_DISEASE, header=None)
    label_df = pd.read_csv(PATH_LABEL, header=None)
    label_index_df = pd.read_csv(PATH_LABEL_INDEX, header=None)

    seq_feat = seq_df.values
    mrna_coloc_feat = mrna_coloc_df.values
    mrna_net_feat = mrna_net_df.values
    drug_feat = drug_df.values
    disease_feat = disease_df.values
    labels = label_df.values

    loc_index_list = label_index_df[0].tolist()
    select_row = np.array([v == 1 for v in loc_index_list])

    return seq_feat, disease_feat, drug_feat, mrna_net_feat, mrna_coloc_feat, labels, select_row


def build_and_scale_features(seq_feat, disease_feat, drug_feat, mrna_net_feat, mrna_coloc_feat):
    merged = np.concatenate(
        (seq_feat, disease_feat, drug_feat, mrna_net_feat, mrna_coloc_feat),
        axis=1
    )
    scaler = StandardScaler()
    merged_scaled = scaler.fit_transform(merged)
    return merged_scaled


def train_one_fold(x_train, y_train):
    model = create_multi_label_model(input_shape=(INPUT_DIM,), num_classes=NUM_CLASSES)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model


def eval_one_fold(y_true, y_score):
    auc_list = [0.0] * NUM_CLASSES
    aupr_list = [0.0] * NUM_CLASSES

    for c in range(NUM_CLASSES):
        y_c_true = y_true[:, c]
        y_c_score = y_score[:, c]
        y_c_pred = (y_c_score > THRESHOLD).astype(int)

        acc = accuracy_score(y_c_true, y_c_pred)
        prec = precision_score(y_c_true, y_c_pred)
        rec = recall_score(y_c_true, y_c_pred)
        f1 = f1_score(y_c_true, y_c_pred)
        auc_val = roc_auc_score(y_c_true, y_c_score)
        aupr_val = average_precision_score(y_c_true, y_c_score)

        print(
            f"Class {CLASS_NAMES[c]} - Accuracy: {acc}, Precision: {prec}, "
            f"Recall: {rec}, F1 Score: {f1}, AUC: {auc_val}, AUPR: {aupr_val}"
        )

        auc_list[c] = auc_val
        aupr_list[c] = aupr_val

    return auc_list, aupr_list


def run_single_fold(x_all, y_all, fold_id):
    np.random.seed(RANDOM_SEED)

    fold_len = len(x_all) // N_SPLITS

    x_shuf, y_shuf = shuffle(x_all, y_all, random_state=RANDOM_SEED)

    test_l = fold_id * fold_len
    test_r = (fold_id + 1) * fold_len
    test_idx = range(test_l, test_r)
    train_idx = [k for k in range(len(x_shuf)) if k not in test_idx]

    x_train, x_test = x_shuf[train_idx], x_shuf[test_idx]
    y_train, y_test = y_shuf[train_idx], y_shuf[test_idx]

    model = train_one_fold(x_train, y_train)
    y_prob = model.predict(x_test)

    auc_list, aupr_list = eval_one_fold(y_test, y_prob)
    return auc_list, aupr_list


def print_single_fold_summary(auc_list, aupr_list):
    avg_auc = 0.0
    avg_aupr = 0.0
    print("-----------------------Single Fold Result-----------------------")
    for c in range(NUM_CLASSES):
        print(f"Class {CLASS_NAMES[c]} - AUC: {auc_list[c]}, AUPR: {aupr_list[c]}")
        avg_auc += auc_list[c]
        avg_aupr += aupr_list[c]
    avg_auc /= NUM_CLASSES
    avg_aupr /= NUM_CLASSES
    print(f"avg_AUC: {avg_auc}, avg_AUPR: {avg_aupr}")


if __name__ == "__main__":
    seq_feat, disease_feat, drug_feat, mrna_net_feat, mrna_coloc_feat, y_all, select_row = load_all_arrays()
    x_all = build_and_scale_features(seq_feat, disease_feat, drug_feat, mrna_net_feat, mrna_coloc_feat)

    y_multilabel = y_all[select_row]
    x_multilabel = x_all[select_row]

    auc_list, aupr_list = run_single_fold(x_all, y_all, FOLD_ID_TO_RUN)
    print_single_fold_summary(auc_list, aupr_list)