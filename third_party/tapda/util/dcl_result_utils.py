"""Evaluation selection and result serialization helpers."""

import json
import os

import numpy as np

def select_eval_logits(logits1, logits2, eval_mode):
    if eval_mode == 'f1':
        return logits1
    if eval_mode == 'f2':
        return logits2
    return 0.5 * (logits1 + logits2)

def save_dataset_result(output_dir, iDataSet, seed, test_result, class_acc, kappa, best_test_acc, eval_mode):
    result = {
        'dataset_index': int(iDataSet),
        'seed': int(seed),
        'eval_mode': eval_mode,
        'acc1': float(test_result['acc1']),
        'acc2': float(test_result['acc2']),
        'acc_ensemble': float(test_result['acc_ensemble']),
        'acc_eval': float(test_result['acc_eval']),
        'class_accuracy': [float(x) for x in class_acc],
        'kappa': float(kappa),
        'best_test_acc_so_far': float(best_test_acc)
    }
    result_path = os.path.join(output_dir, f'dataset_{iDataSet:02d}_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def save_summary(output_dir, acc, A, k, best_test_acc, num_classes, args, acc_f1, acc_f2, acc_ensemble):
    AA = np.mean(A, 1)
    summary = {
        'eval_mode': args.eval_mode,
        'best_test_acc': float(best_test_acc),
        'average_OA': float(np.mean(acc)),
        'std_OA': float(np.std(acc)),
        'average_AA': float(100 * np.mean(AA, 0)),
        'std_AA': float(100 * np.std(AA)),
        'average_kappa': float(100 * np.mean(k)),
        'std_kappa': float(100 * np.std(k)),
        'average_OA_f1': float(np.mean(acc_f1)),
        'average_OA_f2': float(np.mean(acc_f2)),
        'average_OA_ensemble': float(np.mean(acc_ensemble)),
        'class_accuracy_mean': [float(x) for x in (100 * np.mean(A, 0))],
        'class_accuracy_std': [float(x) for x in (100 * np.std(A, 0))],
        'num_classes': int(num_classes)
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def evaluate_results(acc, A, k, train_end, train_start, test_end, test_start, best_test_acc,
                     num_classes, args, acc_f1, acc_f2, acc_ensemble):
    AA = np.mean(A, 1)
    AAMean = np.mean(AA, 0)
    AAStd = np.std(AA)
    AMean = np.mean(A, 0)
    AStd = np.std(A, 0)
    OAMean = np.mean(acc)
    OAStd = np.std(acc)
    kMean = np.mean(k)
    kStd = np.std(k)
    print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
    print("test time per DataSet(s): " + "{:.5f}".format(test_end - test_start))
    print("eval mode: " + str(args.eval_mode))
    print("best accuracy: " + "{:.2f}".format(best_test_acc))
    print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
    print("average OA f1/f2/ensemble: {:.2f} / {:.2f} / {:.2f}".format(
        float(np.mean(acc_f1)), float(np.mean(acc_f2)), float(np.mean(acc_ensemble))
    ))
    print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
    print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
    print("accuracy for each class: ")
    for i in range(num_classes):
        print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
