class_name_path_k400 = '$ROOT/label_rephrase/k400_rephrased_classes.json'
class_name_path_ucf = '$ROOT/label_rephrase/ucf101_rephrased_classes.json'
class_name_path_hmdb = '$ROOT/label_rephrase/hmdb_rephrased_classes.json'

class_name_path_k600_1 = '$ROOT/label_rephrase/k600_split1_rephrased_classes.json'
class_name_path_k600_2 = '$ROOT/label_rephrase/k600_split2_rephrased_classes.json'
class_name_path_k600_3 = '$ROOT/label_rephrase/k600_split3_rephrased_classes.json'

finetuned_performance_file_path_ucf =''
ensemble_performance_file_path_ucf = ''

finetuned_performance_file_path_hmdb =''
ensemble_performance_file_path_hmdb = ''

finetuned_performance_file_path_k600_1 =''
ensemble_performance_file_path_k600_1 = ''

finetuned_performance_file_path_k600_2 =''
ensemble_performance_file_path_k600_2 = ''

finetuned_performance_file_path_k600_3 =''
ensemble_performance_file_path_k600_3 = ''

class_name_path = class_name_path_ucf
class_name_path_k400 = class_name_path_k400
finetuned_performance_file_path = finetuned_performance_file_path_ucf
ensemble_performance_file_path = ensemble_performance_file_path_ucf

import clip
import pickle

class_name = pickle.load(open(class_name_path, 'rb'))
class_name_k400 = pickle.load(open(class_name_path_k400, 'rb'))

finetuned_performance = pickle.load(open(finetuned_performance_file_path, 'rb'))
ensemble_performance = pickle.load(open(ensemble_performance_file_path, 'rb'))

device = 'cuda:0'
model, preprocess = clip.load("ViT-B/16", device)

class_name_feat = model.encode_text(
                    clip.tokenize(
                        ['%s'%v for k, v in class_name]
                        ).to(device)
                    )

class_name_feat = class_name_feat / class_name_feat.norm(dim=-1, p=2, keepdim=True)


class_name_k400_feat = model.encode_text(
                    clip.tokenize(
                        ['%s'%v for k, v in class_name_k400]
                        ).to(device)
                    )

class_name_k400_feat = class_name_k400_feat / class_name_k400_feat.norm(dim=-1, p=2, keepdim=True)

correlation = class_name_feat @ class_name_k400_feat.T
correlation_idx = correlation.max(dim=-1)[0].sort(descending=True)[1]

finetuned_predicts, finetuned_gt = finetuned_performance[0].max(dim=-1)[1], finetuned_performance[1]
ensemble_predicts, ensmeble_gt = ensemble_performance[0].max(dim=-1)[1], ensemble_performance[1]

label_list = list(range(len(set(finetuned_gt))))
finetuned_predict_dict = {}
ensemble_predict_dict = {}

finetuned_accuracy = []
ensemble_accuracy = []
diff_accuracy = []

for l in label_list:
    label_mask = (finetuned_gt == l)
    class_count = label_mask.sum()
    finetuned_correct_num = (finetuned_predicts == finetuned_gt) * label_mask
    finetuned_correct_num = finetuned_correct_num.sum()
    finetuned_predict_dict[str(l)] = (finetuned_correct_num / class_count)
    finetuned_accuracy.append(finetuned_predict_dict[str(l)])
    
    label_mask = (ensmeble_gt == l)
    class_count = label_mask.sum()
    ensemble_correct_num = (ensemble_predicts == ensmeble_gt) * label_mask
    ensemble_correct_num = ensemble_correct_num.sum()
    ensemble_predict_dict[str(l)] = (ensemble_correct_num / class_count)
    ensemble_accuracy.append(ensemble_predict_dict[str(l)])

    diff_accuracy.append(ensemble_predict_dict[str(l)] - finetuned_predict_dict[str(l)])

print(finetuned_accuracy)
print(ensemble_accuracy)
print(diff_accuracy[correlation_idx])