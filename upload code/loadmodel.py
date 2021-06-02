import argparse
import torch
import torchaudio
import joblib
import numpy
from datasets.audio import get_audio_dataset
from models.audiocnn import AudioCNN

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--data-directory",
    type=str,
    required=True,
    help="Directory where subfolders of audio reside",
)

parser.add_argument("-e", "--num-epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=50, help="Batch size")

def predict_out():
  audio_cnn = torch.load('savedmodels/audiocnn_dp_b2_ne10.pt')
  path="/content/drive/MyDrive/LungDataset/Crack/Copy of crack_3__226_1b1_Pl_sc_LittC2SE_s6_a.wav"
  audio_tensor, sample_rate = torchaudio.load(path, normalize=True)
  with torch.no_grad():
    audio_tensor=audio_tensor.to("cuda")
    print("size of tensor:",audio_tensor.size())
    print(type(audio_tensor))
    print("size of tensor:",audio_tensor.size())
    print(audio_tensor)
    audio_tensor = torch.unsqueeze(audio_tensor,0)
    print("size of tensor:",audio_tensor.size())


    output=audio_cnn(audio_tensor)
    _, predicted = torch.max(output.data, 1)
    print("predicted:",predicted.data.cpu().numpy()[0])

def main(data_directory, num_epochs, batch_size):
    test_dataset = get_audio_dataset(
        data_directory, max_length_in_seconds=6, pad_and_truncate=True
    )

    test_dataset_length = len(test_dataset)
    # train_length = round(dataset_length * 0.8)
    # test_length = dataset_length - train_length

    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [int(train_length), int(test_length)]
    # )

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, num_workers=2
    # )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2
    )

    # train_dataloader_len = len(train_dataloader)
    test_dataloader_len = len(test_dataloader)

    # audio_cnn = AudioCNN(len(dataset.classes)).to("cuda")
    # cross_entropy = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(audio_cnn.parameters())

    audio_cnn = torch.load('savedmodels/audiocnn_dp_b2_ne10.pt')#LOADINGMODEL
    test_loss = 0
    correct = 0
    total = 0
    # c = 0
    # w = 0
    # n = 0
    # ct_c = 0 
    # ct_w = 0
    # ct_n = 0 

    with torch.no_grad():
            for sample_idx, (audio, target) in enumerate(test_dataloader):
                # print(test_dataloader)
                audio, target = audio.to("cuda"), target.to("cuda")
                # print(type(audio))  #to know the type of input
                # print("size of tensor:",audio.size())
                output = audio_cnn(audio)
                # print(audio)
                # break
                #print(output)
                test_loss += cross_entropy(output, target)

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                # print(target.data.cpu().numpy()[0])
                y_score.append(output.cpu)
                y.append(target.cpu)
                target=target.data.cpu().numpy()[0]
                predicted=predicted.data.cpu().numpy()[0]

                ny.append(predicted)
                ny_score.append(target)

                target_value.append(target)
                pred_value.append(predicted)
                # print("target, predicted:",target.data.cpu().numpy()[0],predicted.data.cpu().numpy()[0])
                # if(target == 0):
                #   c +=1
                #   if(predicted == 0):
                #     ct_c += 1
                # elif(target == 1):
                #   n +=1
                #   if(predicted == 1):
                #     ct_n += 1
                # else:
                #   w += 1
                #   if(predicted == 2):
                #     ct_w += 1
                correct += (predicted == target).sum().item()
            #Normal
            fpr, tpr, thresholds = metrics.roc_curve(target_value, pred_value, pos_label=1)
            print("FPR,TPR from ROC: ",fpr, tpr)
            normal = metrics.auc(fpr, tpr)
            #Wheeze
            fpr, tpr, thresholds = metrics.roc_curve(target_value, pred_value, pos_label=2)
            print("FPR,TPR from ROC: ",fpr, tpr)
            wheeze = metrics.auc(fpr, tpr)
            #Crack
            fpr, tpr, thresholds = metrics.roc_curve(target_value, pred_value, pos_label=0)
            print("FPR,TPR from ROC: ",fpr, tpr)
            crack = metrics.auc(fpr, tpr)
            print("AUROC of crack, normal, wheeze: ", crack, normal, wheeze)
            print("Average AUROC: ", (crack + normal + wheeze)/3)
            print(f"Evaluation loss: {test_loss.mean().item() / test_dataloader_len}")
            print(f"Evaluation accuracy: {100 * correct / total}")
            #print("Crackle, Normal, Wheeze", c,n,w)
            label = [ "Crackle" , "Normal" , "Wheeze"]
            cnf_matrix = confusion_matrix(ny, ny_score)
            pd_cm = pd.DataFrame(cnf_matrix, index = label, columns = label)
            print("Confusion Matrix")
            pd.set_option('display.max_columns', None)
            print(cnf_matrix)
            FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
            FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            TP = np.diag(cnf_matrix)
            TN = cnf_matrix.sum() - (FP + FN + TP)
            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP/(TP+FN)
            # Specificity or true negative rate
            TNR = TN/(TN+FP) 
            # Precision or positive predictive value
            PPV = TP/(TP+FP)
            # Negative predictive value
            NPV = TN/(TN+FN)
            # Fall out or false positive rate
            FPR = FP/(FP+TN)
            # False negative rate
            FNR = FN/(TP+FN)
            # False discovery rate
            FDR = FP/(TP+FP)
            # Overall accuracy for each class
            ACC = (TP+TN)/(TP+FP+FN+TN)
            
            # for i in range(0, len(FPR)):    
            #   for j in range(i+1, len(FPR)):    
            #    if(FPR[i] > FPR[j]):    
            #       temp = FPR[i];    
            #       FPR[i] = FPR[j];    
            #       FPR[j] = temp;
            #       temp = TPR[i];    
            #       TPR[i] = TPR[j];    
            #       TPR[j] = temp;    
            # # print(FPR, TPR)
            # print("AURROC: ", metrics.auc(FPR,TPR))
            print("accuracy: ", ACC , "TPR: ", TPR , "FPR: ", FPR)
            sns_plot = sns.heatmap(pd_cm,annot = True,cmap="YlGnBu")
            plt.savefig('datasets/conf_matri.png')
            plt.show()

            # print(f"Evaluation loss: {test_loss.mean().item() / test_dataloader_len}")
            # print(f"Evaluation accuracy: {100 * correct / total}")
            # print(f"TPR of crackle: {100* ct_c / c}")
            # print(f"TPR of Normal: {100* ct_n / n}")
            # print(f"TPR of Wheeze: {100* ct_w / w}")
            # print("Crackle, Normal, Wheeze", c,n,w)
if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
    # predict_out()