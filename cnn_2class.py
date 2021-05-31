# CNN 2 Class Classification
import argparse
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets.audio import get_audio_dataset
from models.audiocnn import AudioCNN
import sklearn.metrics as metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--data-directory",
    type=str,
    required=True,
    help="Directory where subfolders of audio reside",
)

parser.add_argument("-e", "--num-epochs", type=int,
                    default=10, help="Number of epochs")
parser.add_argument("-b", "--batch-size", type=int,
                    default=50, help="Batch size")


def main(data_directory, num_epochs, batch_size):
    train_data_directory = data_directory + '/Train'
    test_data_directory = data_directory + '/Valid'
    train_dataset = get_audio_dataset(
        train_data_directory, max_length_in_seconds=6, pad_and_truncate=True
    )
    test_dataset = get_audio_dataset(
        test_data_directory, max_length_in_seconds=6, pad_and_truncate=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=2, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=2, shuffle=True
    )

    train_dataloader_len = len(train_dataloader)
    test_dataloader_len = len(test_dataloader)

    audio_cnn = AudioCNN(len(train_dataset.classes)).to("cuda")
    cross_entropy = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(audio_cnn.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    max_auroc = 0
    loss_array = []
    for epoch in range(num_epochs):
        audio_cnn.train()

        for sample_idx, (audio, target) in enumerate(train_dataloader):
            audio_cnn.zero_grad()
            audio, target = audio.to("cuda"), target.to("cuda")
            output = audio_cnn(audio)
            loss = cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            print(
                f"{epoch:06d}-[{sample_idx + 1}/{train_dataloader_len}]: {loss.mean().item()}"
            )

        batchstr = str(batch_size)
        epochstr = str(num_epochs)
        currcpochstr = str(epoch)
        test_loss = 0
        correct = 0
        total = 0
        # c = 0
        # w = 0
        # n = 0
        # ct_c = 0
        # ct_w = 0
        # ct_n = 0
        target_value = []
        pred_value = []
        with torch.no_grad():
            y = []
            y_score = []

            ny = []
            ny_score = []
            for sample_idx, (audio, target) in enumerate(test_dataloader):
                audio, target = audio.to("cuda"), target.to("cuda")

                output = audio_cnn(audio)
                # print(output)
                test_loss += cross_entropy(output, target)
                #outnum = output.data.cpu().numpy()
                # print(outnum)  to Print the output array of size numclass
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                # print(target.data.cpu().numpy()[0])

                y_score.append(output.cpu)
                y.append(target.cpu)

                #print(output.cpu, target.cpu)

                target = target.data.cpu().numpy()[0]
                predicted = predicted.data.cpu().numpy()[0]

                ny.append(predicted)
                ny_score.append(target)

                target_value.append(target)
                pred_value.append(predicted)
                correct += (predicted == target).sum().item()
            # print(type(pred_value[0]))
            # print(type(target_value[0]))

            # AUROC 2 class
            fpr, tpr, thresholds = metrics.roc_curve(
                target_value, pred_value, pos_label=1)
            print("FPR,TPR from ROC: ", fpr, tpr)
            auroc = metrics.auc(fpr, tpr)
            print("AUROC: ", auroc)

            val_loss = test_loss.mean().item() / test_dataloader_len
            print(
                f"Evaluation loss: {test_loss.mean().item() / test_dataloader_len}")
            loss_array.append(test_loss.mean().item() / test_dataloader_len)
            print(f"Evaluation accuracy: {100 * correct / total}")
            #print("Crackle, Normal, Wheeze", c,n,w)
            label = ["Abnormal", "Normal"]
            cnf_matrix = confusion_matrix(ny, ny_score)
            pd_cm = pd.DataFrame(cnf_matrix, index=label, columns=label)
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
            path1 = 'savedmodels/audiocnn_2class' + batchstr
            #path += batchstr + '_ne' + epochstr + '_ce' + currcpochstr + '.pt'
            #path = path1 + 'lastmodel' + '.pt'
            path = path1 + 'lastmodel_2class' + '.pt'
            torch.save(audio_cnn, path)
            if(auroc > max_auroc):
                max_auroc = auroc
                # path = path1 + 'bestmodel.pt'
                path = path1 + 'bestmodel_2class.pt'
                torch.save(audio_cnn, path)

            print("accuracy: ", ACC, "TPR: ", TPR, "FPR: ", FPR)
            scheduler.step(val_loss)
            sns_plot = sns.heatmap(pd_cm, annot=True, cmap="YlGnBu")
            plt.savefig('datasets/conf_matrix_2Class.png')
            plt.show()
            los_x = list(range(epoch+1))
            plt.title("Learning Curve")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(los_x, loss_array)
            plt.savefig('datasets/learningmatrix_2Class')


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
