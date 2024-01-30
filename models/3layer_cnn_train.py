from scripts.src.data.fasta_parsing import *
from scripts.src.data.sequence_utilities import *
from scripts.src.data.dataset_build import *
from scripts.src.models.pytorch_models import *
import torch.optim as optim
import numpy as np
import copy

class cnn_model_window(nn.Module):
    def __init__(self, filter1, filter2,filter3, noyau1, noyau2, noyau3, linear1, linear2):
        super(cnn_model_window, self).__init__()
        self.conv1 = nn.Conv2d(1, filter1, (noyau1, 4), bias=False)
        self.batch1 = nn.BatchNorm2d(filter1, track_running_stats=False)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(filter1, filter2, noyau2, bias=False)
        self.conv3 = nn.Conv1d(filter2, filter3, noyau3, bias=False)
        self.flatt = nn.Flatten()
        self.dense1 = nn.Linear(((((101-(noyau1-1))//2)-(noyau2-1))-(noyau3-1))*filter3, linear1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(linear1, linear2)
        self.dense_output = nn.Linear(linear2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch1(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatt(x)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        output = self.dense_output(x)
        return output
    
class SeqDataset(Dataset):
    """
    Class for loading Data
    """

    def __init__(self, seq_path, tags_path, names_path, seq_start=False, seq_end=False, transform=None):
        """
        Args :
            seq_path (string) : path to sequences
            tags_path (string) : path to tags
            names_path (string) : path to seq names
            transform (callable, optional) : Optional transform to be applied on sample

        """
        print("Loading sequences ...")
        temp_seqs = np.load(seq_path)
        print("Formating sequences ...")
        temp_seqs = np.resize(temp_seqs, (temp_seqs.shape[0], 1, temp_seqs.shape[1], temp_seqs.shape[2]))
        if (seq_start is not False) and (seq_end is not False):
            temp_seqs = temp_seqs[:,:,seq_start:seq_end,:]
        self.seqs = torch.from_numpy(temp_seqs).float()
        print("Loading tags ...")
        self.tags = np.load(tags_path)
        print("Loading names ...")
        self.names = list(np.load(names_path))
        self.transform = transform

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = self.seqs[idx]
        tag = self.tags[idx]
        name = self.names[idx]
        if self.transform:
            seq = self.transform(seq)

        return {'seq': seq, 'tag': tag, 'name': name}

    
def load_data(path_to_data, hyper_param_batch,seq_start=False, seq_end=False, valid_start = 0.2, valid_end = 0.2, build_path=True):
    """

    """
    if build_path is True:
        seq_train_path = path_to_data + "seqs_train.npy"
        tags_train_path = path_to_data + "tags_train.npy"
        names_train_path = path_to_data + "names_train.npy"

        seq_test_path = path_to_data + "seqs_test.npy"
        tags_test_path = path_to_data + "tags_test.npy"
        names_test_path = path_to_data + "names_test.npy"

    elif build_path is dict:
        try:
            seq_train_path = path_to_data['seqs_train']
            tags_train_path = path_to_data['tags_train']
            names_train_path = path_to_data['names_train']

            seq_test_path = path_to_data['seqs_test']
            tags_test_path = path_to_data['tags_test']
            names_test_path = path_to_data['names_test']
        except KeyError:
            print("build_path has incorrect keys, keys should be the following:")
            print("seqs_train -> path to training sequences as npy file")
            print("tags_train -> path to training tags as npy file")
            print("names_train -> path to training sequence names as npy file")
            print("seqs_test -> path to testing sequences as npy file")
            print("tags_test -> path to testing tags as npy file")
            print("names_test -> path to testing sequence names as npy file")

            raise ValueError("incorrect keys")
    else:
        print("path_to_data must be dict with represented as such : ")
        print("seqs_train -> path to training sequences as npy file")
        print("tags_train -> path to training tags as npy file")
        print("names_train -> path to training sequence names as npy file")
        print("seqs_test -> path to testing sequences as npy file")
        print("tags_test -> path to testing tags as npy file")
        print("names_test -> path to testing sequence names as npy file")

        print("you can also set build_path as True if the dataset is built with the build_dataset script")

        raise ValueError("path_to_data is not dict, and build_path set to False")

    train_data = SeqDataset(seq_path=seq_train_path,
                            tags_path=tags_train_path,
                            names_path=names_train_path,
                            seq_start=seq_start,
                            seq_end=seq_end)
    test_data = SeqDataset(seq_path=seq_test_path,
                           tags_path=tags_test_path,
                           names_path=names_test_path,
                           seq_start=seq_start,
                           seq_end=seq_end)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split_start = int(np.floor(valid_start * num_train))
    split_end = int(np.floor(valid_end * num_train))
    train_idx, valid_idx = indices[split_end:], indices[:split_start]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=hyper_param_batch, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=hyper_param_batch, sampler=valid_sampler)
    test_loader = DataLoader(test_data, batch_size=hyper_param_batch)

    return train_loader, valid_loader, test_loader


def training_pytorch_model(custom_model, train_loader, valid_loader, hyper_param_batch, hyper_param_epoch,
                           criterion, optimizer, device, patience=5, keep_track=True):
    """
    This function trains a pytorch model with progress bar and early stopping

    :custom_model: PyTorch model
    :train_loader: PyTorch DataLoader (containing train data)
    :valid_loader: PyTorch DataLoader (containing validation data)
    :hyper_param_batch: int (batch size)
    :hyper_paraam_epoch: int (number of epochs)
    :criterion: PyTorch Function (loss function)
    :optimizer: Pytorch Optim (optimizer)
    :device: Pytorch Device (GPU or CPU)
    :patience: int (early stopping patience)
    :keep_track: bool (if set to True, will return tracking parameters)
    """
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # to track outputs

    valid_spear = []
    valid_pear = []

    avg_valid_spear = []
    avg_valid_pear = []
    early_id = 0

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for e in range(hyper_param_epoch):

        early_id = e

        print(" ")
        print('Epoch: %d/%d' % (e + 1, hyper_param_epoch))
        # progress bar
        kbar = Kbar(target=(len(train_loader) + len(valid_loader))*hyper_param_batch, width=12)

        custom_model.train()  # Sets the model in training mode, which changes the behaviour of dropout layers...

        for i_batch, item in enumerate(train_loader):
            seq = item['seq'].to(device)
            tag = item['tag'].to(device)

            # Forward pass
            outputs = custom_model(seq)

            tag = tag.view(-1, 1)
            loss = criterion(outputs, tag)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            
            try:
                spearman = spearmanr(tag.tolist(), outputs.tolist())[0]
                pearson = pearsonr(tag.tolist(), outputs.tolist())[0][0]
            except TypeError:
                spearman=0
                pearson=0
            kbar.update(i_batch * hyper_param_batch, values=[("pearson", pearson),
                                                             ("spearman", spearman)])

        custom_model.eval()  # Sets the model in training mode, which changes the behaviour of dropout layers...

        for i_val, item_val in enumerate(valid_loader):
            seq_valid = item_val['seq'].to(device)
            tag_valid = item_val['tag'].to(device)

            outputs_valid = custom_model(seq_valid)
            tag_valid = tag_valid.view(-1, 1)

            spearman = spearmanr(tag_valid.tolist(), outputs_valid.tolist())[0]
            pearson = pearsonr(tag_valid.tolist(), outputs_valid.tolist())[0][0]

            valid_spear.append(spearman)
            valid_pear.append(pearson)

            loss_v = criterion(outputs_valid, tag_valid)

            valid_losses.append(loss_v.item())

            values = [("pe_valid", pearson), ("sp_valid", spearman)]
            kbar.update((i_val + i_batch) * hyper_param_batch, values=values)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        avg_valid_pear.append(np.average(valid_pear))
        avg_valid_spear.append(np.average(valid_spear))

        train_losses = []
        valid_losses = []
        valid_spear = []
        valid_pear = []

        print(" ")

        early_stopping(valid_loss, custom_model)

        if early_stopping.early_stop:
            early_id = e 
            print(" Early Stopping ")
            break

        print_msg = (f'train_loss: {train_loss:.5f} ' + f'average_valid_loss: {valid_loss:.5f} '
                     + f'average pearson corr: {avg_valid_pear[-1]:.5f} '
                     + f'average spearman corr: {avg_valid_spear[-1]:.5f}')
        print("")
        print(print_msg)

    #custom_model.load_state_dict(torch.load('checkpoint.pt'))

    if keep_track is True:
        return avg_train_losses, avg_valid_losses, avg_valid_spear, avg_valid_pear, early_id
    else :
        return 0


def test_pytorch_model(custom_model, test_loader, device, verbose=True, apply_mode=False):
    """
    This function tests a PyTorch model.

    :custom_model: PyTorch model
    :test_loader: PyTorch DataLoader (contains testing set)
    :device: PyTorch device (GPU or CPU)
    :verbose: bool (if set to True, will display Spearman and Pearson correlations).
    :apply_mode: bool (if set to True, will keep track of names as well as tags).
    """
    # initialize lists to monitor test loss and accuracy
    preds = []
    true = []
    
    if apply_mode:
        names=[]
    

    custom_model.train(False)  # prep model for evaluation

    for item in test_loader:
        seq = item['seq'].to(device)
        tag = item['tag'].to(device)
        

        tag = tag.view(-1, 1)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = custom_model(seq)

        true += tag.tolist()
        preds += output.tolist()
        
        if apply_mode:
            names += item['name']

    if verbose is True:
        print("Spearman correlation: ", spearmanr(preds, true)[0])
        print("Pearson correlation: ", pearsonr(preds,true)[0][0])

    if apply_mode:
        return preds, true, names
    else:
        return spearmanr(preds, true)[0], pearsonr(preds,true)[0][0]



def format_path(path):
    real_path = ""
    for c in path:
        if c != '/':
            real_path += c
        else:
            real_path += '|'
    return real_path

def format_classes(classes, human_classes):
    result = []
    for c in classes:
        if (c in human_classes):
            result.append(c)
            print(str(c)+' in human_classes')

    return result

def write_results(classes, on_human, path_to_result="human_exp_results.txt"):
    print('Writting results')
    f = open(path_to_result, "w")
    f.write("Class" + "\t" + "Human_spearman_corr" + "\n")
    for i in range(len(classes)):
        f.write(classes[i] + "\t" + str(on_human[i]) + "\n")

    f.close

    print('End Writting results')

print('Starting training')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
hyper_param_epoch = 50
hyper_param_batch = 256
hyper_param_learning_rate = 0.000558481641412439
base_ratio = 0.1

# These classes are the ones used in the paper, you are free to change this list in order to train the models that you want.
"""
classes = ['T', 'GT', 'AC', 'CTTTT', 'CTTTTT', 'GTTT', 'GTTTTT','A','AG','ATTTTT','ATTT','CT','AGG','GTTTTT', 'AGGG','AT','C','GTT','ATT','ATTTTT','AAAT','AAT','AAAC','AAAAC','AAAAAT','AAAAG','AAC','AAAAT','CTTT','AAAAAC','CCCT','AAAG','AGAT','G','AAAAAG','AAGG','AATT','AATG','CCTT','ATGT','CATT','ATTC','GAAT','ATCT','CCT']
"""

classes = ['T']

size = 101
linear_size = (((size - 4)//2)-4)*30

# PATH TO YOUR SEQUENCES HERE !

path_to_human = "Temp_data/"

#path_to_human_model = "models/human_models/"

#human_classes = np.load("Temp_data/Split_by_class_human/classes.npy")

#print(human_classes)

#classes = format_classes(classes, human_classes)

#print(classes)

on_human = []
spear_corr_human = []
pears_corr_human = []
epochs = []
path_to_human_model_temp = "3layer_cnn_models/"
np.load("configs_CNN/config_A.npy", allow_pickle=True).item()
for cla in classes:

    # Format class names because some classes names countain a "/"
    cla_path = format_path(cla)

    # Update the path to take into account the name of the class
    path_to_human_temp = path_to_human + cla_path + "_"

    #path_to_human_model_temp = path_to_human_model + cla_path + ".pt"


    print("Starting analysis for class : ",cla)

    

    

    # Loading data, refere to 01.Data_processing for more info
    train_human, valid_human, test_human = load_data(path_to_human_temp, hyper_param_batch)
        
    # Making model, refere to 02.Model training for more info
    #model=CnnModel(config["filter1"], config["filter2"],config["filter3"], config["noyau1"], config["noyau2"], config["noyau3"], config["linear1"], config["linear2"])
    if cla == 'T':
        cnn_model = cnn_model_window(81, 97, 75, 8, 9, 3, 442, 149).to(device)
        lr = copy.deepcopy(hyper_param_learning_rate)
        hpb = copy.deepcopy(hyper_param_batch)
    else:
        config = np.load("configs_CNN/config_" + cla + ".npy", allow_pickle=True).item()
        lr = config['lr']
        cnn_model = cnn_model_window()
        hpb = config['batch_size']

    # Loading data, refere to 01.Data_processing for more info
    train_human, valid_human, test_human = load_data(path_to_human_temp, hpb)


    criterion = torch.nn.SmoothL1Loss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr, betas=(0.9, 0.999))
    logs = training_pytorch_model(cnn_model, train_human, valid_human, hpb, hyper_param_epoch,
                                criterion, optimizer, device, keep_track=True)
    data = test_pytorch_model(cnn_model, test_human, device=device)

    # Calculating spearman correlation
    spear_corr_human.append(data[0])
    pears_corr_human.append(data[1])
    epochs.append(logs[4])


    #Saving model to disk
    torch.save(cnn_model.state_dict(), path_to_human_model_temp + cla + ".pt")

print('End training')


"""
# Writing results
write_results(classes, on_human)
"""
print("--- end exp ---")
