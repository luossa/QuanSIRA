import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import json
from torch.nn.utils.rnn import pad_sequence
import glob
import os
import pandas as pd
import torch.nn.functional as F
import ast
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from peft import LoraConfig, get_peft_model
import logging
import argparse
print(torch.version.cuda)

file_path = '/gemini/code/5-day_Indomain.py'
file_name_with_extension = os.path.basename(file_path)
# 使用os.path.splitext分离文件名和扩展名
file_name, file_extension = os.path.splitext(file_name_with_extension)
print(file_name)

parts = file_name.split("_")
first_part = parts[0]  
second_part = parts[1]
train_csv_file = '/gemini/data-1/'+ second_part + '/' + first_part +'/combined_train.csv'
test_csv_file = '/gemini/data-1/'+ second_part + '/' + first_part +'/combined_test.csv'
val_csv_file = '/gemini/data-1/'+ second_part + '/' + first_part +'/combined_val.csv'


train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)
val_df = pd.read_csv(val_csv_file)



parser = argparse.ArgumentParser(description='Hyperparameter Settings')

parser.add_argument('--batch_size', type=int, default=2, help='Size of each training batch')
parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='Number of lr')
parser.add_argument('--max_num_sentences', type=int, default=30, help='Number of max_num_sentences')
parser.add_argument('--hidden_size', type=int, default=1024, help='Number of max_num_sentences')

args, _ = parser.parse_known_args()

print(len(train_df),len(test_df),len(val_df))
train_checkpoint_path = '/gemini/code/checkpoint/'+ file_name + '/'+ 'training.log'
validate_checkpoint_path = '/gemini/code/checkpoint/'+ file_name + '/'+ 'validate.log'
test_checkpoint_path = '/gemini/code/checkpoint/'+ file_name + '/'+ 'test.log'


# 配置训练日志记录器
train_logger = logging.getLogger("train_logger")
train_handler = logging.FileHandler(train_checkpoint_path)
train_formatter = logging.Formatter('%(asctime)s - %(message)s')
train_handler.setFormatter(train_formatter)
train_logger.addHandler(train_handler)
train_logger.setLevel(logging.INFO)

# 配置验证日志记录器
validate_logger = logging.getLogger("validate_logger")
validate_handler = logging.FileHandler(validate_checkpoint_path)
validate_formatter = logging.Formatter('%(asctime)s - %(message)s')
validate_handler.setFormatter(validate_formatter)
validate_logger.addHandler(validate_handler)
validate_logger.setLevel(logging.INFO)

# 配置测试日志记录器
test_logger = logging.getLogger("test_logger")
test_handler = logging.FileHandler(test_checkpoint_path)
test_formatter = logging.Formatter('%(asctime)s - %(message)s')
test_handler.setFormatter(test_formatter)
test_logger.addHandler(test_handler)
test_logger.setLevel(logging.INFO)

class TextDataset(Dataset):
    def __init__(self, max_num_sentences,data,tokenizer):
        # encodings是包含多个编码结果（每个结果包括input_ids和attention_mask）的列表
        self.max_num_sentences = max_num_sentences
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        csv_row = self.data.iloc[idx]
        sentence = ('[Open]'+str(csv_row['combina-Open']).replace('[', '').replace(']', '')+
        '[High]'+str(csv_row['combina-High']).replace('[', '').replace(']', '')+
        '[Low]'+str(csv_row['combina-Low']).replace('[', '').replace(']', '')+
        '[Close]'+str(csv_row['combina-Close']).replace('[', '').replace(']', '')+
        '[Adj Close]'+str(csv_row['combina-Adj_Close']).replace('[', '').replace(']', '')+
        '[Volume]'+str(csv_row['combina-Volume']).replace('[', '').replace(']', ''))
        csv_row_text_list = eval(csv_row['list'])
        
        if len(csv_row_text_list) >= self.max_num_sentences:
            csv_row_text_list = csv_row_text_list[:self.max_num_sentences]
            csv_row_text_encoded = tokenizer(csv_row_text_list, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        else:
            csv_row_text_encoded = tokenizer(csv_row_text_list, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            num_to_pad = self.max_num_sentences - len(csv_row_text_list)
            padded_tensors = [torch.zeros_like(csv_row_text_encoded['input_ids'][0]) for i in range(num_to_pad)]
            padded_tensors = torch.stack(padded_tensors)
            csv_row_text_encoded['input_ids'] = torch.cat((csv_row_text_encoded['input_ids'],padded_tensors),dim=0)#将列表转换成tensor的形式
            csv_row_text_encoded['attention_mask'] = torch.cat((csv_row_text_encoded['attention_mask'],padded_tensors),dim=0)
            
        sentence_encoded = tokenizer(sentence, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            
            
            # 创建填充句子
          
        # 从每个编码字典中提取input_ids和attention_mask，并创建一个新的字典
        item = {
            'input_ids': csv_row_text_encoded['input_ids'],
            'attention_mask': csv_row_text_encoded['attention_mask'],
            'num_input_ids':sentence_encoded['input_ids'],
            'num_attention_mask':sentence_encoded['attention_mask'],
            'risk-label':torch.tensor(csv_row['label-risk'])
        }
        # 添加labels到字典中
        return item

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#模型
class BertWithText(torch.nn.Module):
    def __init__(self,model_text,model_num,max_num_sentences,batch_size,Classifier,input_size,hidden_size,num_classes):
        super(BertWithText, self).__init__()
        #初始化分类器的参数
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
        #初始化一个处理文本数据的模型一个处理数值数据的模型
        self.model_text = model_text
        self.model_num = model_num
        
        self.max_num_sentences = max_num_sentences
        self.Classifier = Classifier(input_size,hidden_size,num_classes)
        
        self.batch_size = batch_size
    def forward(self, batch):
        #提取文本数据的input_ids、attention_mask、token_type_ids
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        #提取数值数据的num_input_ids、num_attention_mask、num_token_type_ids
        num_input_ids = torch.squeeze(batch['num_input_ids'])
        num_attention_mask = torch.squeeze(batch['num_attention_mask'])
        dimen_0 = input_ids.size(0)
        dimen_1 = input_ids.size(1)

        reshaped_input_ids = input_ids.view(dimen_0*dimen_1, 128)
        reshaped_attention_mask = attention_mask.view(dimen_0*dimen_1, 128)
        
        #分别将文本数据和数值数据丢入Bert
        outputs_text = self.model_text(input_ids=reshaped_input_ids, attention_mask=reshaped_attention_mask)
        outputs_num = self.model_num(input_ids = num_input_ids,attention_mask = num_attention_mask)
        
        pooled_output_text = outputs_text.pooler_output
        pooled_output_num = outputs_num.pooler_output
    
        #将文本数据转换成原始的格式
        reshaped_pooled_output = pooled_output_text.view(dimen_0,dimen_1,1024)
        #对文本数据进行pool的操作，分别取最大值、最小值、平均值
        max_value, _ = torch.max(reshaped_pooled_output, dim=1)
        min_value, _ = torch.min(reshaped_pooled_output, dim=1)
        mean_value = torch.mean(reshaped_pooled_output, dim=1)
        
        connect_tensor = torch.cat((max_value, min_value,mean_value,pooled_output_num), dim=1)
        
        logits = self.Classifier(connect_tensor)

        return logits


def train(model, train_loader, optimizer, scheduler, loss_fn,epochs,device):
    model.train()
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []
    for batch in tqdm(train_loader,desc="Training-batch"):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(batch)
        outputs = outputs.float()
            
        labels = batch['risk-label'].long()
            
        labels = labels.to(device)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
            
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(batch['risk-label'].tolist())
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    accuracy = correct_predictions / total_predictions
    Loss = epoch_loss / len(train_loader) 
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return Loss, accuracy, precision, recall, f1
        
def validate(model,val_loader,device):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"val-process"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch).to(device)
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.tolist())
            all_labels.extend(batch['risk-label'].tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy,precision,recall,f1

def test(model,test_loader,device):
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"val-process"):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch)
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.tolist())
            all_labels.extend(batch['risk-label'].tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy,precision,recall,f1




tokenizer = AutoTokenizer.from_pretrained("/gemini/pretrain/BERTweet-large-case")
model_text = AutoModel.from_pretrained("/gemini/pretrain/BERTweet-large-case")
model_num  = AutoModel.from_pretrained("/gemini/pretrain/BERTweet-large-case")

max_num_sentences = args.max_num_sentences
batch_size = args.batch_size
hidden_size = args.hidden_size
epochs = args.epochs
lr = args.lr

input_size = 4096
num_classes = 3

print('max_num_sentences:',max_num_sentences,'batch_size:',batch_size,'hidden_size:',hidden_size,'epochs:',epochs,'lr:',lr)


train_dataset = TextDataset(max_num_sentences,train_df,tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

val_dataset = TextDataset(max_num_sentences,val_df,tokenizer)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

test_dataset = TextDataset(max_num_sentences,test_df,tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = BertWithText(model_text,model_num,max_num_sentences,batch_size,Classifier,input_size,hidden_size,num_classes).to(device)

config = LoraConfig(
    r=16,  # low rank
    lora_alpha=32,  
    lora_dropout=0.05,
    bias="none",
    #task_type="TEXT_CLASSIFICATION",
    target_modules = ["query", "value"]
)


model = get_peft_model(model,config)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.size()}")

total_steps = len(train_loader) * epochs
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.06, num_training_steps=total_steps)
loss_fn = CrossEntropyLoss()

a = 0
#train
for epoch in tqdm(range(epochs), desc="Training-epoch"):
#train
    train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(model,train_loader, optimizer, scheduler, loss_fn,epochs,device)
    print(f'train:Epoch {epoch+1}, train_loss: {train_loss},train_accuracy: {train_accuracy},train_precision: {train_precision},train_recall: {train_recall},train_f1: {train_f1}')
    train_logger.info(f'Epoch {epoch + 1}: train_loss = {train_loss}, train_accuracy = {train_accuracy},train_precision= {train_precision}, train_recall= {train_recall}, train_f1= {train_f1}')
#val    
    val_accuracy, val_precision, val_recall, val_f1 = validate(model, val_loader,device)
    print(f'validate: val_accuracy: {val_accuracy},val_precision: {val_precision},val_recall: {val_recall},val_f1: {val_f1}')
    validate_logger.info(f'validate: val_accuracy = {val_accuracy}, val_precision= {val_precision}, val_recall= {val_recall}, val_f1: {val_f1}')

#test
    test_accuracy, test_precision, test_recall, test_f1 = test(model,test_loader,device)
    print(f'test: test_accuracy: {test_accuracy},test_precision: {test_precision},test_recall: {test_recall},test_f1: {test_f1}')
    test_logger.info(f'test: test_accuracy = {test_accuracy}, test_precision= {test_precision}, test_recall= {test_recall}, test_f1= {test_f1}')
    checkpoint = {'parameter': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
              'epoch': epoch}
    torch.save(checkpoint, '/gemini/code/checkpoint/'+ file_name + '/'+'ckpt_{}.pth'.format(epoch+1))