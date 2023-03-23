#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# In[4]:


import transformers
from torch.utils import data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

# data processing
import re, string
import emoji
import nltk

# dataset
from sklearn.model_selection import train_test_split
import datasets
from datasets import Dataset , Sequence , Value , Features , ClassLabel , DatasetDict

# preprocessing
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize

import re, string

from tqdm import tqdm
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


models = ["distilbert-base-uncased", "bert-base-uncased", "bert-base-cased"]
modelName = models[2] 


# In[10]:


df_train = pd.read_csv("/Users/frankndu/Desktop/Recommendation_System/Corona_NLP_train.csv",encoding='latin-1')
print(df_train.shape)
print(df_train.columns)


# In[11]:


df_train = df_train[["OriginalTweet", "Sentiment"]]
df_train.info()


# In[12]:


df_train.drop_duplicates(subset='OriginalTweet',inplace=True)
df_train = df_train.rename({'OriginalTweet': 'Reviews'}, axis='columns')
df_train['Sentiment']=df_train['Sentiment'].replace({'Neutral':2, 'Positive':3,'Extremely Positive':4, 'Extremely Negative':0,'Negative':1})
df_train['Sentiment']=df_train['Sentiment'].astype(int)

df_train = df_train.reset_index(drop=True)
df_train.head()


# In[13]:


# df.info()
df_train.isnull().sum()


# In[14]:


df_train["Sentiment"].value_counts()


# In[15]:


# word tokenizer
df_train['Reviews_len_by_words'] = df_train['Reviews'].apply(lambda t: len(t.split()))
min_len_word, max_len_word = df_train['Reviews_len_by_words'].min(), df_train['Reviews_len_by_words'].max()
print(min_len_word, max_len_word)


# In[16]:


sns.histplot(df_train['Reviews_len_by_words'])


# In[17]:


df_train.describe()


# In[18]:


class_names = ['Extremely Negative', 'Negative', 'Neutral', 'Positive', 'Extremely Positive']
ax = sns.countplot(x='Sentiment', data=df_train)
ax.set_xticklabels(class_names)


# In[20]:


df_test = pd.read_csv("/Users/frankndu/Desktop/Recommendation_System/Corona_NLP_test.csv",encoding='latin')
df_test = df_test.drop(labels=['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)
df_test.drop_duplicates(subset='OriginalTweet',inplace=True)

df_test = df_test.rename({'OriginalTweet': 'Reviews'}, axis="columns")

df_test['Sentiment']=df_test['Sentiment'].replace({'Neutral':2, 'Positive':3,'Extremely Positive':4, 'Extremely Negative':0,'Negative':1})
df_test['Sentiment']=df_test['Sentiment'].astype(int)
df_test = df_test.reset_index(drop=True)

df_test.head()


# In[21]:


# suffle series
# df_train = df_train.sample(frac=1, random_state=RANDOM_SEED)

print(df_train.shape , df_test.shape)
df_train, df_val = train_test_split(df_train, test_size=0.09, random_state=RANDOM_SEED, stratify=df_train['Sentiment'])
print(df_train.shape , df_val.shape, df_test.shape)


# In[22]:


def createDataset(df, textCol, labelCol):
  dataset_dict = {
    'text' : df[textCol],
    'labels' : df[labelCol],
  }
  sent_tags = ClassLabel(num_classes=5 , names=['Extremely Negative', 'Negative','Neutral','Positive', 'Extremely Positive'])

  return Dataset.from_dict(
    mapping = dataset_dict,
    features = Features({'text' : Value(dtype='string') , 'labels' :sent_tags})
  )


# In[23]:


dataset_train = createDataset(df_train,"Reviews","Sentiment")
dataset_val = createDataset(df_val,"Reviews","Sentiment")
dataset_test = createDataset(df_test,"Reviews","Sentiment")

dataset_sentAnalysis = DatasetDict()
dataset_sentAnalysis["train"] = dataset_train
dataset_sentAnalysis["val"] = dataset_val
dataset_sentAnalysis["test"] = dataset_test

dataset_sentAnalysis


# In[24]:


def convert_to_lower(text):
    return text.lower()

def remove_emojis(text):
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r"<.*?>","",text)

    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        # u"\u200c"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)
    
    return wierd_pattern.sub(r'', text)

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

def preprocessText(text):
  return remove_extra_white_spaces(remove_stopwords(remove_punctuation(remove_numbers(remove_emojis(convert_to_lower(text))))))

def preprocessBatch(batch):
  new_list = []
  for i in batch["text"]:
    new_list.append(remove_extra_white_spaces(remove_stopwords(remove_punctuation(remove_numbers(remove_emojis(convert_to_lower(i)))))))
  batch["text"] = new_list
  return batch


# In[25]:


dataset_sentAnalysis_preprocessed = dataset_sentAnalysis.map(preprocessBatch, batched=True, batch_size=32)


# In[26]:


dataset_sentAnalysis["train"][10]


# In[27]:


dataset_sentAnalysis_preprocessed["train"][10]


# In[28]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(modelName)


# In[29]:


max_len = 128
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=max_len)


# In[30]:


sample_text = "What is  going on @resturant.:( It makes   Me Feel Upset.😞"
tokens = tokenizer.tokenize(sample_text)
print(len(tokens), tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(len(token_ids), token_ids)

token_dictionary = tokenizer(sample_text)
print(token_dictionary)
print(len(token_dictionary.input_ids)) # automatically added cls, sep

tokens = tokenizer.convert_ids_to_tokens(token_dictionary['input_ids'])
print(tokens)

token_encode = tokenizer.encode(sample_text) #convert to tokens ids but with cls+sep
print(token_encode)


# In[31]:


print(tokenizer)
print(tokenizer.sep_token, tokenizer.sep_token_id)
print(tokenizer.cls_token, tokenizer.cls_token_id)
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.unk_token, tokenizer.unk_token_id)


# In[32]:


encoding = tokenizer.encode_plus(
    sample_text,
    max_length=max_len,
    add_special_tokens=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_token_type_ids=False,
    return_tensors="pt"
)
encoding


# In[33]:


dataset_sentAnalysis_encoded = dataset_sentAnalysis_preprocessed.map(tokenize, batched=True, batch_size=32)


# In[34]:


dataset_sentAnalysis_encoded


# In[35]:


from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn

class BertForClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Load model body > return all og the HS
        self.bert = BertModel(config)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, 
                labels=None, **kwargs):
        # Use model body to get encoder representations
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, **kwargs)

        # Apply classifier to encoder representation > [cls]
        sequence_output = self.dropout(outputs[1])
        logits = self.classifier(sequence_output)

        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Return model output object
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# In[36]:


from transformers import AutoConfig

id2label = {
    0: 'Extremely Negative',
    1: 'Negative',
    2: 'Neutral',
    3: 'Positive',
    4: 'Extremely Positive'
}

label2id = { v:k for (k,v) in id2label.items()}

bert_config = AutoConfig.from_pretrained(modelName, 
                                         num_labels=5,
                                         id2label=id2label, label2id=label2id)


# In[37]:


bert_model = (BertForClassification
              .from_pretrained(modelName, config=bert_config)
              .to(device))


# In[39]:


from transformers import Trainer, TrainingArguments
import wandb
wandb.login()

wandb.init(project="bert-for-english-classification")


batch_size = 16
logging_steps = len(dataset_sentAnalysis_encoded["train"]) // batch_size
model_name = f"{modelName}-finetuned-sentimentAnalysis-bert"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=1,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch", 
                                  save_steps=1e6,
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False, 
                                  log_level="error",
                                  report_to="wandb",
                                  run_name="bert-sent-analysis")


# In[40]:


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# In[43]:


from torch.optim import AdamW
from transformers import get_scheduler

optimizer = AdamW(bert_model.parameters(), lr=2e-5)

num_epochs = 1
num_training_steps = num_epochs * logging_steps
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

trainer_preprocessed_lr = Trainer(model=bert_model, args=training_args,
                                  compute_metrics=compute_metrics,
                                  train_dataset=dataset_sentAnalysis_encoded["train"],
                                  eval_dataset=dataset_sentAnalysis_encoded["val"],
                                  tokenizer=tokenizer,
                                  optimizers=(optimizer,lr_scheduler))

trainer_preprocessed_lr.train()


# In[ ]:


model = bert_model
model.eval()
preds_output = trainer_preprocessed_lr.predict(dataset_sentAnalysis_encoded["test"])
pd.DataFrame(list(preds_output.metrics.items())).T


# In[ ]:


PATH = "./bert-classification-classification-head"
torch.save(bert_model.state_dict(), PATH)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


# In[ ]:


y_preds = np.argmax(preds_output.predictions, axis=1)
y_test = np.array(dataset_sentAnalysis_encoded["test"]["labels"])
labels = dataset_sentAnalysis_encoded["train"].features["labels"].names
plot_confusion_matrix(y_preds, y_test, labels)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report

class_names = ['Extremely Negative', 'Negative','Neutral', 'Positive', 'Extremely Positive']
print(classification_report(y_test, y_preds, target_names=class_names))


# In[ ]:


import numpy as np
import torch.nn.functional as F

def predict_text(model,text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    out = F.softmax(outputs.logits,dim=1)
    y_out = np.argmax(out.cpu(),axis=1)
    return out, y_out


# In[ ]:


sample_text = "I know there are certain cases where people can be harsh but i stay depressed because of it"
out, y_out = predict_text(model,sample_text)
out, id2label[y_out.item()]


# In[ ]:


pd.set_option('display.max_colwidth', -1)
pd.DataFrame({
    "Text": sample_text,
    "Sentiment": class_names[y_out.item()]
},index=[0]).T


# In[ ]:





# In[ ]:




