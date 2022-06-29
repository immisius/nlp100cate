
import torch
from torch import nn
from transformers import BertTokenizer
import re
import streamlit as st
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()
MAX_LEN=64
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def text_preprocessing(text):
    text=re.sub(r'[^a-zA-Z0-9]',"",text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def preprocessing_for_bert(data):

    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',       # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks



from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()

        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 300, 4
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

    def forward(self, input_ids, attention_mask):


        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits

model=BertClassifier()
path='model.pth'
model.load_state_dict(torch.load(path))

import streamlit as st

cate=['科学技術','経営','娯楽','健康']
st.markdown("# 文章ラベル付け")
st.markdown("科学技術、経営、娯楽、健康について分類します。")
st.text_input("テキストを入力してください(in english)", key='input',value='Hello, World!')
input=st.session_state['input']
if input:
    id,mask=preprocessing_for_bert([input])
    res=nn.Softmax(dim=1)(model(id,mask))
    best=torch.argmax(res, dim=1).flatten()[0].item()
    for i in range(4):
        v='{:.4f}'.format(res[0][i].item())
        s=cate[i]+':'+v
        st.markdown("### "+s)
    predict=cate[best]
    p='Predict:'+predict
    st.markdown("## "+p)