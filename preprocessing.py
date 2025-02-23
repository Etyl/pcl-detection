import pandas as pd
from dont_patronize_me import DontPatronizeMe
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DPMDataset(Dataset):
    """
    Dataset for regular use with Roberta/Deberta
    
    """
    def __init__(self, df, tokenizer, max_len, test_set=False):
        
        self.test_set = test_set
        self.tokenizer = tokenizer
        self.text = df.text

        if not test_set:
            self.label = df.label

        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if not self.test_set:
            return {
                'text': self.text[idx],
                'target': self.label[idx]
            }
        else:
            return {
                'text': self.text[idx],
            }


    def collate_fn(self, batch):
        texts = []
        labels = []


        for b in batch:
            texts.append(b['text'])
            if not self.test_set:
                labels.append(b['target'])

        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
        if not self.test_set:
            encodings['target'] = torch.tensor(labels)

        return encodings


def labels2file(p, outf_path):
    with open(outf_path,'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi])+'\n')


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from the DontPatronizeMe class and returns the training, validation, test dataframes
    """
    dpm = DontPatronizeMe('./data', './data/task4_test.tsv')
    dpm.load_task1()
    trids = pd.read_csv('./data/train_semeval_parids-labels.csv')
    teids = pd.read_csv('./data/dev_semeval_parids-labels.csv')
    trids.par_id = trids.par_id.astype(str)
    teids.par_id = teids.par_id.astype(str)
    data=dpm.train_task1_df

    # Build training data
    rows = [] # will contain par_id, label and text
    for idx in range(len(trids)):
        parid = trids.par_id[idx]
        #print(parid)
        # select row from original dataset to retrieve `text` and binary label
        keyword = data.loc[data.par_id == parid].keyword.values[0]
        text = data.loc[data.par_id == parid].text.values[0]
        label = data.loc[data.par_id == parid].label.values[0]
        country = data.loc[data.par_id == parid].country.values[0]
        rows.append({
            'par_id':parid,
            'community':keyword,
            'country':country,
            'text':text,
            'label':label
        })
    trdf1 = pd.DataFrame(rows)

    # Build test data
    rows = [] # will contain par_id, label and text
    for idx in range(len(teids)):
        parid = teids.par_id[idx]
        #print(parid)
        # select row from original dataset
        keyword = data.loc[data.par_id == parid].keyword.values[0]
        text = data.loc[data.par_id == parid].text.values[0]
        label = data.loc[data.par_id == parid].label.values[0]
        country = data.loc[data.par_id == parid].country.values[0]
        rows.append({
            'par_id':parid,
            'community':keyword,
            'country':country,
            'text':text,
            'label':label
        })
    tedf1 = pd.DataFrame(rows)

    dpm.load_test()
    data_test = dpm.test_set_df
    data_test["community"] = data_test["keyword"]
    data_test.drop(columns=['art_id','keyword'], inplace=True)

    return trdf1, tedf1, data_test


def clean_text(text: str, whitespacing = True, standard_tokens = True, punctuation=True, stop_words = False,lower=False) -> str:

        stop_words = stopwords.words("english")
        
        if stop_words:
            # Stop words --> token
            words = word_tokenize(text)
            words = [word for word in words if word not in stopwords.words("english") and word.isalpha()]
            text = " ".join(words)

        if lower:
             text=text.lower()

        if whitespacing:
            # Whtespacing and other standarisation
            text = text.strip('"')  # removing " at start of sentences
            text = re.sub(r'\s+', ' ', text)
            text = re.sub('<h>', '.', text)

        if standard_tokens:
            # Removing unecessary info
            text = re.sub("(?<![\w])[0-9]+[.,]?[0-9]*(?![\w])", "[NUM]", text)
            text = re.sub("\[NUM\]-\[NUM\]", "[NUM]", text)
            # To delete account numbers 12-5223-231
            text = re.sub("\[NUM\]-\[NUM\]", "[NUM]", text)
            text = re.sub(r"https? : \S+", "[WEBSITE]", text)  # Tokenize links
            text = re.sub("(?<![\w])20[0-5][0-9]-?[0-9]*", "[YEAR]", text)  # Year token
            text = re.sub(r"@[\S]+", "[USERNAME]", text)  # removing referencing on usernames with @
            text = re.sub("(?<![\w])1[0-9]{3}-?[0-9]*", "[YEAR]", text)  # Year token
            #text = re.sub("(?<=\[NUM\])-(?=[a-zA-Z])", " ", text)

        if punctuation:
            text = re.sub(r":\S+", "", text)  # removing smileys with : (like :),:D,:( etc)
            text = re.sub(r"\"+", "", text)  # replacing repetitions of punctations
            text = re.sub(r"(\W)(?=\1)", "", text)  # replacing repetitions of punctations
            
        return text.strip()



def augmenter (text, deletion=False, Synonym=False, Spelling=False, CWE=False):

    augmented_texts=text

    if Spelling:
        aug = naw.SpellingAug()
        augmented_texts = aug.augment(augmented_texts, n=3)
        
    if deletion:
        aug = naw.RandomWordAug(action="delete")
        augmented_texts = aug.augment(augmented_texts)

    if Synonym:
        aug = naw.SynonymAug(aug_src='wordnet')
        augmented_texts = aug.augment(augmented_texts)
    
    if CWE:
        aug = naw.ContextualWordEmbsAug(
            model_path = 'distilbert-base-uncased', 
            action = "substitute",
            device = device,
            top_k = 20
        )
        augmented_texts = aug.augment(text)
        
    return augmented_texts[0]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data to be used by the model
    """
    df = df.copy()
    # df["text"] = df.apply(lambda row: f"This concerns the {row['community']}. {row['text']}", axis=1)
    df["text"] = df["text"].apply(lambda x: clean_text(x, 
        whitespacing=False, 
        standard_tokens=True,
        punctuation=True, 
        stop_words=True, 
        lower=False
    ))
    return df


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new PCL data
    """
    df = df.copy(deep=True)
    augmented_rows = []

    for index, row in df.iterrows():
        if row['label'] == 0:
            continue
        else:
            # Apply the augmenter function to the 'text' column
            augmented_text = augmenter(row['text'], deletion=False, Synonym=True, Spelling=False, CWE=True)
            # Create a new row with the augmented text
            augmented_row = row.copy()
            augmented_row['text'] = augmented_text
            
            # Append the new row to the list
            augmented_rows.append(augmented_row)

    # Convert the list of augmented rows to a DataFrame
    augmented_rows_df = pd.DataFrame(augmented_rows)

    # Append the new DataFrame to the original DataFrame
    augmented_df = pd.concat([df, augmented_rows_df])
    return augmented_df




if __name__ == "__main__":
    trdf1, tedf1, data_test = load_data()