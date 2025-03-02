import pandas as pd
from dont_patronize_me import DontPatronizeMe
from typing import Tuple
import torch
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
            device = device,
            action = "substitute",
            top_k = 20
        )
        augmented_texts = aug.augment(text)
        
    return augmented_texts[0]


def augment_data_df(df: pd.DataFrame):
    augmented_train_df = df.copy()
    augmented_rows = []

    for index, row in augmented_train_df.iterrows():
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
    augmented_rows_df["label"] = 1

    # Append the new DataFrame to the original DataFrame
    augmented_train_df = pd.concat([augmented_train_df, augmented_rows_df])

    return augmented_train_df


def preprocess_data(df: pd.DataFrame, clean_data=False, augment_data=False, add_country=False, add_community=False) -> pd.DataFrame:
    """
    Preprocess the data to be used by the model
    """
    df = df.copy()
    if clean_data:
        df["text"] = df["text"].apply(lambda x: clean_text(x, 
            whitespacing=False, 
            standard_tokens=True,
            punctuation=True, 
            stop_words=False, 
            lower=False
        ))
    if augment_data:
        df = augment_data_df(df)
    if add_country:
        df["text"] = df.apply(lambda row: f"{row["country"]} , {row['text']}", axis=1)
    if add_community:
        df["text"] = df.apply(lambda row: f"{row['community']} , {row['text']}", axis=1)
    
    return df

