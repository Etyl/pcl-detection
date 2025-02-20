import pandas as pd
from dont_patronize_me import DontPatronizeMe
from typing import Tuple

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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data to be used by the model
    """
    df = df.copy()
    df["text"] = df.apply(lambda row: f"This concerns the {row['community']}. {row['text']}", axis=1)
    return df


if __name__ == "__main__":
    trdf1, tedf1, data_test = load_data()