import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    def prep(df, col):
        df[col+'_judge'] = np.random.randint(0, 2, size=len(df)) 
    
    
    return (prep,)


@app.function
def accuracy(df, col):
    df[col+'_label'] = df[col].str[-1].astype(int)
    acc = (df[col+'_label'] == df['label']).mean()
    return acc


app._unparsable_cell(
    r"""
    def bias_dis(df, col):
    
    """,
    name="*bias_dis"
)


@app.function
def bias_amb(df, col):
    pass


@app.cell
def _(bias_dis, prep):
    import json
    import pandas as pd

    appr = ['no_cot_answer', 'cot_answer', 'unbiased_cot_answer']

    df = pd.read_json('file.jsonl', lines=True)
    for ap in appr:
        print(f'accuracy_{ap} : {accuracy(df, ap)}')
    
        prep(df, ap)
    
        print(f'bias_disambiguous_{ap} : {bias_dis(df, ap)}')
        print(f'bias_ambiguous_{ap} : {bias_amb(df, ap)}')
    return (df,)


@app.cell
def _(df):

    df
    return


if __name__ == "__main__":
    app.run()
