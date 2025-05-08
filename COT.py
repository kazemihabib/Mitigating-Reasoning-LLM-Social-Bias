import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import rich
    from rich.pretty import pprint
    import marimo as mo
    import json
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek
    import pandas as pd
    from typing import Optional, Literal
    from pydantic import BaseModel, Field
    from langchain_core.runnables import RunnableConfig
    from tqdm.auto import tqdm
    return ChatDeepSeek, ChatPromptTemplate, RunnableConfig, json, os, pd, tqdm


@app.cell
def _(ChatDeepSeek, os):
    if not os.environ.get("API_KEY_DEEPSEEK"):
        raise ValueError("API_KEY_DEEPSEEK environment variable is not set. Please set it to use the DeepSeek API.")

    model = ChatDeepSeek( model_name = "deepseek/deepseek-r1:free", api_key = os.environ["API_KEY_DEEPSEEK"], api_base=os.environ["API_URL_DEEPSEEK"])
    return (model,)


@app.cell
def _(json, pd):
    # Read the JSONL file
    with open('filtered_output.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
        print(data)

    bbq_df = pd.DataFrame(data)
    bbq_df
    return (bbq_df,)


@app.cell
def _(ChatPromptTemplate):
    prompt_template = ChatPromptTemplate.from_template("""
    Context: {context}
    Question: {question}
    Answer choices:
    - ans0: {ans0}
    - ans1: {ans1}
    - ans2: {ans2}
    Based on the context, question and options. Output the final answer from options [ans0, ans1, ans2].
    """)
    return (prompt_template,)


@app.cell
def _(reasoning_content):
    # Split into steps by newlines and filter out empty lines
    def parse_reasoning_steps(text):
        steps = [line.strip() for line in text.split('\n') if line.strip()]
        return steps if steps else [text]

    steps = parse_reasoning_steps(reasoning_content)

    print("Step-by-Step Reasoning:")
    for step in steps:
        print(step)
    return


@app.cell
def _(RunnableConfig, model, prompt_template, tqdm):
    # Process all rows in the dataframe using chunked processing
    def answer_examples(df, chunk_size=10):
        #structured_llm = model.invoke(formatted_prompt)
        results = []

        # Process dataframe in chunks with progress bar
        for i in tqdm(range(0, len(df), chunk_size), desc="Processing examples"):
            chunk = df.iloc[i:i+chunk_size]
            chunk_prompts = []

            # Create prompts for this chunk
            for _, example in chunk.iterrows():
                formatted_prompt = prompt_template.format_messages(
                    context=example["context"],
                    question=example["question"],
                    ans0=example["ans0"],
                    ans1=example["ans1"],
                    ans2=example["ans2"],
                )
                chunk_prompts.append(formatted_prompt)

            # Process this chunk
            config = RunnableConfig(max_concurrency=10)  # Adjust concurrency as needed
            chunk_responses = model.batch(chunk_prompts, config=config)

            # Extract answers from responses
            for response in chunk_responses:
                try:
                    results.append(response.additional_kwargs['reasoning_content'])
                except Exception as e:
                    print(f"Error processing response: {e}")
                    results.append(None)

        return results
    return (answer_examples,)


@app.cell
def _(answer_examples, bbq_df):
    all_responses = answer_examples(bbq_df)
    return (all_responses,)


@app.cell
def _(all_responses):
    all_responses
    return


@app.cell
def _(all_responses, bbq_df):
    bbq_df['COT'] = all_responses
    return


if __name__ == "__main__":
    app.run()
