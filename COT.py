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
    return ChatDeepSeek, ChatPromptTemplate, json, os, pd, pprint


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
def _(ChatPromptTemplate, bbq_df):
    prompt_template = ChatPromptTemplate.from_template("""
    Context: {context}
    Question: {question}
    Answer choices:
    - ans0: {ans0}
    - ans1: {ans1}
    - ans2: {ans2}
    Based on the context, question and options. Output the final answer from options [ans0, ans1, ans2].
    """)


    example = bbq_df.iloc[0]
    formatted_prompt = prompt_template.format_messages(
        context=example["context"],
        question=example["question"],
        ans0=example["ans0"],
        ans1=example["ans1"],
        ans2=example["ans2"]
    )

    # Display the formatted prompt
    formatted_prompt
    return (formatted_prompt,)


@app.cell
def _(response):
    reasoning_content = response.additional_kwargs['reasoning_content']
    return (reasoning_content,)


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
def _(formatted_prompt, model, pprint):
    # structured_llm = model.with_structured_output(FinalAnswer)
    response = model.invoke(formatted_prompt)

    # Example usage
    pprint(response)
    return (response,)


if __name__ == "__main__":
    app.run()
