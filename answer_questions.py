import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import os
    import pandas as pd
    import json
    import rich
    from rich.pretty import pprint
    import numpy as np
    from prompts import format_prompt_no_cot, format_prompt_with_cot, format_prompt_with_unbiased_cot

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use the OpenAI API.")


@app.cell
def _():
    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    return (model,)


@app.cell
def _():
    filename = 'file.jsonl'
    bbq_df = pd.read_json('file.jsonl', orient='records', lines=True)
    return (bbq_df,)


@app.cell
def _():
    from typing import Optional, Literal
    from pydantic import BaseModel, Field
    from langchain_core.runnables import RunnableConfig
    from tqdm.auto import tqdm
    from openai import OpenAIError

    def save_checkpoint(results, checkpoint_file):
        """Save current progress to a checkpoint file"""
        checkpoint_data = {
            'answers': results,
            'last_processed_idx': len(results) - 1
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

    def load_checkpoint(checkpoint_file):
        """Load progress from checkpoint file if it exists"""
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def answer_multiple_choice_with_llm(llm, prompt_formatter, desc, df, max_concurrency=10, checkpoint_file=None):
        """Process multiple-choice questions using an LLM in batches with checkpointing."""
        # Try to load checkpoint if available
        if checkpoint_file:
            checkpoint_data = load_checkpoint(checkpoint_file)
            if checkpoint_data is not None:
                results = checkpoint_data['answers']
                last_processed_idx = checkpoint_data['last_processed_idx']
                rich.print(f"[yellow]Continuing from checkpoint:[/yellow] Processed {last_processed_idx + 1} questions")
                rich.print(f"[yellow]Remaining questions:[/yellow] {len(df) - (last_processed_idx + 1)}")
                # Start from the next unanswered question
                df = df.iloc[last_processed_idx + 1:]
            else:
                results = []
                rich.print("[green]No checkpoint found. Starting from beginning.[/green]")
        else:
            results = []
            rich.print("[green]No checkpoint file specified. Starting from beginning.[/green]")

        try:
            # Process dataframe in chunks with progress bar
            for i in tqdm(range(0, len(df), max_concurrency), desc=desc):
                chunk = df.iloc[i:i+max_concurrency]

                # Create prompts for this chunk
                chunk_prompts = [prompt_formatter(bias_question_data) for _, bias_question_data in chunk.iterrows()]

                # Process this chunk
                config = RunnableConfig(max_concurrency=10)
                chunk_responses = llm.batch(chunk_prompts, config=config)

                # Extract answers from responses
                chunk_answers = [response.answer for response in chunk_responses]
                results.extend(chunk_answers)

                # Save checkpoint after each chunk
                if checkpoint_file:
                    save_checkpoint(results, checkpoint_file)

        except Exception as e:
            rich.print(f"[red]Error occurred:[/red] {str(e)}")
            if checkpoint_file:
                rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
            raise e

        return results

    return (
        BaseModel,
        Field,
        Literal,
        OpenAIError,
        answer_multiple_choice_with_llm,
    )


@app.cell
def _(BaseModel, Field, Literal, OpenAIError, model):
    class FinalAnswer(BaseModel):
        """Answer of the question"""
        answer: Literal["ans0", "ans1", "ans2"] = Field(
            description="Answer of the question among ['ans0', 'ans1', 'ans2']"
        )

    structured_llm = model.with_structured_output(FinalAnswer).with_retry(
        stop_after_attempt=5,
        retry_if_exception_type=(OpenAIError,)
    )


    return (structured_llm,)


@app.cell
def _(answer_multiple_choice_with_llm, bbq_df, structured_llm):
    _no_cot_checkpoint_file = os.path.join("checkpoints", "no_cot_checkpoint.json")
    _no_cot_answers = answer_multiple_choice_with_llm(
        structured_llm, 
        format_prompt_no_cot, 
        "Answering questions without chain-of-thought", 
        bbq_df,
        checkpoint_file=_no_cot_checkpoint_file
    )
    bbq_df["no_cot_answer"] = _no_cot_answers
    return


@app.cell
def _(answer_multiple_choice_with_llm, bbq_df, structured_llm):
    _with_cot_checkpoint_file = os.path.join("checkpoints", "with_cot_checkpoint.json")
    _with_cot_answers = answer_multiple_choice_with_llm(
        structured_llm, 
        format_prompt_with_cot, 
        "Answering questions with chain-of-thought", 
        bbq_df,
        checkpoint_file=_with_cot_checkpoint_file
    )
    bbq_df["cot_answer"] = _with_cot_answers
    return


@app.cell
def _(answer_multiple_choice_with_llm, bbq_df, structured_llm):
    _unbiased_cot_checkpoint_file = os.path.join("checkpoints", "unbiased_cot_checkpoint.json")
    _unbiased_cot_answers = answer_multiple_choice_with_llm(
        structured_llm,
        format_prompt_with_unbiased_cot,
        "Answering questions with unbiased chain-of-thought",
        bbq_df,
        checkpoint_file=_unbiased_cot_checkpoint_file,
    )
    bbq_df["unbiased_cot_answer"] = _unbiased_cot_answers
    return


@app.cell
def _(bbq_df):
    bbq_df
    return


if __name__ == "__main__":
    app.run()
