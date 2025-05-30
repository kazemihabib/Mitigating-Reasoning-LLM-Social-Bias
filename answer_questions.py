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
    from prompts import (
        format_prompt_no_cot,
        format_prompt_with_cot,
        format_prompt_with_unbiased_cot,
    )
    from typing import Optional, Literal, List, Dict, Any, Callable, Union
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel, Field
    from tqdm.auto import tqdm
    from openai import OpenAIError
    from utils import save_checkpoint, load_checkpoint


@app.cell
def _():
    from langchain.chat_models import init_chat_model

    from langchain_openai import ChatOpenAI

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable is not set. Please set it to use the OpenAI API."
        )

    model = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0)
    return (model,)


@app.cell
def _(model):
    class FinalAnswer(BaseModel):
        """Structured output model for multiple-choice question answers"""

        answer: Literal["ans0", "ans1", "ans2"] = Field(
            description="The selected answer from the available choices. Must be exactly one of: ans0, ans1, or ans2.",
        )

    structured_llm = model.with_structured_output(FinalAnswer).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 9},
        retry_if_exception_type=(OpenAIError, ValueError),
    )

    return (structured_llm,)


@app.function
def answer_multiple_choice_with_llm(
    llm: BaseChatModel,
    prompt_formatter: Callable[[pd.Series], List[Any]],
    desc: str,
    df: pd.DataFrame,
    max_concurrency: int = 10,
    checkpoint_file: Optional[str] = None,
) -> List[str]:
    """Process multiple-choice questions using an LLM in batches with checkpointing.

    Args:
        llm: Language model instance
        prompt_formatter: Function to format prompts from question data
        desc: Description for progress bar
        df: DataFrame containing questions
        max_concurrency: Maximum number of concurrent requests
        checkpoint_file: Optional path to checkpoint file

    Returns:
        List of answer strings
    """
    # Try to load checkpoint if available
    if checkpoint_file:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data is not None:
            results = checkpoint_data["answers"]
            last_processed_idx = checkpoint_data["last_processed_idx"]
            rich.print(
                f"[yellow]Continuing from checkpoint:[/yellow] Processed {last_processed_idx + 1} questions"
            )
            rich.print(
                f"[yellow]Remaining questions:[/yellow] {len(df) - (last_processed_idx + 1)}"
            )
            # Start from the next unanswered question
            df = df.iloc[last_processed_idx + 1 :]
        else:
            results = []
            rich.print("[green]No checkpoint found. Starting from beginning.[/green]")
    else:
        results = []
        rich.print(
            "[green]No checkpoint file specified. Starting from beginning.[/green]"
        )

    try:
        # Process dataframe in chunks with progress bar
        for i in tqdm(range(0, len(df), max_concurrency), desc=desc):
            chunk = df.iloc[i : i + max_concurrency]

            # Create prompts for this chunk
            chunk_prompts = [
                prompt_formatter(bias_question_data)
                for _, bias_question_data in chunk.iterrows()
            ]

            # Process this chunk
            config = RunnableConfig(max_concurrency=max_concurrency)
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
            rich.print(
                f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}"
            )
        raise e

    return results


@app.function
def create_checkpoint_file(checkpoint_path: str) -> str:
    """Create a checkpoint file for saving progress."""
    os.makedirs("checkpoints", exist_ok=True)
    return os.path.join("checkpoints", checkpoint_path)


@app.cell
def _():
    def answer_no_cot(
        checkpoint_name: str,
        df: pd.DataFrame,
        llm: BaseChatModel,
        max_concurrency: int = 50,
    ) -> list:
        no_cot_checkpoint_file = create_checkpoint_file(
            f"{checkpoint_name}_no_cot_checkpoint.json"
        )
        return answer_multiple_choice_with_llm(
            llm,
            format_prompt_no_cot,
            "Answering questions WITHOUT chain-of-thought",
            df,
            max_concurrency=max_concurrency,
            checkpoint_file=no_cot_checkpoint_file,
        )

    def answer_with_cot(
        checkpoint_name: str,
        df: pd.DataFrame,
        llm: BaseChatModel,
        max_concurrency: int = 50,
    ) -> list:
        with_cot_checkpoint_file = create_checkpoint_file(
            f"{checkpoint_name}_with_cot_checkpoint.json"
        )
        return answer_multiple_choice_with_llm(
            llm,
            format_prompt_with_cot,
            "Answering questions WITH chain-of-thought",
            df,
            max_concurrency=max_concurrency,
            checkpoint_file=with_cot_checkpoint_file,
        )

    def answer_unbiased_cot(
        checkpoint_name: str,
        df: pd.DataFrame,
        llm: BaseChatModel,
        max_concurrency: int = 50,
    ) -> list:
        unbiased_cot_checkpoint_file = create_checkpoint_file(
            f"{checkpoint_name}_unbiased_cot_checkpoint.json"
        )
        return answer_multiple_choice_with_llm(
            llm,
            format_prompt_with_unbiased_cot,
            "Answering questions WITH unbiased chain-of-thought",
            df,
            max_concurrency=max_concurrency,
            checkpoint_file=unbiased_cot_checkpoint_file,
        )

    return answer_no_cot, answer_unbiased_cot, answer_with_cot


@app.cell
def _(answer_no_cot, answer_unbiased_cot, answer_with_cot):
    def get_all_answers(output_dir: str, dataset_path: str, llm: BaseChatModel):
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        checkpoint_name = dataset_path.replace(os.path.sep, "_")
        df = pd.read_json(dataset_path, orient="records", lines=True)

        no_cot_answers_raw = answer_no_cot(checkpoint_name, df, llm, max_concurrency=50)
        df["no_cot_answer"] = [
            int(ans.removeprefix("ans")) for ans in no_cot_answers_raw
        ]

        cot_answers_raw = answer_with_cot(checkpoint_name, df, llm, max_concurrency=50)
        df["cot_answer"] = [int(ans.removeprefix("ans")) for ans in cot_answers_raw]

        unbiased_cot_answers_raw = answer_unbiased_cot(
            checkpoint_name, df, llm, max_concurrency=50
        )
        df["unbiased_cot_answer"] = [
            int(ans.removeprefix("ans")) for ans in unbiased_cot_answers_raw
        ]

        output_path = os.path.join(
            output_dir, f"{dataset_name}-answers-nocot-cot-unbiasedcot.jsonl"
        )
        df.to_json(output_path, orient="records", lines=True)
        rich.print(f"[green]Processed and saved:[/green] {output_path}")

    return (get_all_answers,)


@app.cell
def _answer_multiple_choice_with_llm(get_all_answers, structured_llm):
    import glob

    DATASETS_DIR = "post_judge_datasets/BBQ"
    OUTPUT_DIR = "answers/BBQ"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_files = glob.glob(os.path.join(DATASETS_DIR, "*.jsonl"))
    results = {}
    # just_answer_this = "datasets/BBQ/Age_judge_agg.jsonl"
    just_answer_this = None

    if just_answer_this is None:
        for dataset_path in dataset_files:
            get_all_answers(OUTPUT_DIR, dataset_path, structured_llm)
    else:
        get_all_answers(OUTPUT_DIR, just_answer_this, structured_llm)

    return


if __name__ == "__main__":
    app.run()
