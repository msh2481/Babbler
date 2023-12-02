# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn.functional as F
from beartype import beartype as typed
from torch import Tensor as TT
from jaxtyping import Float, Int
from typing import Mapping
from itertools import islice
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from IPython.display import clear_output
from trl import DPOTrainer


%load_ext autoreload
%autoreload 2


# %%
model_name = "roneneldan/TinyStories-1M"
model = AutoModelForCausalLM.from_pretrained(model_name)
model_ref = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
tokenizer.pad_token = tokenizer.eos_token

# %%

train_dataset = Dataset.from_dict({
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
})

eval_dataset = Dataset.from_dict({
    "prompt": [
        "2 + 2 = ",
        "3 + 3 = ",
    ],
    "chosen": [
        "4",
        "6",
    ],
    "rejected": [
        "1",
        "5",
    ]
})

# %%
@typed
def train(batch_size: int, lr: float, beta: float) -> None:
    max_steps = 100

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        num_train_epochs=100,
        remove_unused_columns=False,
        learning_rate=lr,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=10, 
        eval_steps=500,
        output_dir="trainer",
        optim="adamw_torch",
        warmup_steps=150,
        report_to="none",
        save_total_limit=1,
        # bf16=True,
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=128,
        max_target_length=128,
        max_prompt_length=128,
        # generate_during_eval=True,
    )
    # dpo_trainer.add_callback(DVCLiveCallback())
    dpo_trainer.train()

# %%
train(batch_size=1, lr=1e-3, beta=0.1)

# %%
@typed
def show_string_with_weights(s: list[str], w: list[float] | Float[TT, "seq"]) -> None:
    from IPython.display import HTML, display
    from matplotlib.colors import rgb2hex
    from matplotlib import colormaps

    cmap = colormaps["coolwarm"]

    def brighten(rgb):
        return tuple([(x + 1) / 2 for x in rgb])

    colors = [brighten(cmap(alpha)) for alpha in w]
    html_str_colormap = " ".join(
        [
            f'<span style="background-color: {rgb2hex(color)}; padding: 1px; margin: 0px; border-radius: 5px;">{word}</span>'
            for word, color in zip(s, colors)
        ]
    )
    display(HTML(html_str_colormap))


@typed
def explore(sample: Mapping[str, list[int]]) -> None:
    device = "cuda" if t.cuda.is_available() else "cpu"
    model.to(device=device)
    gen_length = 10
    with t.no_grad():
        inputs = tokenizer(sample["prompt"], return_tensors="pt")
        sampled_tokens = (
            model_ref.generate(
                **inputs,
                max_new_tokens=gen_length,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=[[tokenizer.pad_token_id]],
                do_sample=True,
            )[0]
            .detach()
            .cpu()
        )
        result = tokenizer.decode(sampled_tokens)
        print(result)

@typed
def evaluate(n_samples: int) -> None:
    for sample in islice(train_dataset, n_samples):
        explore(sample)

# %%

evaluate(n_samples=10)
# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
name = input("Model name, e.g. brackets-flat_shuffle: ")
model.push_to_hub(name)
tokenizer.push_to_hub(name)
# %%
1 / 0

# %%
import gc

gc.collect()
t