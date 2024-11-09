from llama_cpp import Llama
import pathlib
import pandas as pd
import math

#Defining models
MODEL_Q8_0 = Llama(
    model_path="Llama-3.2-1B-Instruct-Q8_0.gguf",
    n_ctx=128, n_gpu_layers=128, logits_all=True
)

#Defining function for getting a response
def query_with_logprobs(model, question):
    prompt = f"Q: {question} A:"
    output = model(prompt=prompt, max_tokens=1, temperature=10000, logprobs=True)
    response = output["choices"][0]
    logprobs = response["logprobs"]["top_logprobs"][0]  # Get logprobs for first token

    # Extract logprobs for TRUE and FALSE
    logprob_true = math.exp(logprobs.get(" TRUE", float("-inf")))
    score = (logprob_true) / (logprob_true + math.exp(logprobs.get(" FALSE", float("-inf"))))
    return logprob_true, score

#Resolve Wikidata IDs to English labels
df_verify = pd.read_excel("verify_with_llama_add_refined.xlsx")

#Defining function to check a relation
def check_relation_with_llama(subject, object):
    logprob_true, score = query_with_logprobs(
        MODEL_Q8_0, f"Are the concepts '{subject}' and '{object}' directly related? Answer with one word: TRUE or FALSE."
    )
    return logprob_true, score

df_verify["llama_logprobs"] = ""  # Initialize a new column for LLAMA validity
df_verify["llama_true"] = ""

for index, row in df_verify.iterrows():
    subject = row["subj"]
    object = row["obj"]
    logprob_true, score = check_relation_with_llama(subject, object)

    # Calculate normalized score as a confidence metric based on logprobs
    df_verify.at[index, "llama_logprobs"] = score
    df_verify.at[index, "llama_true"] = logprob_true
    print(index)

df_verify.to_excel("verify_with_llama_add_logprobs.xlsx", index=False)
