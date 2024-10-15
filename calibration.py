from llama_cpp import Llama
import pathlib
import pandas as pd

MODEL_Q8_0 = Llama(
    model_path="Llama-3.2-1B-Instruct-Q8_0.gguf",
    n_ctx=128, n_gpu_layers=128)

def query(model, question):
    model_name = pathlib.Path(model.model_path).name
    prompt = f"Q: {question} A:"
    output = model(prompt=prompt, max_tokens=1) # if max tokens is zero, depends on n_ctx
    response = output["choices"][0]["text"]
    return response

df_verify = pd.read_excel("rand_rel.xlsx")

def check_relation_with_llama(subject, property, object):
  response = ""
  while (response.upper().strip() != "TRUE") and (response.upper().strip() != "FALSE"):
    response = query(MODEL_Q8_0, f"Is the relation [{subject}, {property}, {object}] accurate? Answer with one word: TRUE or FALSE.")
    generated_text = response
  return response.upper().strip()

for index, row in df_verify.iterrows():
  subject = row["subj"]
  property = row["p"]
  object = row["obj"]
  is_valid = []
  for i in range(30):
    is_valid.append(check_relation_with_llama(subject, property, object))
    print(is_valid.count("TRUE")/len(is_valid))
    df_verify.at[index, "response_"+str(i)] = is_valid.count("TRUE")/len(is_valid)
df_verify.to_excel("rand_rel_1.xlsx", index=False)
