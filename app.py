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

df_items = pd.read_excel("Classeur1.xlsx", sheet_name="Items")
df_props = pd.read_excel("Classeur1.xlsx", sheet_name="Properties")
l_items = dict(map(lambda i,j : (i,j) , list(df_items["ID"]), list(df_items["Label"])))
l_props = dict(map(lambda i,j : (i,j) , list(df_props["ID"]), list(df_props["Label"])))
df_verify = pd.read_excel("ClinMed.xlsx", sheet_name="To Verify")
def get_item (x):
  try:
    return l_items[x]
  except KeyError:
    return "____"
df_verify["subj"] = df_verify["subject"].apply(get_item)
df_verify["obj"] = df_verify["object"].apply(get_item)
df_verify["p"] = df_verify["prop"].apply(lambda x: l_props[x])

def check_relation_with_llama(subject, property, object):
  response = ""
  while (response.upper().strip() != "TRUE") and (response.upper().strip() != "FALSE"):
    response = query(MODEL_Q8_0, f"Is the relation [{subject}, {property}, {object}] accurate? Answer with one word: TRUE or FALSE.")
    generated_text = response
  return response.upper().strip()

df_verify["llama_valid"] = ""  # Initialize a new column for LLAMA validity

for index, row in df_verify.iterrows():
  subject = row["subj"]
  property = row["p"]
  object = row["obj"]

  if subject != "____" and object != "____":
    is_valid = []
    for i in range(5):
      is_valid.append(check_relation_with_llama(subject, property, object))
    print(is_valid.count("TRUE")/5)

    df_verify.at[index, "llama_valid"] = is_valid.count("TRUE")/5

df_verify.to_excel("verify_with_llama.xlsx", index=False)