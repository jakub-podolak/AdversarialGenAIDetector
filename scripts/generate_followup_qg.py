import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset

from dotenv import load_dotenv

hf_token = os.getenv["HF_TOKEN"]


print("Loading model...")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=hf_token,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU available: {torch.cuda.is_available()}")

# 1. Download FollowUpQ dataset from Huggingface
print("Loading dataset...")
dataset = load_dataset("Vivian12300/FollowUpQG")

# 2. Extract question, answer, and human-generated follow-up questions
print("Extracting questions, answers...")
questions = dataset["train"]["question"][:10]
answers = dataset["train"]["answer"][:10]
human_fqg = dataset["train"]["follow-up"][:10]


# 3. Function to generate machine-generated follow-up questions using LLaMA
def generate_followup_question(question, answer):
    prompt = f"""Given the following question and answer, generate a relevant follow-up question: 
                 Question: {question}, Answer: {answer}, Follow-up question:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=50, do_sample=True, top_p=0.95, temperature=0.8
        )

    # Decode and return the generated follow-up question
    followup_question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extract the relevant part after "Follow-up question:"
    followup_question = followup_question.split("Follow-up question:")[-1].strip()

    # Truncate at the first newline, if it exists
    followup_question = followup_question.split("\n", 1)[0].strip()

    return followup_question


# 4. Generate follow-up questions and store the results in a list
print("Generate follow-up questions...")
machine_fqg = []

for question, answer in tqdm(
    zip(questions, answers), total=len(questions), desc="Generating follow-up questions"
):
    followup = generate_followup_question(question, answer)
    machine_fqg.append(followup)

# 5. Save the data to a JSON file
print("Saving results...")
output_file = "generated_followup_questions.json"

# Create a list of dictionaries for each question and answer pair
output_data = []
for i in range(len(questions)):
    output_data.append(
        {
            "Original Question": questions[i],
            "Answer": answers[i],
            "Machine-generated Follow-up Question": machine_fqg[i],
            "Human-generated Follow-up Question": human_fqg[i],
        }
    )

# Write the data to a JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file}")

# Print some examples
print("\nExamples of generated follow-up questions:")
for i in range(5):
    print(f"Original Question: {questions[i]}")
    print(f"Answer: {answers[i]}")
    print(f"Machine-generated Follow-up Question: {machine_fqg[i]}")
    print(f"Human-generated Follow-up Question: {human_fqg[i]}\n")
