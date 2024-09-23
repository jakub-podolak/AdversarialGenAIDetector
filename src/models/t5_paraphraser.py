from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-large" # T5-large, as in the RADAR paper
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def paraphrase_text(text, num_return_sequences=1, top_k=50, top_p=0.95, temperature=1.0):
    input_text = f"Paraphrase: {text}" # This was the prompt used in RADAR (We can experiment with this?)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        input_ids,
        max_length=512,
        do_sample=True,
        top_k=top_k,                  # Top-k sampling (default value from RADAR)
        top_p=top_p,                  # Nucleus sampling (default value from RADAR)
        temperature=temperature,
        num_return_sequences=num_return_sequences,  # Number of paraphrases to return
        no_repeat_ngram_size=2,       # Prevents repetitive phrases
    )

    paraphrases = [tokenizer.decode(output, 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True) for output in outputs]
    return paraphrases

# Here's an example how to use the model for paraphrasing
input_text = "The AI model was able to detect human-like text with high accuracy."
paraphrases = paraphrase_text(input_text, num_return_sequences=3, top_k=50, top_p=0.95)

# Print the paraphrased outputs
for idx, paraphrase in enumerate(paraphrases, 1):
    print(f"Paraphrase {idx}: {paraphrase}")
