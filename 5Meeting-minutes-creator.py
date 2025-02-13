import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
    pipeline
)
from huggingface_hub import login
import torch

def main():
    # Path to local audio file
    audio_filename = "denver_extract.mp3"
    
    # Authenticate with Hugging Face
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        hf_token = input("Enter your Hugging Face token: ").strip()
    
    login(hf_token, add_to_git_credential=True)

    # Use Whisper ASR model for speech recognition
    ASR_MODEL = "openai/whisper-tiny"  # Can be changed to "whisper-small", etc.
    device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
    
    asr_pipeline = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)

    print("Transcribing audio...")
    transcription_result = asr_pipeline(audio_filename, return_timestamps=True)

    # Ensure the result contains text
    transcription = transcription_result.get("text", "")
    if not transcription:
        print("Error: No transcription found.")
        return
    
    print("Transcription:\n", transcription)
    
    # Prepare prompt for LLaMA model
    system_message = (
        "You are an assistant that produces minutes of meetings from transcripts, "
        "with a summary, key discussion points, takeaways, and action items with owners, in markdown format."
    )
    user_prompt = (
        "Below is an extract transcript of a Denver council meeting. "
        "Please write minutes in markdown, including a summary with attendees, location and date; "
        "discussion points; takeaways; and action items with owners.\n\n" + transcription
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    
    # Configure 4-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLAMA, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    # Format messages into a prompt
    try:
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    except AttributeError:
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Ensure `inputs` is a dictionary
    if isinstance(inputs, torch.Tensor):  
        inputs = {"input_ids": inputs}  

    # Move inputs to the correct device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {key: value.to(device_str) for key, value in inputs.items()}

    # Create a streamer for tokenized output
    streamer = TextStreamer(tokenizer)

    # Load LLaMA model with quantization
    print("Loading LLaMA model...")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA,
        device_map="auto",
        # quantization_config=quant_config,
        use_auth_token=True
    )

    # Generate meeting minutes from the transcript
    print("Generating meeting minutes...")
    try:
        outputs = model.generate(**inputs, max_new_tokens=2000, streamer=streamer)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during generation: {e}")
        return

    # Print the formatted meeting minutes
    print("\n--- Meeting Minutes ---\n")
    print(response)

if __name__ == "__main__":
    main()
