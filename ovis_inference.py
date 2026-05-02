import torch
from PIL import Image
from transformers import AutoModelForCausalLM

def load_ovis():
    cache_path = r"D:\huggingface_cache"
    
    print("[INFO] Loading Ovis in STRICT CPU mode (Bypassing meta-tensors)...")
    
    # device_map="auto" ko hata diya taaki meta-tensors na bane
    # model seedha RAM mein load hoga
    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis1.6-Gemma2-9B",
        torch_dtype=torch.float32, # CPU ke liye float32 sabse stable hai
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_path
    ).to("cpu") # Zabardasti CPU par bhejo
    
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def run_ovis_inference(image, text, model, text_tokenizer, visual_tokenizer):
    query = f'<image>\n{text}'

    # Inputs prepare karo
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    
    # Ensure everything is on CPU
    input_ids = input_ids.unsqueeze(0).to("cpu")
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to("cpu")
    pixel_values = [pixel_values.to(dtype=torch.float32, device="cpu")]

    print("[INFO] Generating explanation on CPU (This will take time)...")
    
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=150, # Token limit aur kam kar di taaki jaldi ho
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        
        output_ids = model.generate(
            input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask, 
            **gen_kwargs
        )[0]
        
        return text_tokenizer.decode(output_ids, skip_special_tokens=True)