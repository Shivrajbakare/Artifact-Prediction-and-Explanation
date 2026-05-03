import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_ovis():
    print("[INFO] Loading Ovis (Bypassing Gemma-2 lm_head 4-bit Bug)...")
    cache_path = "./huggingface_cache"
    
    # THE FINAL FIX: Added "lm_head" to the skip list. 
    # This prevents the bitsandbytes AssertionError at generation time.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual_tokenizer", "backbone", "vision_model", "head", "attention", "lm_head"]
    )
    
    # Changed device_map to {"": 0} to fix the "FP4 state not initialized" warning
    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis1.6-Gemma2-9B",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        cache_dir=cache_path,
        device_map={"": 0} 
    )
    
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()

    return model, text_tokenizer, visual_tokenizer

def run_ovis_inference(image, text, model, text_tokenizer, visual_tokenizer):
    query = f'<image>\n{text}'
    device = "cuda"

    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id).to(device)
    
    if isinstance(pixel_values, list):
        pixel_values = [p.to(dtype=torch.float16, device=device) for p in pixel_values]
    else:
        pixel_values = [pixel_values.to(dtype=torch.float16, device=device)]

    print("[INFO] Generating explanation on GPU...")
    
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=150,
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
