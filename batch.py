import os
import re
import yaml
import torch
import json
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return config
    except FileNotFoundError:
        print(f"Error: Config file {config_path} not found. Please provide a valid config.yaml.")
        exit(1)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and generate captions using MiMo-VL model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--input_folder", type=str, default=None, help="Path to input folder.")
    parser.add_argument("--output_folder", type=str, default=None, help="Path to output folder.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for caption.")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to prepend to caption prompt.")
    parser.add_argument("--save_format", type=str, default=None, help="Format for captions.")
    parser.add_argument("--max_width", type=int, default=None, help="Max width for resizing.")
    parser.add_argument("--max_height", type=int, default=None, help="Max height for resizing.")
    parser.add_argument("--batch_size", type=int, default=None, help="Number of images to process in a single batch.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--include_thinking", action="store_true", default=None, help="Include thinking tags in output.")
    parser.add_argument("--output_json", action="store_true", default=None, help="Output captions in JSON format.")
    parser.add_argument("--remove_chinese", action="store_true", default=None, help="Remove Chinese characters from output.")
    parser.add_argument("--normalize_text", action="store_true", default=None, help="Normalize punctuation and remove Markdown formatting.")
    parser.add_argument("--print_captions", action="store_true", default=None, help="Print generated captions to console.")
    parser.add_argument("--print_captioning_status", action="store_true", default=None, help="Print status messages for caption saving.")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing caption files.")
    parser.add_argument("--prepend_string", type=str, default=None, help="String to prepend to captions.")
    parser.add_argument("--append_string", type=str, default=None, help="String to append to captions.")
    parser.add_argument("--strip_linebreaks", action="store_true", default=None, help="Remove line breaks from captions.")
    parser.add_argument("--use_custom_prompts", action="store_true", default=None, help="Enable custom prompts per image.")
    parser.add_argument("--custom_prompt_extension", type=str, default=None, help="File extension for custom prompt files.")
    parser.add_argument("--use_metadata_prompts", action="store_true", default=None, help="Enable extracting prompts from image metadata.")
    return parser.parse_args()

def filter_images_without_output(input_folder, save_format, overwrite):
    images_to_caption = []
    skipped_images = 0
    total_images = 0
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                total_images += 1
                image_path = os.path.join(root, file)
                output_path = os.path.splitext(image_path)[0] + save_format
                if not overwrite and os.path.exists(output_path):
                    skipped_images += 1
                    print(f"Skipping {image_path} because caption exists and overwrite is {overwrite}")
                else:
                    images_to_caption.append(image_path)
    return images_to_caption, total_images, skipped_images

def remove_chinese_chars(text):
    return re.sub(r'[\u4e00-\u9fff]', '', text)

def normalize_text(text):
    replacements = {
        '\u2014': ', ',  # Em dash (—) to comma and space
        '\u2018': "'",   # Left single quote (‘) to apostrophe
        '\u2019': "'",   # Right single quote (’) to apostrophe
        '\u201c': '"',   # Left double quote (“) to straight quote
        '\u201d': '"',   # Right double quote (”) to straight quote
        '\u3002': '',    # Ideographic full stop (。) to empty
        '\uff1a': '',    # Full-width colon (：) to empty
        '\u3001': '',    # Ideographic comma (，) to empty
        '\uff1b': '',    # Full-width semicolon (；) to empty
    }
    for unicode_char, simple_char in replacements.items():
        text = text.replace(unicode_char, simple_char)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold **text**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic *text*
    text = re.sub(r'_(.*?)_', r'\1', text)        # Italic _text_
    text = re.sub(r'^#{1,6}\s*(.*?)$', r'\1', text, flags=re.MULTILINE)  # Headers # text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links [text](url)
    text = re.sub(r'`(.*?)`', r'\1', text)        # Inline code `code`
    text = re.sub(r'^>\s*(.*?)$', r'\1', text, flags=re.MULTILINE)  # Blockquotes > text
    text = re.sub(r'^[-*]\s*(.*?)$', r'\1', text, flags=re.MULTILINE)  # Unordered lists - text or * text
    text = re.sub(r'^\d+\.\s*(.*?)$', r'\1', text, flags=re.MULTILINE)  # Ordered lists 1. text
    text = re.sub(r'[–—]', '-', text)  # Normalize dashes
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse spaces
    return text

def save_caption_to_file(image_path, caption, thinking_text, settings):
    txt_file_path = os.path.splitext(image_path)[0] + settings['save_format']
    if settings['remove_chinese']:
        caption = remove_chinese_chars(caption)
        thinking_text = remove_chinese_chars(thinking_text)
    if settings['normalize_text']:
        caption = normalize_text(caption)
        thinking_text = normalize_text(thinking_text)
    if settings['output_json']:
        data = {
            "thinking_text": thinking_text,
            "caption_text": caption
        }
        try:
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                json.dump(data, txt_file, indent=4, ensure_ascii=False)
            if settings['print_captioning_status']:
                print(f"Caption for {os.path.abspath(image_path)} saved in {settings['save_format']} format.")
        except Exception as e:
            print(f"Failed to save caption for {os.path.abspath(image_path)}: {e}")
    else:
        final_caption = settings['prepend_string'] + caption + settings['append_string']
        if settings['include_thinking'] and thinking_text:
            final_caption = f"<think>{thinking_text}</think>{final_caption}"
        final_caption = re.sub(r'\s+', ' ', final_caption)
        try:
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(final_caption or "Failed to generate caption")
            if settings['print_captioning_status']:
                print(f"Caption for {os.path.abspath(image_path)} saved in {settings['save_format']} format.")
        except Exception as e:
            print(f"Failed to save caption for {os.path.abspath(image_path)}: {e}")

def get_custom_prompt(image_path, settings):
    # Priority 1: Custom prompt file
    if settings['use_custom_prompts']:
        prompt_file = os.path.splitext(image_path)[0] + settings['custom_prompt_extension']
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    custom_prompt = f.read().strip()
                if custom_prompt:
                    return custom_prompt
            except Exception as e:
                print(f"Error reading custom prompt for {image_path}: {e}")

    # Priority 2: Metadata prompt
    if settings['use_metadata_prompts']:
        metadata_prompt = _extract_metadata_from_file(image_path)
        if metadata_prompt:
            return metadata_prompt

    # Priority 3: Default prompt
    return settings['prompt']

def _parse_png_parameters(metadata):
    parsed_data = {"positive_prompt": ""}
    params_str = metadata.get('metadata', {}).get('parameters', '')
    if not isinstance(params_str, str):
        return None
    neg_prompt_index = params_str.find('Negative prompt:')
    steps_index = params_str.find('Steps:')
    if neg_prompt_index != -1:
        parsed_data['positive_prompt'] = params_str[:neg_prompt_index].strip()
    elif steps_index != -1:
        parsed_data['positive_prompt'] = params_str[:steps_index].strip()
    else:
        parsed_data['positive_prompt'] = params_str.strip()
    return parsed_data['positive_prompt']

def _extract_metadata_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.png':
            with Image.open(file_path) as img:
                info = dict(img.info)
            metadata = {"file_path": file_path, "metadata": info}
            return _parse_png_parameters(metadata)
        else:
            return None
    except Exception as e:
        print(f"Error reading metadata from {file_path}: {e}")
        return None

def process_images_in_folder(images_to_caption, settings, mimo_model, mimo_processor):
    batch_size = settings.get('batch_size', 1)
    
    for i in tqdm(range(0, len(images_to_caption), batch_size), desc="Processing Batches"):
        batch_paths = images_to_caption[i:i+batch_size]
        batch_images = []
        batch_prompts = []

        for image_path in batch_paths:
            try:
                with Image.open(image_path) as img:
                    img.verify()
                img = Image.open(image_path).convert("RGB")
                image = resize_image_proportionally(img, settings['max_width'], settings['max_height'])
                batch_images.append(image)
                batch_prompts.append(get_custom_prompt(image_path, settings))
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Add placeholders to keep batch size consistent for processing
                batch_images.append(None)
                batch_prompts.append(None)

        # Filter out failed loads
        valid_indices = [j for j, img in enumerate(batch_images) if img is not None]
        if not valid_indices:
            print(f"Skipping batch starting at index {i} as no images could be loaded.")
            continue

        valid_images = [batch_images[j] for j in valid_indices]
        valid_prompts = [batch_prompts[j] for j in valid_indices]
        valid_paths = [batch_paths[j] for j in valid_indices]

        try:
            results = mimo_caption(
                valid_images,
                settings['system_prompt'],
                valid_prompts,
                settings['repetition_penalty'],
                settings['temperature'],
                settings['top_k'],
                settings['output_json'],
                settings['include_thinking'],
                mimo_model,
                mimo_processor,
                settings['strip_linebreaks']
            )

            for idx, (caption, thinking_text) in enumerate(results):
                image_path = valid_paths[idx]
                if len(caption.strip()) < 10:
                    print(f"Short caption for {image_path}: '{caption}'")
                if not caption.strip() and not thinking_text.strip():
                    print(f"Empty output for {image_path}, saving fallback caption.")
                    save_caption_to_file(image_path, "Failed to generate caption", "", settings)
                    continue
                save_caption_to_file(image_path, caption, thinking_text, settings)
                if settings['print_captions']:
                    print(f"Caption for {image_path}: {caption}")

        except Exception as e:
            print(f"Error processing batch starting with {batch_paths[0]}: {e}")
            for image_path in valid_paths:
                save_caption_to_file(image_path, "Failed to generate caption", "", settings)
        
        torch.cuda.empty_cache()


def resize_image_proportionally(image, max_width=None, max_height=None):
    if (max_width is None or max_width <= 0) and (max_height is None or max_height <= 0):
        return image
    original_width, original_height = image.size
    if ((max_width is None or original_width <= max_width) and
        (max_height is None or original_height <= max_height)):
        return image
    if max_width and max_height:
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        ratio = min(width_ratio, height_ratio)
    elif max_width:
        ratio = max_width / original_width
    else:
        ratio = max_height / original_height
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

def mimo_caption(images, system_prompt, prompts, repetition_penalty, temperature, top_k, output_json, include_thinking, mimo_model, mimo_processor, strip_linebreaks):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure PyTorch is compiled with CUDA support.")

    conversations = []
    for prompt in prompts:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        conversations.append(conversation)

    try:
        text_prompts = [mimo_processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
        
        inputs = mimo_processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = mimo_model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=True,
                    temperature=temperature,
                    use_cache=True,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        output_texts = mimo_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results = []
        for output_text in output_texts:
            if strip_linebreaks:
                output_text = output_text.replace('\n', ' ')
            
            think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
                caption = re.sub(r'<think>.*?</think>', '', output_text).strip()
            else:
                thinking_text = ""
                caption = re.sub(r'<think>.*$', '', output_text).strip()

            # Common text cleaning for both caption and thinking_text
            def clean_text(text):
                text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                text = re.sub(r'\*(.*?)\*', r'\1', text)
                text = re.sub(r'_(.*?)_', r'\1', text)
                text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
                text = re.sub(r'`(.*?)`', r'\1', text)
                text = re.sub(r'^>\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
                text = re.sub(r'^\d+\.\s*(.*?)$', r'\1', text, flags=re.MULTILINE)
                text = re.sub(r'[–—]', '-', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            caption = clean_text(caption)
            thinking_text = clean_text(thinking_text)

            if output_json:
                caption = re.sub(r'<think>.*?</think>|<think>.*$', '', caption).strip()
                thinking_text = re.sub(r'<think>.*?</think>|<think>.*$', '', thinking_text).strip()

            if not include_thinking and len(caption) < 10 and thinking_text:
                print(f"Caption too short ('{caption}'), using thinking text")
                caption = thinking_text
                thinking_text = ""
            
            if caption and caption[-1] not in '.!?':
                print(f"Warning: Caption for image may be truncated: {caption[-50:]}")
            
            results.append((caption, thinking_text))
            
        return results
    except Exception as e:
        print(f"Error in mimo_caption: {e}")
        return [("", "")] * len(images)

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config)
    
    # Resolve settings: args > config
    script_dir = Path(__file__).parent
    settings = {
        'input_folder': args.input_folder if args.input_folder is not None else config['input_folder'],
        'output_folder': args.output_folder if args.output_folder is not None else config['output_folder'],
        'save_format': args.save_format if args.save_format is not None else config['save_format'],
        'overwrite': args.overwrite if args.overwrite is not None else config['overwrite'],
        'prompt': args.prompt if args.prompt is not None else config['default_prompt'],
        'system_prompt': args.system_prompt if args.system_prompt is not None else config['default_system_prompt'],
        'max_width': args.max_width if args.max_width is not None else config['max_width'],
        'max_height': args.max_height if args.max_height is not None else config['max_height'],
        'batch_size': args.batch_size if args.batch_size is not None else config.get('batch_size', 1),
        'repetition_penalty': args.repetition_penalty if args.repetition_penalty is not None else config['repetition_penalty'],
        'temperature': args.temperature if args.temperature is not None else config['temperature'],
        'top_k': args.top_k if args.top_k is not None else config['top_k'],
        'include_thinking': args.include_thinking if args.include_thinking is not None else config['include_thinking'],
        'output_json': args.output_json if args.output_json is not None else config['output_json'],
        'remove_chinese': args.remove_chinese if args.remove_chinese is not None else config['remove_chinese'],
        'normalize_text': args.normalize_text if args.normalize_text is not None else config['normalize_text'],
        'print_captions': args.print_captions if args.print_captions is not None else config['print_captions'],
        'print_captioning_status': args.print_captioning_status if args.print_captioning_status is not None else config['print_captioning_status'],
        'prepend_string': args.prepend_string if args.prepend_string is not None else config['prepend_string'],
        'append_string': args.append_string if args.append_string is not None else config['append_string'],
        'strip_linebreaks': args.strip_linebreaks if args.strip_linebreaks is not None else config['strip_linebreaks'],
        'use_custom_prompts': args.use_custom_prompts if args.use_custom_prompts is not None else config['use_custom_prompts'],
        'custom_prompt_extension': args.custom_prompt_extension if args.custom_prompt_extension is not None else config['custom_prompt_extension'],
        'use_metadata_prompts': args.use_metadata_prompts if args.use_metadata_prompts is not None else config.get('use_metadata_prompts', False)
    }
    
    # Resolve relative paths for folders
    settings['input_folder'] = str(script_dir / settings['input_folder']) if not args.input_folder else settings['input_folder']
    settings['output_folder'] = str(script_dir / settings['output_folder']) if not args.output_folder else settings['output_folder']
    
    # Print effective settings cleanly
    print("Captioning using settings:")
    for key, value in settings.items():
        print(f"{key}: {value}")
    
    print (f"\nStarting captioning process\n")
    images_to_caption, total_images, skipped_images = filter_images_without_output(
        settings['input_folder'],
        settings['save_format'],
        settings['overwrite']
    )
    print(f"Found {total_images} image{'s' if total_images != 1 else ''} to caption.")
    if not settings['overwrite']:
        print(f"{skipped_images} image{'s' if skipped_images != 1 else ''} already have captions with format {settings['save_format']}, skipping.")
    
    images_to_process_count = len(images_to_caption)
    batch_size = settings.get('batch_size', 1)
    total_batches = (images_to_process_count + batch_size - 1) // batch_size if images_to_process_count > 0 else 0

    print(f"Using batch size: {batch_size}")
    print(f"Total batches required: {total_batches}\n")
    print(f"Will caption {images_to_process_count} image{'s' if images_to_process_count != 1 else ''}.\n")
    
    if len(images_to_caption) == 0:
        print("No images to process. Exiting.")
    else:
        model_id = "XiaomiMiMo/MiMo-VL-7B-RL"
        mimo_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        mimo_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        process_images_in_folder(
            images_to_caption,
            settings,
            mimo_model,
            mimo_processor
        )