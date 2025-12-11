import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTConfig, SFTTrainer, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
from sklearn.model_selection import train_test_split

# ============================================================================
# PART 1: PREPARE YOUR DATASET FOR SFT
# ============================================================================

def load_and_process_fire_csv(csv_path="data.csv"):
    """
    Load and process the fire incident CSV data.
    
    Returns formatted dataset ready for training.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create training examples by combining environmental context with actions
    formatted_data = []
    
    for idx, row in df.iterrows():
        # Build environmental context
        context_parts = []
        
        if pd.notna(row.get('Fire Name')):
            context_parts.append(f"Fire Incident: {row['Fire Name']}")
        
        if pd.notna(row.get('Fire Status')):
            context_parts.append(f"Current Status: {row['Fire Status']}")
        
        if pd.notna(row.get('Environmental Observations')):
            context_parts.append(f"Environmental Conditions: {row['Environmental Observations']}")
        
        if pd.notna(row.get('Coordinate Location')):
            context_parts.append(f"Location: {row['Coordinate Location']}")
        
        if pd.notna(row.get('Unit Type Assigned')):
            context_parts.append(f"Unit Type: {row['Unit Type Assigned']}")
        
        environmental_context = "\n".join(context_parts)
        
        # Get action
        action = row.get('Action', '')
        reasoning = row.get('Reasoning', '')
        
        if pd.notna(action) and action.strip():
            # Create a structured prompt
            if pd.notna(reasoning) and reasoning.strip():
                full_action = f"{action}\nReasoning: {reasoning}"
            else:
                full_action = action
            
            formatted_text = f"""### Fire Incident Analysis

{environmental_context}

### Recommended Action
{full_action}"""
            
            formatted_data.append({'text': formatted_text})
    
    print(f"Created {len(formatted_data)} training examples")
    print(f"Sample example:\n{'-'*60}\n{formatted_data[0]['text'][:500]}...\n{'-'*60}")
    
    return Dataset.from_dict({'text': [d['text'] for d in formatted_data]})


def prepare_fire_dataset(data_path_or_list):
    """
    Convert your fire data into HuggingFace Dataset format.
    
    Expected format: List of dicts with keys:
    - 'environmental_context': str (model outputs describing fire characteristics)
    - 'action': str (natural language action for firefighters)
    """
    
    if isinstance(data_path_or_list, str):
        # Check if it's a CSV file
        if data_path_or_list.endswith('.csv'):
            return load_and_process_fire_csv(data_path_or_list)
        else:
            with open(data_path_or_list, 'r') as f:
                data = json.load(f)
    else:
        data = data_path_or_list
    
    formatted_data = []
    for item in data:
        formatted_data.append({
            'text': f"Environmental Data:\n{item['environmental_context']}\n\nAction:\n{item['action']}"
        })
    
    return Dataset.from_dict({'text': [d['text'] for d in formatted_data]})

# ============================================================================
# PART 2: SUPERVISED FINE-TUNING (SFT)
# ============================================================================

def run_sft(
    model_name="microsoft/phi-2",  # Smaller model that works well on consumer GPUs
    dataset=None,
    output_dir="./sft_model",
    use_4bit=True
):
    """
    Fine-tune a foundation model on your fire response dataset.
    Uses 4-bit quantization for efficient training on consumer GPUs.
    """
    
    print(f"\n{'='*60}")
    print("STARTING SUPERVISED FINE-TUNING (SFT)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Output directory: {output_dir}")
    print(f"Using 4-bit quantization: {use_4bit}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Initialize tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # LoRA configuration for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "dense"]  # Attention layers
    )
    
    # Training configuration
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        dataset_text_field="text",
        packing=True,
        fp16=torch.cuda.is_available(),
        report_to="none",  # Disable wandb/tensorboard for simplicity
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )
    
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("SFT TRAINING COMPLETE")
    print(f"{'='*60}\n")
    
    return trainer

# ============================================================================
# PART 3: REAL-TIME REINFORCEMENT LEARNING WITH USER FEEDBACK
# ============================================================================

class SimpleRewardModel(torch.nn.Module):
    """
    Simple reward model that learns from user accept/reject feedback.
    In production, this would be a separate fine-tuned model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        reward = torch.tanh(last_hidden.mean(dim=1))
        return reward

def create_reward_signal(user_feedback, feedback_type='binary'):
    """
    Convert user feedback (accept/reject) to reward signal.
    
    Args:
        user_feedback: bool (True = accept, False = reject)
        feedback_type: 'binary' or 'scalar'
    
    Returns:
        reward: float between -1 and 1
    """
    if feedback_type == 'binary':
        return 1.0 if user_feedback else -1.0
    else:
        # Could be continuous score from 0-10
        return (user_feedback / 5.0) - 1.0

def run_online_rl(
    model_path,
    prompt_dataset,
    num_rl_steps=50,
    output_dir="./rl_model"
):
    """
    Run real-time reinforcement learning with simulated user feedback.
    
    In production, this would continuously:
    1. Model generates response to prompt
    2. User accepts/rejects (simulated here based on quality heuristics)
    3. Reward signal updates model
    """
    print(f"\n{'='*60}")
    print("STARTING REINFORCEMENT LEARNING (PPO)")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Number of RL steps: {num_rl_steps}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the fine-tuned model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        offload_dir="./offload"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    ppo_config = PPOConfig(
        model_name=model_path,
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
    )
    
    print("Initializing PPO Trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
    )
    
    print(f"\nStarting online RL training for {num_rl_steps} steps...")
    print("Using simulated feedback based on response quality heuristics\n")
    
    generation_kwargs = {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    for step in range(num_rl_steps):
        # Get a prompt from the dataset
        prompt_idx = step % len(prompt_dataset)
        prompt_text = prompt_dataset[prompt_idx]['text']
        
        # Extract just the context part (before "### Recommended Action")
        if "### Recommended Action" in prompt_text:
            context_only = prompt_text.split("### Recommended Action")[0].strip()
        else:
            context_only = prompt_text[:200]  # First 200 chars
        
        # Tokenize the prompt
        query_tensors = tokenizer.encode(context_only, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            response_tensors = model.generate(
                query_tensors,
                **generation_kwargs
            )
        
        # Decode the response
        response_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
        generated_part = response_text[len(context_only):].strip()
        
        # Simulate user feedback based on quality heuristics
        reward = evaluate_response_quality(generated_part)
        
        if (step + 1) % 10 == 0:
            print(f"\n--- Step {step + 1}/{num_rl_steps} ---")
            print(f"Context: {context_only[:100]}...")
            print(f"Generated: {generated_part[:150]}...")
            print(f"Reward: {reward:.2f}")
        
        # Prepare for PPO update
        response_only = response_tensors[0][query_tensors.shape[1]:]
        
        # PPO step
        stats = ppo_trainer.step(
            [query_tensors[0]],
            [response_only],
            [torch.tensor(reward)]
        )
        
        if (step + 1) % 10 == 0 and stats:
            print(f"Stats: {stats}")
    
    print(f"\nSaving RL-trained model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n{'='*60}")
    print("RL TRAINING COMPLETE")
    print(f"{'='*60}\n")


def evaluate_response_quality(response_text):
    """
    Evaluate the quality of a generated response using heuristics.
    Returns a reward between -1 and 1.
    
    In production, this would be replaced with actual user feedback
    or a trained reward model.
    """
    reward = 0.0
    
    # Positive signals
    action_keywords = ['deploy', 'evacuate', 'establish', 'call', 'set up', 
                       'supply', 'patrol', 'apply', 'manage', 'coordinate']
    if any(keyword in response_text.lower() for keyword in action_keywords):
        reward += 0.3
    
    # Check for reasoning/explanation
    reasoning_keywords = ['because', 'due to', 'reasoning:', 'to ensure', 'in order to']
    if any(keyword in response_text.lower() for keyword in reasoning_keywords):
        reward += 0.2
    
    # Check for specificity (mentions units, resources, directions)
    specific_keywords = ['engine', 'truck', 'battalion', 'water', 'perimeter', 
                        'north', 'south', 'east', 'west', 'resources']
    if any(keyword in response_text.lower() for keyword in specific_keywords):
        reward += 0.2
    
    # Length check - not too short, not too long
    if 20 < len(response_text) < 500:
        reward += 0.2
    elif len(response_text) < 10:
        reward -= 0.3
    
    # Negative signals
    if len(response_text) < 5:
        reward = -1.0
    
    # Normalize to [-1, 1]
    return np.clip(reward, -1.0, 1.0)

# ============================================================================
# PART 4: FULL PIPELINE EXAMPLE
# ============================================================================

def test_model(model_path, test_prompts):
    """
    Test the trained model with sample prompts.
    """
    print(f"\n{'='*60}")
    print("TESTING TRAINED MODEL")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"INPUT:\n{prompt}\n")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()
        
        print(f"GENERATED:\n{generated}\n")
        print("-" * 60)


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print(" " * 20 + "FIREFIGHTER AI DECISION SUPPORT SYSTEM")
    print(" " * 25 + "Full Production Pipeline")
    print("="*80 + "\n")
    
    # Step 1: Load and prepare dataset from CSV
    print("STEP 1: Loading and processing fire incident data from CSV...")
    dataset = prepare_fire_dataset("data.csv")
    
    # Split into train and validation sets
    train_test = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test['train']
    val_dataset = train_test['test']
    
    print(f"\nDataset split:")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    
    # Show a sample
    print(f"\nSample training example:")
    print("-" * 60)
    print(train_dataset[0]['text'][:400] + "...")
    print("-" * 60)
    
    # Step 2: Run Supervised Fine-Tuning
    print("\n\nSTEP 2: Running Supervised Fine-Tuning (SFT)...")
    sft_trainer = run_sft(
        model_name="microsoft/phi-2",  # Using Phi-2 - good performance, fits on consumer GPUs
        dataset=train_dataset,
        output_dir="./sft_model",
        use_4bit=True
    )
    
    # Step 3: Run Reinforcement Learning
    print("\n\nSTEP 3: Running Reinforcement Learning with PPO...")
    run_online_rl(
        model_path="./sft_model",
        prompt_dataset=val_dataset,
        num_rl_steps=30,  # Reduced for faster iteration
        output_dir="./rl_model"
    )
    
    # Step 4: Test the final model
    print("\n\nSTEP 4: Testing the trained model...")
    
    test_prompts = [
        """### Fire Incident Analysis

Fire Incident: Test Residential Fire
Current Status: Structure fire spreading to adjacent building
Environmental Conditions: Wind speed 20 mph SW, Temperature 88F, Low humidity
Location: Residential neighborhood
Unit Type: Fire Engine

### Recommended Action""",
        """### Fire Incident Analysis

Fire Incident: Wildfire Test
Current Status: Growing wildfire in forest area
Environmental Conditions: Extreme winds 45 mph, very dry conditions, dense vegetation
Location: Sierra Nevada Mountains

### Recommended Action"""
    ]
    
    test_model("./rl_model", test_prompts)
    
    print("\n" + "="*80)
    print(" " * 30 + "PIPELINE COMPLETE")
    print("="*80)
    print("\nModels saved:")
    print("  - Supervised Fine-Tuned Model: ./sft_model")
    print("  - RL-Enhanced Model: ./rl_model")
    print("\nYou can now use the RL model for inference on new fire incidents.")
    print("="*80 + "\n")