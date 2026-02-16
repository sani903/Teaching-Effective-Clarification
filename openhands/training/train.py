import os
import json
import ray
import torch
import logging
import numpy as np
import pandas as pd
import gc
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, deque

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RewardConfig:
    model_name: str = "Qwen/Qwen3-32B"
    temperature: float = 0.001
    max_tokens: int = 1000
    
    # Stage thresholds
    stage1_threshold: float = 0.5  # Redundancy
    stage2_threshold: float = 0.5  # Novelty
    stage3_threshold: float = 0  # Answerability
    
    # Weights
    stage1_weight: float = 0.25     # Redundancy
    stage2_weight: float = 0.25    # Novelty
    stage3_weight: float = 0.25     # Answerability
    stage4_weight: float = 0.25    # Utility + Specificity
    
    # Generation cache
    cache_size: int = 2  # Keep last N generations
    
    # Batching
    reward_batch_size: int = 16

@dataclass
class Config:
    base_model: str = ""
    train_data: str = ""
    csv_path: str = ""
    output_dir: str = ""
    
    # Training
    learning_rate: float = 5e-6
    batch_size: int = 4
    num_epochs: int = 1
    max_length: int = 1024
    max_new_tokens: int = 256
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 5
    
    # GRPO
    num_samples_per_prompt: int = 8
    baseline_type: str = "mean"
    kl_coef: float = 0.05
    
    # Generation
    policy_temperature: float = 0.9
    policy_top_p: float = 0.9
    use_generation_cache: bool = True
    
    # Cross-sample diversity bonus
    diversity_bonus_weight: float = 0
    
    # Reward model
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    
    # GPUs
    actor_gpu: int = 0
    reference_gpu: int = 1
    reward_gpu: int = 2
    
    # Saving
    save_steps: int = 10
    keep_last_checkpoints: int = 5
    log_every: int = 5

def format_prompt(text: str) -> str:
    """Extract task and format prompt"""
    cleaned = text.strip()
    if "Task:" in cleaned:
        task_start = cleaned.find("Task:") + 5
        cleaned = cleaned[task_start:].strip()
    
    system = "Generate questions, if any, to ask the user to recover missing information required to solve the task."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{cleaned}<|im_end|>\n<|im_start|>assistant\n"

def extract_task_only(text: str) -> str:
    """Extract just the task content"""
    if "Task:" in text:
        task_start = text.find("Task:") + 5
        return text[task_start:].strip()
    return text.strip()

# ============================================================================
# Generation Cache
# ============================================================================

class GenerationCache:
    """Cache recent generations for novelty checking"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = deque(maxlen=max_size)
    
    def add(self, generation: str):
        """Add a generation to the cache"""
        self.cache.append(generation)
    
    def get_recent(self) -> List[str]:
        """Get all recent generations"""
        return list(self.cache)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()

# ============================================================================
# Diversity Calculation
# ============================================================================

def compute_cross_sample_diversity(samples_lists: List[List[str]]) -> List[float]:
    """
    Reward diversity across N samples for the same prompt.
    Returns a bonus (0-1) for each prompt based on unique questions.
    """
    bonuses = []
    
    for questions_lists in samples_lists:
        all_questions = []
        for questions in questions_lists:
            all_questions.extend(questions)
        
        if not all_questions:
            bonuses.append(0.0)
            continue
        
        # Normalize questions for comparison
        normalized = []
        for q in all_questions:
            q_norm = ' '.join(sorted(set(q.lower().split())))
            normalized.append(q_norm)
        
        unique_ratio = len(set(normalized)) / len(normalized)
        bonuses.append(unique_ratio)
    
    return bonuses

# ============================================================================
# Four-Stage Reward Model with Generation Cache
# ============================================================================

@ray.remote(num_gpus=2, num_cpus=8)
class OptimizedRewardWorker:
    """
    Four-stage reward model:
    Stage 1: Redundancy check (non-redundancy score)
    Stage 2: Novelty check (similarity to recent generations)
    Stage 3: Answerability check
    Stage 4: Utility + Specificity check
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.reward_config = config.reward_config
        
        logger.info(f"Loading reward model: {self.reward_config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.reward_config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.parse_stats = {
            'json_failures': 0,
            'empty_questions': 0,
            'answer_format_issues': 0,
            'total_calls': 0
        }
        
        self.call_count = 0
        
        # Generation cache
        self.generation_cache = GenerationCache(max_size=self.reward_config.cache_size)
        
        logger.info(f"✓ Reward model ready (4 stages, cache_size={self.reward_config.cache_size})")
    
    def _create_messages(self, system: str, user: str) -> List[Dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    
    def _call_model_batch(self, messages_list: List[List[Dict]]) -> List[str]:
        """Batched model calls"""
        if not messages_list:
            return []
        
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in messages_list
        ]
        
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.reward_config.max_tokens,
                temperature=self.reward_config.temperature if self.reward_config.temperature > 0 else None,
                do_sample=self.reward_config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        self.call_count += len(messages_list)
        
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[inputs['input_ids'][i].shape[0]:],
                skip_special_tokens=True
            ).strip()
            
            
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = response.strip()

            for stop in ["```", "<|im_end|>", "\n\nNote:"]:
                if stop in response:
                    response = response[:response.index(stop)]

            responses.append(response.strip())        
        return responses

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Robust JSON extraction - always returns dict or None"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = text.strip()
        self.parse_stats['total_calls'] += 1
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match)
                if isinstance(result, dict):
                    return result
            except:
                fixed = match.replace("'", '"')
                fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                fixed = re.sub(r'(\w+):', r'"\1":', fixed)
                try:
                    result = json.loads(fixed)
                    if isinstance(result, dict):
                        return result
                except:
                    continue
        try:
            fields = {}
            for field in ['redundant_count', 'answerable_count', 'similar_count', 'total', 'utility', 'specificity']:
                pattern = rf'"{field}"\s*:\s*(\d+(?:\.\d+)?)'
                match = re.search(pattern, text)
                if match:
                    val = match.group(1)
                    fields[field] = float(val) if '.' in val else int(val)
            exp_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', text)
            if exp_match:
                fields['explanation'] = exp_match.group(1)
            if fields:
                return fields  # This is already a dict
        except Exception as e:
            logger.debug(f"JSON extraction fallback failed: {e}")
        self.parse_stats['json_failures'] += 1
        return None  # Always return None instead of other types

    

    def _parse_questions(self, text: str) -> List[str]:
        """Parse questions from generation"""
        if not text or text.strip() == "<think></think>":
            return []
        
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        questions = []
        numbered = re.findall(r'^\d+\.\s*(.+?)(?=\n\d+\.|$)', text, re.MULTILINE | re.DOTALL)
        if numbered:
            for q in numbered:
                clean = q.strip()
                if len(clean) > 10:
                    questions.append(clean)
            return questions
        
        for line in text.split('\n'):
            line = re.sub(r'^\d+\.\s*', '', line).strip()
            if len(line) > 10:
                questions.append(line)
        
        if not questions and text and text.strip() != "<think></think>":
            self.parse_stats['empty_questions'] += 1
        
        return questions
    
    def _stage1_batch(self, contexts: List[str], questions_lists: List[List[str]]) -> List[Tuple[float, str, List[Dict]]]:
        """Stage 1: Batched redundancy check"""
        results = []
        context_to_indices = defaultdict(list)
        
        for i, context in enumerate(contexts):
            context_to_indices[context].append(i)
        
        all_results = [None] * len(contexts)
        
        for context, indices in context_to_indices.items():
            questions_batch = [questions_lists[i] for i in indices]
            
            non_empty = [(i, qs) for i, qs in zip(indices, questions_batch) if qs]
            if not non_empty:
                for i in indices:
                    all_results[i] = (0.5, "No questions", [])
                continue
            
            batch_indices, batch_questions = zip(*non_empty)
            
            try:
                normal_batch_indices = []
                normal_batch_questions = []
                
                for batch_idx, questions in zip(batch_indices, batch_questions):
                    # Normal processing needed
                    normal_batch_indices.append(batch_idx)
                    normal_batch_questions.append(questions)
                
                # Pass 1: Answer questions (only for non-Bokeh batches)
                messages_list = []
                for questions in normal_batch_questions:
                    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                    
                    system = """Answer questions based solely on provided context.
    If context doesn't contain enough information, respond with "I don't know".

    Format your response EXACTLY as:
    A1: <answer>
    A2: <answer>
    ...

    Do not add any extra text, explanations, or formatting."""                    
                    user = f"""Context:
    {context}

    Questions:
    {questions_text}

    Based ONLY on the context, answer each question. Format:
    A1: <answer or "I don't know">
    A2: <answer or "I don't know">
    ...
    Answers:"""
                    
                    messages_list.append(self._create_messages(system, user))
                
                answers_texts = self._call_model_batch(messages_list)
                
                all_qa_pairs = []
                for questions, answers_text in zip(normal_batch_questions, answers_texts):
                    answer_lines = [l.strip() for l in answers_text.split('\n') if l.strip()]
                    qa_pairs = []

                    for i, question in enumerate(questions):
                        answer = "Formatting issue"
        
                        patterns = [
                            rf"^A{i+1}:\s*(.+)",           # A1: answer
                            rf"^A{i+1}\.\s*(.+)",          # A1. answer
                            rf"^Answer\s+{i+1}:\s*(.+)",   # Answer 1: answer
                            rf"^{i+1}\.\s*(.+)",           # 1. answer
                            rf"^\**A{i+1}\**:\s*(.+)",     # **A1**: answer
                        ]
        
                        for line in answer_lines:
                            for pattern in patterns:
                                match = re.match(pattern, line, re.IGNORECASE)
                                if match:
                                    answer = match.group(1).strip()
                                    break
                            if answer != "Formatting issue":
                                break
        
                        if answer == "Formatting issue":
                            self.parse_stats['answer_format_issues'] += 1
        
                        qa_pairs.append({'question': question, 'answer': answer})

                    all_qa_pairs.append(qa_pairs)

                # Pass 2: Evaluate redundancy
                messages_list = []
                for questions, qa_pairs in zip(normal_batch_questions, all_qa_pairs):
                    qa_text = "\n".join([
                        f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}"
                        for i, qa in enumerate(qa_pairs)
                    ])
                    
                    system = """Evaluate whether questions were already answered by context.
    Respond ONLY with valid JSON in the exact format shown below."""

                    user = f"""Context:
    {context}

    Questions and Answers:
    {qa_text}

    Count questions with SUFFICIENT answers (not "I don't know" or "Formatting issue"). These are REDUNDANT. 

    Respond with ONLY this JSON (no extra text):
    {{"redundant_count": <number>, "total": {len(questions)}, "explanation": "<brief>"}}

    Your response:"""
                    messages_list.append(self._create_messages(system, user))
                
                responses = self._call_model_batch(messages_list)
                
                for batch_idx, questions, qa_pairs, response in zip(
                    normal_batch_indices, normal_batch_questions, all_qa_pairs, responses
                ):
                    data = self._extract_json(response)
                    
                    if not data or 'redundant_count' not in data:
                        redundant = sum(1 for qa in qa_pairs if "don't know" not in qa['answer'].lower())
                        score = 1.0 - (redundant / len(questions))
                        all_results[batch_idx] = (score, f"Parse failed (fallback: {redundant}/{len(questions)})", qa_pairs)
                    else:
                        redundant = data.get('redundant_count', 0)
                        total = data.get('total', len(questions))
                        explanation = data.get('explanation', '')
                        score = 1.0 - (redundant / total) if total > 0 else 0.5
                        all_results[batch_idx] = (score, f"{redundant}/{total} redundant. {explanation}", qa_pairs)
            
            except Exception as e:
                logger.error(f"Stage 1 batch error: {e}")
                for idx in batch_indices:
                    all_results[idx] = (0.5, f"Error: {str(e)[:50]}", [])
                continue
        
        for i in range(len(all_results)):
            if all_results[i] is None:
                all_results[i] = (0.5, "Processing skipped", [])
        
        return all_results
    
    def _stage2_novelty_batch(self, current_generations: List[str], questions_lists: List[List[str]]) -> List[Tuple[float, str]]:
        """
        Stage 2: Novelty check - compare current generation to cached previous generations
        """
        cached_generations = self.generation_cache.get_recent()
        
        if not cached_generations:
            # No cache yet, all generations are novel
            return [(1.0, "First generations (cache empty)") for _ in current_generations]
        
        messages_list = []
        valid_indices = []
        
        # Format cached generations for display
        cached_text = "\n\n".join([
            f"Generation {i+1}:\n{gen}"
            for i, gen in enumerate(cached_generations)
        ])
        
        for i, (current_gen, questions) in enumerate(zip(current_generations, questions_lists)):
            if not questions:
                continue
            
            questions_text = "\n".join(f"{j+1}. {q}" for j, q in enumerate(questions))
            
            system = """Evaluate question novelty by comparing current questions to previous generations and questions within the same generation set.
Respond ONLY with valid JSON."""

            user = f"""Previous Generations:
{cached_text}

Current Generation:
{questions_text}

Count how many current questions are SIMILAR to previously asked questions or to other questions in the current generation.

Respond with ONLY this JSON (no extra text):
{{"similar_count": <number>, "total": {len(questions)}, "explanation": "<brief>"}}

Your response:"""            
            messages_list.append(self._create_messages(system, user))
            valid_indices.append(i)
        
        if messages_list:
            responses = self._call_model_batch(messages_list)
        else:
            responses = []
        
        results = []
        response_idx = 0
        
        for i, questions in enumerate(questions_lists):
            if not questions:
                results.append((0.5, "No questions"))
            elif i in valid_indices:
                response = responses[response_idx]
                response_idx += 1
                
                data = self._extract_json(response)
                if not data or 'similar_count' not in data:
                    results.append((0.7, "Parse failed"))
                else:
                    similar = data.get('similar_count', 0)
                    total = data.get('total', len(questions))
                    explanation = data.get('explanation', '')
                    # High score = fewer similar questions = more novel
                    score = 1.0 - (similar / total) if total > 0 else 0.7
                    results.append((score, f"{similar}/{total} similar. {explanation}"))
            else:
                results.append((0.5, "Skipped"))
        
        return results
    
    def _stage3_answerability_batch(self, original_issues: List[Optional[str]], questions_lists: List[List[str]]) -> List[Tuple[float, str]]:
        """Stage 3: Batched answerability check with two-pass evaluation"""
        results = []
        valid_data = []
        
        # Collect valid entries for processing
        for i, (orig, questions) in enumerate(zip(original_issues, questions_lists)):
            if not questions:
                results.append((0.5, "No questions"))
            elif not orig or orig.strip() == "":
                results.append((0.7, "No original issue"))
            else:
                valid_data.append((i, orig, questions))
                results.append(None)  # Placeholder
        
        if not valid_data:
            return results
        
        try:
            # Pass 1: Answer questions from original bug report
            messages_list = []
            for _, orig, questions in valid_data:
                questions_text = "\n".join([f"{j+1}. {q}" for j, q in enumerate(questions)])
                
                system = """Answer questions based solely on the original bug report.
    If the report doesn't contain enough information, respond with "I don't know".
    If a question is generic or doesn't ask for specific information, respond with "Generic question".

    Format your response EXACTLY as:
    A1: <answer>
    A2: <answer>
    ..."""
                
                user = f"""Original Bug Report:
    {orig}

    Questions:
    {questions_text}

    Based ONLY on the original bug report, determine if the user who filed the report can answer each question. For each question:
    - If the report contains the information needed to answer: provide the answer
    - If the report doesn't have this information: "I don't know"
    - If the question is too generic ("provide more details", "clarify requirements"): "Generic question"

    Format:
    A1: <answer or "I don't know" or "Generic question">
    A2: <answer or "I don't know" or "Generic question">
    ...

    Answers:"""
                
                messages_list.append(self._create_messages(system, user))
            
            answers_texts = self._call_model_batch(messages_list)
            
            # Parse answers and create Q&A pairs
            all_qa_pairs = []
            for (_, orig, questions), answers_text in zip(valid_data, answers_texts):
                answer_lines = [l.strip() for l in answers_text.split('\n') if l.strip()]
                qa_pairs = []
                
                for i, question in enumerate(questions):
                    answer = "Formatting issue"
                    
                    patterns = [
                        rf"^A{i+1}:\s*(.+)",           # A1: answer
                        rf"^A{i+1}\.\s*(.+)",          # A1. answer
                        rf"^Answer\s+{i+1}:\s*(.+)",   # Answer 1: answer
                        rf"^{i+1}\.\s*(.+)",           # 1. answer
                        rf"^\**A{i+1}\**:\s*(.+)",     # **A1**: answer
                    ]
                    
                    for line in answer_lines:
                        for pattern in patterns:
                            match = re.match(pattern, line, re.IGNORECASE)
                            if match:
                                answer = match.group(1).strip()
                                break
                        if answer != "Formatting issue":
                            break
                    
                    if answer == "Formatting issue":
                        self.parse_stats['answer_format_issues'] += 1
                    
                    qa_pairs.append({'question': question, 'answer': answer})
                
                all_qa_pairs.append(qa_pairs)
            
            # Pass 2: Evaluate answerability
            messages_list = []
            for (_, orig, questions), qa_pairs in zip(valid_data, all_qa_pairs):
                qa_text = "\n".join([
                    f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}"
                    for i, qa in enumerate(qa_pairs)
                ])
                
                system = """Evaluate whether questions can be answered by the user based on their original bug report.
    Respond ONLY with valid JSON in the exact format shown below."""
                
                user = f"""Original Bug Report:
    {orig}

    Questions and Answers:
    {qa_text}

    Count questions that are ANSWERABLE by the user (those with actual answers, not "I don't know" or "Generic question"). 
    Generic questions that don't ask for specific information are NOT answerable.

    Respond with ONLY this JSON (no extra text):
    {{"answerable_count": <number>, "total": {len(questions)}, "explanation": "<brief>"}}

    Your response:"""
                
                messages_list.append(self._create_messages(system, user))
            
            responses = self._call_model_batch(messages_list)
            
            # Process responses and fill in results
            for (idx, orig, questions), qa_pairs, response in zip(valid_data, all_qa_pairs, responses):
                data = self._extract_json(response)
                
                if not data or 'answerable_count' not in data:
                    # Fallback: count non-"I don't know" and non-"Generic" answers
                    answerable = sum(1 for qa in qa_pairs 
                                if "don't know" not in qa['answer'].lower() 
                                and "generic question" not in qa['answer'].lower()
                                and "formatting issue" not in qa['answer'].lower())
                    score = answerable / len(questions) if len(questions) > 0 else 0.5
                    results[idx] = (score, f"Parse failed (fallback: {answerable}/{len(questions)})")
                else:
                    answerable = data.get('answerable_count', 0)
                    total = data.get('total', len(questions))
                    explanation = data.get('explanation', '')
                    score = answerable / total if total > 0 else 0.5
                    results[idx] = (score, f"{answerable}/{total} answerable. {explanation}")
        
        except Exception as e:
            logger.error(f"Stage 3 batch error: {e}")
            for i, result in enumerate(results):
                if result is None:
                    results[i] = (0.5, f"Error: {str(e)[:50]}")
        
        # Ensure all results are filled
        for i in range(len(results)):
            if results[i] is None:
                results[i] = (0.5, "Processing skipped")
        
        return results
    
    def _stage4_utility_specificity_batch(self, contexts: List[str], questions_lists: List[List[str]]) -> List[Tuple[float, str]]:
        """
        Stage 4: Information type scoring
        Rewards based on question relevance and information type importance
        """
        messages_list = []
        valid_indices = []
        
        for i, (context, questions) in enumerate(zip(contexts, questions_lists)):
            if not questions:
                continue
            
            questions_text = "\n".join(f"{j+1}. {q}" for j, q in enumerate(questions))
            
            system = """You are classifying clarification questions by information type for software issues.
    Respond with valid JSON only."""
            
            user = f"""Task:
    {context}

    Questions:
    {questions_text}

    Classify EACH question by the information need it addresses:

    INFORMATION TYPES (type: description):

    error_info: What is going wrong, why, and how do we know?
    - Examples: "What's the error message?", "Can you share the stack trace?", "What incorrect output do you see?"

    reproduction: Under what conditions does the failure occur?
    - Examples: "What command triggers this?", "What input causes the crash?", "Can you provide steps to reproduce?"

    expected_behavior: What should happen instead of the observed behavior?
    - Examples: "What output format do you expect?", "What should the function return?", "What's the desired behavior?"

    implementation: How should the solution be implemented?
    - Examples: "Should this use async?", "What approach do you prefer?", "Any constraints on the solution?"

    external_refs: What external systems or resources are relevant?
    - Examples: "Link to the API docs?", "Which dataset are you using?", "Related PR or issue?"

    version_env: What configuration is needed to reproduce?
    - Examples: "Python version?", "OS details?", "GPU or CPU?"

    irrelevant: Question mentions entities not in the issue, is too generic, or unrelated to the task.
    - Examples: "Can you provide more details?", "What library are you using?" (when library is already specified), "Can you share a screenshot?" (for non-visual issues), questions about Bokeh/Jupyter when not mentioned in the issue
    - Vague or off-topic questions that don't target specific information gaps in this particular issue
    
    Respond with ONLY this JSON:
    {{"classifications": [
        {{"question_num": 1, "info_type": "<type>"}},
        {{"question_num": 2, "info_type": "<type>"}},
        ...
    ]}}"""

            messages_list.append(self._create_messages(system, user))
            valid_indices.append(i)
        
        if messages_list:
            responses = self._call_model_batch(messages_list)
        else:
            responses = []
        
        # Rewards based on frequency/importance from taxonomy
        TYPE_REWARDS = {
            "error_info": 1.0,          
            "expected_behavior": 0.58,    
            "implementation": 0.65,       
            "reproduction": 0.42,         
            "version_env": 0.59,
            "external_refs": 0.23,        
            "irrelevant": 0.0,          # No reward for irrelevant questions
        }
        
        results = []
        response_idx = 0
        
        for i, questions in enumerate(questions_lists):
            if not questions:
                results.append((0.0, "No questions"))
            elif i in valid_indices:
                response = responses[response_idx]
                response_idx += 1
                
                data = self._extract_json(response)
                if not data or 'classifications' not in data:
                    results.append((0.3, "Parse failed"))
                else:
                    classifications = data.get('classifications', [])
                    
                    if len(classifications) != len(questions):
                        results.append((0.3, f"Expected {len(questions)} classifications, got {len(classifications)}"))
                        continue
                    
                    type_scores = []
                    type_counts = {}
                    
                    for cls in classifications:
                        info_type = cls.get('info_type', '').lower().strip()
                        base_reward = TYPE_REWARDS.get(info_type, 0.3)
                        
                        type_scores.append(base_reward)
                        type_counts[info_type] = type_counts.get(info_type, 0) + 1
                    
                    final_score = sum(type_scores) / len(questions) if questions else 0.0
                    
                    # Create explanation
                    type_summary = ", ".join([f"{t}:{c}" for t, c in sorted(type_counts.items())])
                    detailed_scores = " + ".join([f"{s:.3f}" for s in type_scores])
                    explanation = f"Scores: [{detailed_scores}] = {final_score:.3f}, Types: [{type_summary}]"
                    
                    results.append((final_score, explanation))
            else:
                results.append((0.0, "Skipped"))
        
        return results
        
    def score_batch(self, prompts: List[str], generations: List[str],
                   original_issues: List[Optional[str]]) -> List[Dict]:
        """Four-stage batch scoring with generation cache"""
        contexts = [extract_task_only(p) for p in prompts]
        all_questions = [self._parse_questions(gen) for gen in generations]
        
        # Stage 1: Redundancy
        stage1_results = self._stage1_batch(contexts, all_questions)
        
        # Filter for Stage 2
        stage2_inputs = []
        stage2_indices = []
        for i, (s1_score, _, _) in enumerate(stage1_results):
            if s1_score >= self.reward_config.stage1_threshold:
                stage2_inputs.append((generations[i], all_questions[i]))
                stage2_indices.append(i)
        
        # Stage 2: Novelty (compare to cache)
        if stage2_inputs:
            gen_batch, qs_batch = zip(*stage2_inputs)
            stage2_batch_results = self._stage2_novelty_batch(list(gen_batch), list(qs_batch))
        else:
            stage2_batch_results = []
        
        # Filter for Stage 3
        stage3_inputs = []
        stage3_indices = []
        s2_idx = 0
        for i in stage2_indices:
            s2_score, _ = stage2_batch_results[s2_idx]
            s2_idx += 1
            if s2_score >= self.reward_config.stage2_threshold:
                stage3_inputs.append((original_issues[i], all_questions[i]))
                stage3_indices.append(i)
        
        # Stage 3: Answerability
        if stage3_inputs:
            orig_batch, qs_batch = zip(*stage3_inputs)
            stage3_batch_results = self._stage3_answerability_batch(list(orig_batch), list(qs_batch))
        else:
            stage3_batch_results = []
        
        # Filter for Stage 4
        stage4_inputs = []
        stage4_indices = []
        s3_idx = 0
        for i in stage3_indices:
            s3_score, _ = stage3_batch_results[s3_idx]
            s3_idx += 1
            if s3_score >= self.reward_config.stage3_threshold:
                stage4_inputs.append((contexts[i], all_questions[i]))
                stage4_indices.append(i)
        
        # Stage 4: Utility + Specificity
        if stage4_inputs:
            ctx_batch, qs_batch = zip(*stage4_inputs)
            stage4_batch_results = self._stage4_utility_specificity_batch(list(ctx_batch), list(qs_batch))
        else:
            stage4_batch_results = []
        
        # Update cache with new generations
        for gen in generations:
            if gen and gen.strip() != "<think></think>":
                self.generation_cache.add(gen)
        
        # Assemble final results
        results = []
        s2_idx = 0
        s3_idx = 0
        s4_idx = 0
        
        for i, (s1_score, s1_reason, qa_pairs) in enumerate(stage1_results):
            questions = all_questions[i]
            
            if not questions:
                results.append({
                    'reward': 0.0,
                    'stage1_score': 0.0,
                    'stage2_score': 0.0,
                    'stage3_score': 0.0,
                    'stage4_score': 0.0,
                    'passed_stage': 0,
                    'num_questions': 0,
                    'reasoning': "No questions"
                })
                continue
            
            # Failed Stage 1
            if s1_score < self.reward_config.stage1_threshold:
                results.append({
                    'reward': 0.0,
                    'stage1_score': s1_score,
                    'stage2_score': 0.0,
                    'stage3_score': 0.0,
                    'stage4_score': 0.0,
                    'passed_stage': 1,
                    'num_questions': len(questions),
                    'reasoning': f"[S1] {s1_reason} [FILTERED]"
                })
                continue
            
            # Get Stage 2 result
            s2_score, s2_reason = stage2_batch_results[s2_idx]
            s2_idx += 1
            
            # Failed Stage 2
            if s2_score < self.reward_config.stage2_threshold:
                results.append({
                    'reward': s1_score * self.reward_config.stage1_weight * 0.5,
                    'stage1_score': s1_score,
                    'stage2_score': s2_score,
                    'stage3_score': 0.0,
                    'stage4_score': 0.0,
                    'passed_stage': 2,
                    'num_questions': len(questions),
                    'reasoning': f"[S1] {s1_reason} | [S2] {s2_reason} [FILTERED]"
                })
                continue
            
            # Get Stage 3 result
            s3_score, s3_reason = stage3_batch_results[s3_idx]
            s3_idx += 1
            
            # Failed Stage 3
            if s3_score < self.reward_config.stage3_threshold:
                s2_scaled = s2_score * np.sqrt(max(s1_score, 0.01))
                partial_reward = (s1_score * self.reward_config.stage1_weight +
                                s2_scaled * self.reward_config.stage2_weight) * 0.5
                
                results.append({
                    'reward': partial_reward,
                    'stage1_score': s1_score,
                    'stage2_score': s2_score,
                    'stage3_score': s3_score,
                    'stage4_score': 0.0,
                    's2_scaled': s2_scaled,
                    'passed_stage': 3,
                    'num_questions': len(questions),
                    'reasoning': f"[S1] {s1_reason} | [S2] {s2_reason} | [S3] {s3_reason} [FILTERED]"
                })
                continue
            
            # Get Stage 4 result
            s4_score, s4_reason = stage4_batch_results[s4_idx]
            s4_idx += 1
            
            # Calculate final reward (all stages passed)
            s2_scaled = s2_score 
            s3_scaled = s3_score 
            
            reward = (s1_score * self.reward_config.stage1_weight +
                     s2_score * self.reward_config.stage2_weight +
                     s3_score * self.reward_config.stage3_weight +
                     s4_score * self.reward_config.stage4_weight)
            
            results.append({
                'reward': reward,
                'stage1_score': s1_score,
                'stage2_score': s2_score,
                'stage3_score': s3_score,
                'stage4_score': s4_score,
                's2_scaled': s2_scaled,
                's3_scaled': s3_scaled,
                'passed_stage': 4,
                'num_questions': len(questions),
                'reasoning': f"[S1] {s1_reason} | [S2] {s2_reason} | [S3] {s3_reason} | [S4] {s4_reason}"
            })
        
        return results
    
    def get_stats(self) -> Dict:
        total = self.parse_stats['total_calls']
        return {
            'total_calls': self.call_count,
            'json_parse_success_rate': 1 - (self.parse_stats['json_failures'] / max(total, 1)),
            'empty_question_rate': self.parse_stats['empty_questions'] / max(self.call_count, 1),
            'answer_format_issues': self.parse_stats['answer_format_issues'],
            'cache_size': len(self.generation_cache.get_recent())
        }

# ============================================================================
# Optimized Reference Worker
# ============================================================================

@ray.remote(num_gpus=1, num_cpus=4)
class OptimizedReferenceWorker:
    """Reference model with batched log prob computation"""
    def __init__(self, config: Config):
        self.config = config
        
        logger.info("Loading reference model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        logger.info("✓ Reference ready")
    
    def compute_log_probs_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Batched log probability computation"""
        log_probs = []
        
        batch_size = 4
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_responses = responses[i:i+batch_size]
                
                full_texts = [format_prompt(p) + r for p, r in zip(batch_prompts, batch_responses)]
                prompt_texts = [format_prompt(p) for p in batch_prompts]
                
                full_inputs = self.tokenizer(
                    full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length + self.config.max_new_tokens
                ).to(self.model.device)
                
                prompt_inputs = self.tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.model.device)
                
                outputs = self.model(**full_inputs)
                
                for j in range(len(batch_prompts)):
                    prompt_mask = prompt_inputs['attention_mask'][j]
                    prompt_len = prompt_mask.sum().item()
                    full_mask = full_inputs['attention_mask'][j]
                    full_len = full_mask.sum().item()
                    response_len = full_len - prompt_len
                    
                    if response_len <= 0:
                        log_probs.append(-8.0)
                        continue
                    
                    logits = outputs.logits[j, prompt_len-1:full_len-1, :]
                    labels = full_inputs['input_ids'][j, prompt_len:full_len]
                    
                    if labels.numel() == 0:
                        log_probs.append(-8.0)
                    else:
                        lp = F.log_softmax(logits, dim=-1).gather(
                            1, labels.unsqueeze(1)
                        ).squeeze(1)
                        log_probs.append(
                            float(np.clip(torch.mean(lp).cpu().item(), -15.0, 0.0))
                        )
        
        return log_probs

# ============================================================================
# Optimized GRPO Actor
# ============================================================================

class OptimizedGRPOActor:
    """GRPO actor with batched generation"""
    
    def __init__(self, config: Config):
        self.config = config
        
        logger.info("Loading GRPO actor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        self.scheduler = None
        self.step = 0
        self.accum_step = 0
        
        logger.info("✓ GRPO Actor ready")
    
    def setup_scheduler(self, total_steps: int):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps,
            div_factor=10,
            final_div_factor=100
        )
        logger.info(f"✓ Scheduler: {total_steps} steps")
    
    def generate_samples_batch(self, prompts: List[str]) -> List[List[str]]:
        """Batched generation with KV caching"""
        all_samples = [[] for _ in prompts]
        
        with torch.no_grad():
            formatted = [format_prompt(p) for p in prompts]
            
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.model.device)
            
            eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>") or \
                     self.tokenizer.eos_token_id
            
            for sample_idx in range(self.config.num_samples_per_prompt):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.policy_temperature,
                    top_p=self.config.policy_top_p,
                    do_sample=True,
                    eos_token_id=eos_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=self.config.use_generation_cache
                )
                
                for i, output in enumerate(outputs):
                    response = self.tokenizer.decode(
                        output[inputs['input_ids'][i].shape[0]:],
                        skip_special_tokens=True
                    ).strip().replace("<|im_end|>", "").strip()
                    
                    all_samples[i].append(response if response else "<think></think>")
        
        return all_samples
    
    def compute_log_prob(self, prompt: str, response: str) -> torch.Tensor:
        """Compute log probability"""
        full = format_prompt(prompt) + response
        inputs = self.tokenizer(
            full, 
            return_tensors="pt", 
            truncation=True,
            max_length=self.config.max_length + self.config.max_new_tokens
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        prompt_len = self.tokenizer(
            format_prompt(prompt), 
            return_tensors="pt"
        )['input_ids'].shape[1]
        
        outputs = self.model(**inputs)
        logits = outputs.logits[0, prompt_len-1:-1, :]
        labels = inputs['input_ids'][0, prompt_len:]
        
        if labels.numel() == 0:
            return torch.tensor(-8.0, device=logits.device, requires_grad=True)
        
        lp = F.log_softmax(logits, dim=-1).gather(
            1, labels.unsqueeze(1)
        ).squeeze(1)
        return torch.clamp(torch.mean(lp), -15.0, 0.0)
    
    def update(self, batch: Dict) -> Optional[Dict]:
        """GRPO update"""
        losses = []
        advantages_all = []
        kl_divs = []
        
        for prompt, samples, rewards, ref_lps in zip(
            batch['prompts'], 
            batch['samples'],
            batch['rewards'],
            batch['ref_log_probs']
        ):
            policy_lps = torch.stack([
                self.compute_log_prob(prompt, sample)
                for sample in samples
            ])
            
            ref_lps_tensor = torch.tensor(
                ref_lps, 
                device=policy_lps.device, 
                dtype=policy_lps.dtype
            )
            kl_div = policy_lps - ref_lps_tensor
            kl_divs.append(kl_div.mean().item())
            
            rewards_tensor = torch.tensor(
                rewards, 
                device=policy_lps.device, 
                dtype=policy_lps.dtype
            )
            penalized_rewards = rewards_tensor - self.config.kl_coef * kl_div
            
            if self.config.baseline_type == "mean":
                baseline = penalized_rewards.mean()
            else:
                baseline = penalized_rewards.max()
            
            advantages = penalized_rewards - baseline
            advantages_all.extend(advantages.tolist())
            
            loss = -(policy_lps * advantages).mean()
            losses.append(loss)
        
        total_loss = torch.stack(losses).mean()
        total_loss = total_loss / self.config.gradient_accumulation_steps
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf loss!")
            return None
        
        total_loss.backward()
        self.accum_step += 1
        
        if self.accum_step >= self.config.gradient_accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.accum_step = 0
            
            return {
                'loss': total_loss.item() * self.config.gradient_accumulation_steps,
                'grad_norm': grad_norm.item(),
                'mean_advantage': np.mean(advantages_all),
                'mean_reward': np.mean([r for rs in batch['rewards'] for r in rs]),
                'mean_kl': np.mean(kl_divs),
                'lr': self.optimizer.param_groups[0]['lr'],
            }
        
        return None
    
    def save(self, step: int):
        path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        state = {
            'step': step,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(state, os.path.join(path, 'training_state.pt'))
        
        logger.info(f"✓ Saved checkpoint-{step}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint and resume training state"""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
    
        self.model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
        )
        self.model.train()
    
        self.tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True
        )
    
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        self.optimizer = torch.optim.AdamW(
        self.model.parameters(),
        lr=self.config.learning_rate,
        weight_decay=0.01
        )
    
        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
        
            self.step = training_state.get('step', 0)
        
            try:
                self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
                logger.info(f"✓ Restored optimizer state")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")
                logger.warning("Will continue with fresh optimizer state")
        
            if 'scheduler_state_dict' in training_state and training_state['scheduler_state_dict']:
                self._pending_scheduler_state = training_state['scheduler_state_dict']
        
            logger.info(f"✓ Loaded checkpoint from step {self.step}")
            logger.info(f"✓ Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        else:
            logger.warning("⚠ No training_state.pt found!")
            self.step = 0
        return self.step

# ============================================================================
# Dataset
# ============================================================================

def extract_questions_as_list(checklist_text):
    """Parse checklist string to list"""
    if pd.isna(checklist_text) or str(checklist_text).strip() == '':
        return []
    
    checklist_str = str(checklist_text).strip()
    numbered = re.findall(r'^\d+\.\s*(.+?)(?=\n\d+\.|$)', checklist_str, re.MULTILINE | re.DOTALL)
    if numbered:
        return [q.strip() for q in numbered if q.strip()]
    
    lines = checklist_str.split('\n')
    return [l.strip() for l in lines if l.strip() and len(l.strip()) > 10]

class Dataset_(Dataset):
    def __init__(self, jsonl_path: str, csv_path: str):
        self.data = []
        
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        if 'prompt' in item:
                            self.data.append(item)
                    except:
                        continue
        
        logger.info(f"Loaded {len(self.data)} examples from JSONL")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            checklist_to_original = {}
            for _, row in df.iterrows():
                if pd.notna(row.get('checklist')):
                    questions = extract_questions_as_list(row['checklist'])
                    if questions:
                        key = tuple(sorted(questions))
                        if pd.notna(row.get('original_text')):
                            checklist_to_original[key] = str(row['original_text']).strip()
            
            matched = 0
            for entry in self.data:
                gt = entry.get('ground_truth', [])
                if gt:
                    gt_key = tuple(sorted(gt))
                    if gt_key in checklist_to_original:
                        entry['original_issue'] = checklist_to_original[gt_key]
                        matched += 1
                    else:
                        entry['original_issue'] = None
                else:
                    entry['original_issue'] = None
            
            logger.info(f"Matched {matched} entries to original issues")
        else:
            logger.warning(f"CSV not found: {csv_path}")
            for entry in self.data:
                entry['original_issue'] = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# Utils
# ============================================================================

def find_latest(output_dir: str) -> Optional[str]:
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) 
                   if d.startswith('checkpoint-') and d != 'checkpoint-final']
    if not checkpoints:
        return None
    
    checkpoints.sort(
        key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else -1, 
        reverse=True
    )
    latest = os.path.join(output_dir, checkpoints[0])
    
    return latest if os.path.exists(
        os.path.join(latest, 'training_state.pt')
    ) else None

def cleanup(output_dir: str, keep: int = 5):
    if not os.path.exists(output_dir):
        return
    
    checkpoints = [d for d in os.listdir(output_dir) 
                   if d.startswith('checkpoint-') and d != 'checkpoint-final']
    
    if len(checkpoints) <= keep:
        return
    
    checkpoints.sort(
        key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else -1, 
        reverse=True
    )
    
    for checkpoint in checkpoints[keep:]:
        path = os.path.join(output_dir, checkpoint)
        try:
            import shutil
            shutil.rmtree(path)
            logger.info(f"🗑️ Deleted {checkpoint}")
        except Exception as e:
            logger.warning(f"Failed to delete {checkpoint}: {e}")

# ============================================================================
# Training
# ============================================================================

def train(config: Config, resume_from: Optional[str] = None):
    import wandb
    import time
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Ray init
    if not ray.is_initialized():
        ray.init(
            num_cpus=16, 
            num_gpus=4, 
            include_dashboard=False,
            _temp_dir=os.environ.get('RAY_TMPDIR', '/tmp/ray')
        )
    
    try:
        # Initialize workers
        logger.info("\nInitializing workers...")
        reference = OptimizedReferenceWorker.remote(config)
        reward_model = OptimizedRewardWorker.remote(config)
        actor = OptimizedGRPOActor(config)
        # Data
        dataset = Dataset_(config.train_data, config.csv_path)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: {
                'prompts': [x['prompt'] for x in b],
                'original_issues': [x.get('original_issue') for x in b]
            }
        )
        # Setup
        total_steps = (len(loader) * config.num_epochs) // \
                      config.gradient_accumulation_steps
        actor.setup_scheduler(total_steps)        
        # Resume
        start_step = 0
        start_epoch = 0
        start_batch_idx = 0

        if resume_from and os.path.exists(resume_from):
            logger.info("="*70)
            logger.info(f"RESUMING FROM: {resume_from}")
            logger.info("="*70)
            start_step = actor.load_checkpoint(resume_from)
            batches_per_epoch = len(loader)  # ✅ Use len(loader)!
            start_epoch = start_step // batches_per_epoch
            start_batch_idx = start_step % batches_per_epoch
            logger.info(f"Resuming from step {start_step}")
            logger.info(f"Epoch {start_epoch + 1}, batch {start_batch_idx}/{batches_per_epoch}")
            if hasattr(actor, '_pending_scheduler_state'):
                actor.scheduler.load_state_dict(actor._pending_scheduler_state)
                delattr(actor, '_pending_scheduler_state')
                logger.info("✓ Loaded scheduler state")        
        wandb.init(
            project="grpo_generation_cache", 
            name="4stage_novelty", 
            config=config.__dict__
        )
        
        step = start_step
        stage_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        # Timing
        step_times = []
        
        for epoch in range(config.num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
            logger.info(f"{'='*70}")
            
            for batch_idx, batch in enumerate(loader):
                # Skip batches if resuming in the middle of an epoch
                if epoch == start_epoch and batch_idx < start_batch_idx:
                    continue  # Skip already processed batches
                
                step_start = time.time()                
                # Generate N samples per prompt
                samples = actor.generate_samples_batch(batch['prompts'])
                gen_time = time.time() - step_start
                
                # Flatten for reward model
                all_prompts = []
                all_samples = []
                all_originals = []
                for prompt, prompt_samples, orig in zip(
                    batch['prompts'], samples, batch['original_issues']
                ):
                    all_prompts.extend([prompt] * len(prompt_samples))
                    all_samples.extend(prompt_samples)
                    all_originals.extend([orig] * len(prompt_samples))
                
                # Score with reward model (4 stages with cache)
                reward_start = time.time()
                reward_results = ray.get(
                    reward_model.score_batch.remote(
                        all_prompts, all_samples, all_originals
                    )
                )
                rewards_flat = [r['reward'] for r in reward_results]
                reward_time = time.time() - reward_start
                
                # Parse questions for diversity bonus
                all_questions = []
                for sample in all_samples:
                    questions = []
                    text = re.sub(r'<think>.*?</think>', '', sample, flags=re.DOTALL)
                    numbered = re.findall(r'^\d+\.\s*(.+?)(?=\n\d+\.|$)', text, re.MULTILINE | re.DOTALL)
                    if numbered:
                        questions = [q.strip() for q in numbered if q.strip() and len(q.strip()) > 10]
                    all_questions.append(questions)
                
                # Track stage statistics
                for result in reward_results:
                    stage_counts[result['passed_stage']] += 1
                
                # Reshape rewards
                rewards = []
                idx = 0
                for prompt_samples in samples:
                    prompt_rewards = rewards_flat[idx:idx+len(prompt_samples)]
                    rewards.append(prompt_rewards)
                    idx += len(prompt_samples)
                
                # Get reference log probs
                ref_start = time.time()
                ref_log_probs_flat = ray.get(
                    reference.compute_log_probs_batch.remote(all_prompts, all_samples)
                )
                ref_time = time.time() - ref_start
                
                # Reshape ref log probs
                ref_log_probs = []
                idx = 0
                for prompt_samples in samples:
                    prompt_ref_lps = ref_log_probs_flat[idx:idx+len(prompt_samples)]
                    ref_log_probs.append(prompt_ref_lps)
                    idx += len(prompt_samples)
                
                # Update policy
                update_start = time.time()
                metrics = actor.update({
                    'prompts': batch['prompts'],
                    'samples': samples,
                    'rewards': rewards,
                    'ref_log_probs': ref_log_probs
                })
                update_time = time.time() - update_start
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                if metrics:
                    if step % config.log_every == 0:
                        # Sample reward breakdown
                        sample_idx = np.random.randint(len(reward_results))
                        sample = reward_results[sample_idx]
                        
                        # Compute throughput
                        samples_per_sec = (config.batch_size * config.num_samples_per_prompt) / step_time
                        
                        
                        logger.info(
                            f"S{step} | Loss={metrics['loss']:.4f} | "
                            f"R={metrics['mean_reward']:.3f} | "
                            f"A={metrics['mean_advantage']:.3f} | "
                            f"KL={metrics['mean_kl']:.3f}"
                        )
                        logger.info(
                            f"      Time: {step_time:.2f}s "
                            f"(gen={gen_time:.2f}s, reward={reward_time:.2f}s, "
                            f"ref={ref_time:.2f}s, update={update_time:.2f}s) | "
                            f"Throughput: {samples_per_sec:.1f} samples/s"
                        )
                        
                        if sample['passed_stage'] == 4:
                            logger.info(
                                f"      Sample: "
                                f"s1={sample['stage1_score']:.2f} | "
                                f"s2={sample['stage2_score']:.2f}→{sample.get('s2_scaled', 0):.2f} | "
                                f"s3={sample['stage3_score']:.2f}→{sample.get('s3_scaled', 0):.2f} | "
                                f"s4={sample['stage4_score']:.2f} | "
                                f"stage={sample['passed_stage']}"
                            )
                        else:
                            logger.info(
                                f"      Sample: "
                                f"s1={sample['stage1_score']:.2f} | "
                                f"s2={sample['stage2_score']:.2f} | "
                                f"s3={sample['stage3_score']:.2f} | "
                                f"s4={sample['stage4_score']:.2f} | "
                                f"stage={sample['passed_stage']}"
                            )
                    
                    # Log to wandb
                    log_dict = {'step': step, **metrics}
                    
                    # Timing info
                    log_dict['time/step'] = step_time
                    log_dict['time/generation'] = gen_time
                    log_dict['time/reward'] = reward_time
                    log_dict['time/reference'] = ref_time
                    log_dict['time/update'] = update_time
                    log_dict['throughput/samples_per_sec'] = (config.batch_size * config.num_samples_per_prompt) / step_time
                    
                    # Reward model stats
                    reward_stats = ray.get(reward_model.get_stats.remote())
                    log_dict['reward_model/json_success_rate'] = reward_stats.get('json_parse_success_rate', 1.0)
                    log_dict['reward_model/empty_question_rate'] = reward_stats.get('empty_question_rate', 0.0)
                    log_dict['reward_model/answer_format_issues'] = reward_stats.get('answer_format_issues', 0)
                    log_dict['reward_model/total_calls'] = reward_stats.get('total_calls', 0)
                    log_dict['reward_model/cache_size'] = reward_stats.get('cache_size', 0)
                    
                    # Stage statistics
                    total = sum(stage_counts.values())
                    if total > 0:
                        log_dict['stage/filtered_0'] = stage_counts[0] / total
                        log_dict['stage/filtered_1'] = stage_counts[1] / total
                        log_dict['stage/filtered_2'] = stage_counts[2] / total
                        log_dict['stage/filtered_3'] = stage_counts[3] / total
                        log_dict['stage/passed_4'] = stage_counts[4] / total
                    
                    # Dimension scores
                    stage1_flat = [r['stage1_score'] for r in reward_results]
                    stage2_flat = [r['stage2_score'] for r in reward_results]
                    stage3_flat = [r['stage3_score'] for r in reward_results]
                    stage4_flat = [r['stage4_score'] for r in reward_results]
                    
                    log_dict['reward_dist/mean'] = np.mean(rewards_flat)
                    log_dict['reward_dist/std'] = np.std(rewards_flat)
                    log_dict['reward_dist/min'] = np.min(rewards_flat)
                    log_dict['reward_dist/max'] = np.max(rewards_flat)
                    log_dict['reward_dist/range'] = np.max(rewards_flat) - np.min(rewards_flat)
                    
                    log_dict['stage_scores/stage1_mean'] = np.mean(stage1_flat)
                    log_dict['stage_scores/stage2_mean'] = np.mean(stage2_flat)
                    log_dict['stage_scores/stage3_mean'] = np.mean(stage3_flat)
                    log_dict['stage_scores/stage4_mean'] = np.mean(stage4_flat)
                    
                    wandb.log(log_dict)
                    
                    if step % config.save_steps == 0 and step > 0:
                        actor.save(step)
                        cleanup(config.output_dir, config.keep_last_checkpoints)
                    
                    step += 1
                    
                    # Memory cleanup
                    if step % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
        
        # Save final
        final = os.path.join(config.output_dir, "checkpoint-final")
        os.makedirs(final, exist_ok=True)
        actor.model.save_pretrained(final)
        actor.tokenizer.save_pretrained(final)
        
        # Final stats
        reward_stats = ray.get(reward_model.get_stats.remote())
        avg_step_time = np.mean(step_times)
        avg_throughput = (config.batch_size * config.num_samples_per_prompt) / avg_step_time
        
        wandb.finish()
        
    finally:
        if ray.is_initialized():
            ray.shutdown()

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", 
                       default="")
    parser.add_argument("--train_data", 
                       default="")
    parser.add_argument("--csv", dest="csv_path",
                       default="")
    parser.add_argument("--output_dir", 
                       default="")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    
    # Reward model
    parser.add_argument("--reward_model", 
                       default="Qwen/Qwen3-32B")
    parser.add_argument("--reward_batch_size", type=int, default=16)
    parser.add_argument("--cache_size", type=int, default=2,
                       help="Number of recent generations to cache for novelty check")
    parser.add_argument("--stage1_threshold", type=float, default=0.3)
    parser.add_argument("--stage2_threshold", type=float, default=0.3)
    parser.add_argument("--stage3_threshold", type=float, default=0)
    
    # Diversity
    parser.add_argument("--diversity_bonus_weight", type=float, default=0.0)
    
    # Training
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None)
    
    args = parser.parse_args()
    
    reward_config = RewardConfig(
        model_name=args.reward_model,
        reward_batch_size=args.reward_batch_size,
        cache_size=args.cache_size,
        stage1_threshold=args.stage1_threshold,
        stage2_threshold=args.stage2_threshold,
        stage3_threshold=args.stage3_threshold
    )
    
    config = Config(
        base_model=args.base_model,
        train_data=args.train_data,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples_per_prompt=args.num_samples,
        kl_coef=args.kl_coef,
        reward_config=reward_config,
        diversity_bonus_weight=args.diversity_bonus_weight
    )
    
    resume = None
    if args.resume_from and os.path.exists(args.resume_from):
        resume = args.resume_from
    elif args.auto_resume:
        resume = find_latest(config.output_dir)
        if resume:
            logger.info(f"Auto-resuming: {resume}")
    
    train(config, resume)
