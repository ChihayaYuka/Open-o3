import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import PPOConfig, PPOTrainer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 参数配置
model_name = "your-model" # 被知乎搞怕了不设置默认模型了
sentence_model = SentenceTransformer('your-model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据预处理
with open("train_dataset.json") as f:
    dataset = json.load(f)

# 3. 模型加载
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bifloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4. 奖励计算器
class RewardCalculator:
    def __init__(self):
        self.format_penalty = -5.0  # 对不遵守格式的回答一个负奖励
        self.logic_word_bonus = 0.3  # Logic word 的奖励
        
    def calculate_reward(self, generated_text, reference_steps):
        if not self._validate_format(generated_text):
            return self.format_penalty

        generated_steps = self._extract_steps(generated_text)
        sim_score = self._calculate_similarity(generated_steps, reference_steps)
        logic_bonus = self._count_logic_words(' '.join(generated_steps)) * self.logic_word_bonus
        
        return sim_score + logic_bonus
    
    def _validate_format(self, text):
        return re.search(r"<Thought>(.*?)</Thought>", text, re.DOTALL) is not None
    
    def _extract_steps(self, text):
        match = re.search(r"<Thought>(.*?)</Thought>", text, re.DOTALL)
        if not match: return []
        steps_text = match.group(1)
        return re.findall(r"<Step>(.*?)</Step>", steps_text, re.DOTALL)
    
    def _calculate_similarity(self, gen_steps, ref_steps):
        if not gen_steps: return 0.0
        min_len = min(len(gen_steps), len(ref_steps))
        gen_emb = sentence_model.encode(gen_steps[:min_len])
        ref_emb = sentence_model.encode(ref_steps[:min_len])
        
        return cosine_similarity(gen_emb, ref_emb).diagonal().mean().item()
    
    def _count_logic_words(self, text):
        return len(re.findall(r"\b(wait|first|then|therefore|because|so)\b", text.lower()))

# 5. 训练配置
ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=1.4e-5,
    steps=10000,
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
)

def format_prompt(question):
    return f"""深思熟虑的解决这个问题并且按照以下格式回答，Let's think step by step.
问题：{question}
<Thought>
{{response}}
</Thought>"""

training_data = [
    {
        "query": format_prompt(item["question"]),
        "reference_steps": item["steps"]
    } for item in dataset
]

reward_calculator = RewardCalculator()
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    model_ref,
    tokenizer,
    dataset=training_data,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 417,
}

for epoch in range(ppo_config.steps):
    for data in ppo_trainer.dataloader:
        query = data["query"]
        reference_steps = data["reference_steps"]
        input_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
        response_ids = ppo_trainer.generate(
            input_ids, 
            return_prompt=False,
            **generation_kwargs
        )
        response = tokenizer.decode(response_ids[0])
        reward = reward_calculator.calculate_reward(response, reference_steps)
        stats = ppo_trainer.step([input_ids], [response_ids], [torch.tensor(reward).to(device)])
        if epoch % 10 == 0:
            print(f"Step {epoch} | Reward: {reward:.2f} | Response: {response[:200]}")

model.save_pretrained("rl_trained_model")
