import os
import sys
import time
import math
import json
import logging
import datetime
from typing import List, Tuple, Optional, Dict, Any

import torch
import numpy as np
from config import config
from scipy.stats import entropy, kstest
from scipy.linalg import hadamard
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification

logging.basicConfig(
   level=logging.DEBUG,
   format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("o3")  

class o3:
   def __init__(self,
                system_prompt: Optional[str] = None,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                enable_tda: bool = False,
                tda_persistence_threshold: float = 0.5,
                save_results: bool = False,
                result_dir: str = "./results") -> None:
       """
       初始化 o3 实例。

       Args:
           system_prompt: 系统提示，用于构造推理上下文（例如：λ演算模板）。
           device: 运行设备，"cuda" 或 "cpu"。
           enable_tda: 是否启用拓扑数据分析 (TDA)。
           tda_persistence_threshold: TDA中持久性阈值，用于过滤噪声簇。
           save_results: 是否将生成结果保存到文件。
           result_dir: 结果保存的目录。
       """
       self.system_prompt = system_prompt
       self.token_counter = 0
       self.device = device
       self.save_results = save_results
       self.result_dir = result_dir

       # 信息论参数
       self.entropy_amplifier = 0.324
       self.kl_threshold = 0.707

       # TDA 参数
       self.enable_tda = enable_tda
       self.tda_persistence_threshold = tda_persistence_threshold
       if self.enable_tda:
           try:
               import gudhi
               self.gudhi = gudhi
               logger.info("TDA enabled")  
           except ImportError:
               logger.warning("gudhi library not installed, disabling TDA. Please install: pip install gudhi")  
               self.enable_tda = False

       # 加载基础语言模型和验证器
       try:
           m_config = config()
           self.model = AutoModelForCausalLM.from_pretrained(m_config.get_baseline_name()).to(self.device)
           self.tokenizer = AutoTokenizer.from_pretrained(m_config.get_baseline_name())
           self.validator = BertForSequenceClassification.from_pretrained('bert-validator').to(self.device)
           logger.info("Successfully loaded pre-trained models, running on device: %s", self.device)  
       except Exception as e:
           logger.error("Failed to load pre-trained models: %s", e)  
           sys.exit(1)

       # 用于性能统计的字典
       self.run_stats: Dict[str, float] = {}
       # 用于记录各阶段生成的数学指标
       self.math_metrics: Dict[str, Any] = {}

   def compute_shannon_entropy(self, text: str) -> float:
       """
       计算文本的 Shannon 熵，反映文本信息的不确定性。

       Args:
           text: 输入字符串。

       Returns:
           Shannon 熵值。
       """
       if not text:
           return 0.0
       freq = {ch: text.count(ch) for ch in set(text)}
       total = len(text)
       probs = [count / total for count in freq.values()]
       shannon_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
       logger.debug("Shannon Entropy: %.4f", shannon_entropy)  
       return shannon_entropy

   def compute_joint_entropy(self, text1: str, text2: str) -> float:
       """
       计算两个文本的联合熵，反映二者共同信息的不确定性。

       Args:
           text1: 第一个文本。
           text2: 第二个文本。

       Returns:
           联合熵值。
       """
       joint_text = text1 + text2
       return self.compute_shannon_entropy(joint_text)

   def compute_mutual_information(self, text1: str, text2: str) -> float:
       """
       计算两个文本之间的互信息，用于衡量它们共享的信息量。

       Args:
           text1: 第一个文本。
           text2: 第二个文本。

       Returns:
           互信息值。
       """
       h1 = self.compute_shannon_entropy(text1)
       h2 = self.compute_shannon_entropy(text2)
       joint_h = self.compute_joint_entropy(text1, text2)
       mi = h1 + h2 - joint_h
       logger.debug("Mutual Information: %.4f", mi)  
       return mi

   def probability_distribution(self, text: str) -> np.ndarray:
       """
       计算文本中各字符的概率分布（归一化直方图）。

       Args:
           text: 输入文本。

       Returns:
           概率分布数组。
       """
       if not text:
           return np.array([])
       total = len(text)
       freq = {ch: text.count(ch) for ch in set(text)}
       dist = np.array([freq[ch] / total for ch in sorted(freq.keys())])
       logger.debug("Character probability distribution: %s", dist)  
       return dist

   def compute_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
       """
       计算数据均值的置信区间（基于 t 分布）。

       Args:
           data: 数值列表。
           confidence: 置信水平，默认为 0.95。

       Returns:
           (下界, 上界)
       """
       if not data:
           return (0.0, 0.0)
       import scipy.stats as st
       a = 1.0 * np.array(data)
       n = len(a)
       mean, sem = np.mean(a), st.sem(a)
       h = sem * st.t.ppf((1 + confidence) / 2., n-1)
       logger.debug("Confidence Interval: [%.4f, %.4f]", mean - h, mean + h)  
       return mean - h, mean + h

   def perform_statistical_tests(self, sample: List[float]) -> Dict[str, float]:
       """
       对样本数据执行基本统计检验，例如正态性检验和均值检验。

       Args:
           sample: 数值列表样本。

       Returns:
           包含各统计指标的字典。
       """
       from scipy import stats
       if not sample:
           return {}
       normality_p = stats.shapiro(sample)[1]
       t_stat, t_p = stats.ttest_1samp(sample, popmean=np.mean(sample))
       stats_result = {"shapiro_p": normality_p, "ttest_stat": t_stat, "ttest_p": t_p}
       logger.debug("Statistical test results: %s", stats_result)  
       return stats_result

   def _complexity_scaling(self, text: str) -> int:
       """
       根据文本复杂度估计生成答案的数量 n ∈ [25, 4170]。
       利用 Sardina-Pasteur 定律，将文本的相对字符熵与 Hadamard 矩阵计算结果结合，映射为生成数量。

       Args:
           text: 输入提示文本。

       Returns:
           生成答案数量 n。
       """
       text_length = len(text)
       if text_length == 0:
           logger.warning("Input text is empty, returning minimum complexity 25")  
           return 25

       char_set = set(text)
       counts = np.array([text.count(c) for c in char_set], dtype=np.float64)
       prob = counts / text_length
       rel_entropy = entropy(prob) / (math.log2(len(prob) + 1e-9) + 1e-9)
       self.math_metrics["relative_entropy"] = rel_entropy
       logger.debug("Calculated relative character entropy: %.4f", rel_entropy)  

       rows = text_length % 64 or 1
       try:
           latin_matrix = hadamard(64)[:rows, :]
       except Exception as e:
           logger.error("Hadamard matrix generation failed: %s", e)  
           latin_matrix = np.ones((rows, 64))

       text_bytes = np.frombuffer(text.encode(), dtype=np.uint8)
       cov_matrix = np.cov(latin_matrix * text_bytes[:latin_matrix.shape[0]])
       trace_val = np.trace(cov_matrix)
       complexity = int(trace_val * rel_entropy * self.entropy_amplifier)
       n = int(min(4170, max(25, complexity)))
       self.math_metrics["complexity_value"] = complexity
       logger.info("Text complexity mapped to n = %d", n)  
       return n

   def _latin_hypercube_sampling(self, n: int) -> torch.Tensor:
       """
       利用拉丁超立方采样生成参数矩阵。参数空间为：
           - temperature: [0.1, 1.5]
           - top_p: [0.3, 0.99]

       Args:
           n: 需要生成的参数对数量。

       Returns:
           参数矩阵，每行为 (temperature, top_p)。
       """
       dim = int(np.sqrt(n))
       if dim < 1:
           dim = 1
       temp = torch.linspace(0.1, 1.5, steps=dim, device=self.device)
       top_p = torch.linspace(0.3, 0.99, steps=dim, device=self.device)
       grid = torch.stack(torch.meshgrid(temp, top_p, indexing='ij'), dim=-1).reshape(-1, 2)
       indices = torch.randperm(grid.shape[0], device=self.device)
       sampled_params = grid[indices[:n]]
       logger.info("Generated %d parameter sets", sampled_params.shape[0])  
       return sampled_params

   def _ks_test_validation(self, prompt: str, answer: str) -> bool:
       """
       利用 Kolmogorov-Smirnov 检验评估生成答案的 token 分布与均匀分布的差异。
       当 p 值大于 0.05 时认为答案具有足够的随机性和多样性。

       Args:
           prompt: 输入提示（未使用，仅扩展接口）。
           answer: 生成的答案文本。

       Returns:
           True 当 KS 检验通过；False 否则。
       """
       a_tokens = self.tokenizer(answer)['input_ids']
       vocab_size = self.tokenizer.vocab_size
       a_hist = np.histogram(a_tokens, bins=np.arange(vocab_size + 1))[0]
       a_dist = a_hist / a_hist.sum()
       uniform_dist = np.ones(vocab_size) / vocab_size
       ks_statistic, p_value = kstest(a_dist, uniform_dist)
       logger.debug("KS Test: statistic=%.4f, p_value=%.4f", ks_statistic, p_value)  
       return p_value > 0.05

   def _validate_with_kl(self, prompt: str, answer: str) -> bool:
       """
       基于 Kullback-Leibler 散度对生成答案进行自我审查，
       检查答案 token 分布与提示 token 分布之间的差异是否低于设定阈值。

       Args:
           prompt: 原始输入提示文本。
           answer: 生成的答案文本。

       Returns:
           True 当 KL 散度低于阈值；False 否则。
       """
       p_tokens = self.tokenizer(prompt)['input_ids']
       a_tokens = self.tokenizer(answer)['input_ids']
       vocab_size = self.tokenizer.vocab_size
       p_hist = np.histogram(p_tokens, bins=np.arange(vocab_size + 1))[0] + 1e-9
       a_hist = np.histogram(a_tokens, bins=np.arange(vocab_size + 1))[0] + 1e-9
       p_dist = p_hist / p_hist.sum()
       a_dist = a_hist / a_hist.sum()
       kl_div = entropy(p_dist, qk=a_dist)
       logger.debug("KL Divergence: %.4f, Threshold: %.4f", kl_div, self.kl_threshold)  
       return kl_div < self.kl_threshold

   def _generate_with_retry(self, prompt: str, params: torch.Tensor, max_retries: int = 3) -> List[str]:
       """
       使用马尔可夫链蒙特卡洛（MCMC）机制生成答案，每组参数最多重试 max_retries 次，
       结合 KS 检验和 KL 散度确保生成的答案具有高质量。

       Args:
           prompt: 输入提示文本。
           params: 参数对张量，每行表示 (temperature, top_p)。
           max_retries: 每组参数的最大重试次数。

       Returns:
           有效答案列表。
       """
       valid_answers: List[str] = []
       for idx, (t, p) in enumerate(params):
           current_prompt = prompt
           for attempt in range(max_retries):
               inputs = self.tokenizer(current_prompt, return_tensors='pt').to(self.device)
               try:
                   outputs = self.model.generate(
                       **inputs,
                       do_sample=True,
                       temperature=t.item(),
                       top_p=p.item(),
                       max_new_tokens=256
                   )
               except Exception as e:
                   logger.error("Error during generation: %s", e)  
                   continue

               self.token_counter += outputs.numel()
               answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
               logger.debug("Parameter set %d, Attempt %d, Answer: %s", idx, attempt + 1, answer.strip())  

               if self._ks_test_validation(current_prompt, answer):
                   if self._validate_with_kl(current_prompt, answer):
                       logger.debug("Parameter set %d passed validation on attempt %d", idx, attempt + 1)  
                       valid_answers.append(answer)
                       break
                   else:
                       logger.warning("Parameter set %d, Attempt %d: KL validation failed", idx, attempt + 1)  
               else:
                   logger.warning("Parameter set %d, Attempt %d: KS test failed", idx, attempt + 1)  
               current_prompt = f"Previous attempt: {answer}\nImprove: "
           else:
               logger.warning("Parameter set %d reached maximum retries", idx)  
       return valid_answers

   def _spectral_borda_voting(self, answers: List[str]) -> str:
       """
       使用谱聚类和 Borda计数对候选答案进行投票，
       计算答案的语义嵌入，构造归一化余弦相似度矩阵，然后通过聚类和排序选出最优答案。

       Args:
           answers: 候选答案列表。

       Returns:
           最优答案字符串。
       """
       inputs = self.tokenizer(answers, padding=True, truncation=True, return_tensors='pt').to(self.device)
       with torch.no_grad():
           embeds = self.validator.bert(**inputs).last_hidden_state.mean(dim=1)
       norms = embeds.norm(dim=1, keepdim=True).clamp(min=1e-9)
       cosine_sim = (embeds @ embeds.T) / (norms * norms.T)
       num_clusters = min(len(answers), 2)
       if num_clusters < 2:
           logger.warning("Insufficient candidate answers, returning the first answer directly")  
           return answers[0]

       best_n_clusters = num_clusters
       best_silhouette = -1
       for n_clusters in range(2, min(len(answers), 5)):
           try:
               spectral = SpectralClustering(n_clusters=n_clusters,
                                             affinity='precomputed',
                                             assign_labels='discretize',
                                             random_state=42)
               labels = spectral.fit_predict(cosine_sim.cpu().numpy())
               silhouette = silhouette_score(cosine_sim.cpu().numpy(), labels)
               if silhouette > best_silhouette:
                   best_silhouette = silhouette
                   best_n_clusters = n_clusters
           except Exception as e:
               logger.warning("Error during spectral clustering with n=%d: %s", n_clusters, e)  
       logger.info("Spectral clustering selected the best number of clusters: %d", best_n_clusters)  
       spectral = SpectralClustering(n_clusters=best_n_clusters,
                                     affinity='precomputed',
                                     assign_labels='discretize',
                                     random_state=42)
       labels = spectral.fit_predict(cosine_sim.cpu().numpy())

       # 若启用 TDA，则对聚类结果进行细化
       if self.enable_tda:
           labels = self._tda_cluster_refinement(embeds.cpu().numpy(), labels)

       scores = np.zeros(len(answers))
       for cluster in range(best_n_clusters):
           indices = np.where(labels == cluster)[0]
           if len(indices) == 0:
               continue
           sub_sim = cosine_sim[indices][:, indices]
           borda_scores = sub_sim.sum(dim=1).argsort(descending=True)
           for rank, idx in enumerate(borda_scores):
               scores[indices[idx]] += (len(indices) - rank)
       best_idx = int(np.argmax(scores))
       logger.info("Borda count selected the best answer index: %d", best_idx)  
       return answers[best_idx]

   def _tda_cluster_refinement(self, embeddings: np.ndarray, initial_labels: np.ndarray) -> np.ndarray:
       """
       利用拓扑数据分析 (TDA) 对谱聚类结果进行细化，
       通过计算 Vietoris-Rips 复形与持久同调，过滤噪声簇并重新标记。

       Args:
           embeddings: 答案的语义嵌入数组。
           initial_labels: 初始聚类标签。

       Returns:
           优化后的聚类标签数组。
       """
       logger.info("Starting TDA cluster refinement")  
       point_cloud = embeddings
       rips_complex = self.gudhi.RipsComplex(points=point_cloud, max_edge_length=2.0)
       simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
       persistence = simplex_tree.persistence()
       barcode = simplex_tree.persistence_intervals_in_dimension(0)
       significant_intervals = [interval for interval in barcode if (interval[1] - interval[0]) > self.tda_persistence_threshold]
       noise_clusters = set()
       for interval in significant_intervals:
           for i, label in enumerate(initial_labels):
               norm_val = np.linalg.norm(point_cloud[i])
               if interval[0] <= norm_val <= interval[1]:
                   noise_clusters.add(label)
       refined_labels = initial_labels.copy()
       new_label = initial_labels.max() + 1
       for i, label in enumerate(initial_labels):
           if label in noise_clusters:
               refined_labels[i] = new_label
               new_label += 1
       logger.info("TDA refinement completed, number of noise clusters: %d", len(noise_clusters))  
       return refined_labels


   def _plot_token_histogram(self, text: str, title: str = "Token Distribution") -> None:
       """
       绘制输入文本的 token 分布直方图，并在需要时保存图片到结果目录。

       Args:
           text: 输入文本。
           title: 图表标题。
       """
       tokens = self.tokenizer(text)['input_ids']
       vocab_size = self.tokenizer.vocab_size
       hist, bins = np.histogram(tokens, bins=np.arange(vocab_size + 1))
       plt.figure(figsize=(10, 6))
       plt.bar(bins[:-1], hist, width=1.0, edgecolor='black')
       plt.title(title)
       plt.xlabel("Token ID")
       plt.ylabel("Frequency")
       plt.tight_layout()
       if self.save_results:
           if not os.path.exists(self.result_dir):
               os.makedirs(self.result_dir)
           filename = os.path.join(self.result_dir, f"token_hist_{int(time.time())}.png")
           plt.savefig(filename)
           logger.info("Saved token distribution histogram: %s", filename)  
       else:
           plt.show()
       plt.close()

   def _save_results(self, results: Dict[str, Any]) -> None:
       """
       将结果以 JSON 格式保存到文件中。

       Args:
           results: 包含结果信息的字典。
       """
       if not os.path.exists(self.result_dir):
           os.makedirs(self.result_dir)
       filename = os.path.join(self.result_dir, f"reasoner_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
       try:
           with open(filename, "w", encoding="utf-8") as f:
               json.dump(results, f, ensure_ascii=False, indent=4)
           logger.info("Results saved to: %s", filename)  
       except Exception as e:
           logger.error("Failed to save results: %s", e)  

   def hyperparameter_search(self, prompt: str, param_grid: Dict[str, List[float]], num_trials: int = 10) -> Tuple[float, float]:
       """
       利用随机搜索对 (temperature, top_p) 参数组合进行超参数优化，
       以期提高生成答案的质量。

       Args:
           prompt: 输入提示文本。
           param_grid: 包含 'temperature' 和 'top_p' 的取值列表字典。
           num_trials: 试验次数。

       Returns:
           最佳 (temperature, top_p) 参数组合。
       """
       best_score = -1
       best_params = (0.0, 0.0)
       for i in range(num_trials):
           temp = np.random.choice(param_grid.get("temperature", [0.5, 1.0]))
           top_p = np.random.choice(param_grid.get("top_p", [0.8, 0.9]))
           params_tensor = torch.tensor([[temp, top_p]], device=self.device)
           answers = self._generate_with_retry(prompt, params_tensor, max_retries=2)
           if not answers:
               continue
           score = sum(len(ans) for ans in answers) / len(answers)
           if score > best_score:
               best_score = score
               best_params = (temp, top_p)
           logger.debug("Trial %d: temp=%.2f, top_p=%.2f, score=%.4f", i + 1, temp, top_p, score)  
       logger.info("Hyperparameter search results: Best temp=%.2f, top_p=%.2f", best_params[0], best_params[1])  
       return best_params


   def solve(self, query: str, batch_size: int = 8, plot_hist: bool = False) -> Tuple[str, int]:
       """
       Args:
           query: 用户查询（例如：证明命题）。
           batch_size: 每批参数样本数。
           plot_hist: 是否绘制 token 直方图。

       Returns:
           (最优答案字符串, 累计生成的 token 数)。
       """
       full_prompt = f"{self.system_prompt}\n{query}" if self.system_prompt else query
       logger.info("Constructed full prompt: %s", full_prompt)  
       if plot_hist:
           self._plot_token_histogram(full_prompt, title="Input Prompt Token Distribution")
       n = self._complexity_scaling(full_prompt)
       param_space = self._latin_hypercube_sampling(n)
       answers: List[str] = []
       total_params = param_space.shape[0]
       start_time = time.time()

       for i in range(0, total_params, batch_size):
           batch_params = param_space[i:i + batch_size]
           new_answers = self._generate_with_retry(full_prompt, batch_params)
           answers.extend(new_answers)
           logger.info("Current number of generated answers: %d", len(answers))  
           if len(answers) >= 0.25 * n and len(set(answers)) < 5:
               logger.info("Early stopping: Sufficient answers generated with low diversity")  
               n = min(n, 2 * len(answers))
               break

       end_time = time.time()
       self.run_stats["generation_time_sec"] = end_time - start_time
       logger.info("Answer generation time: %.2f seconds", self.run_stats["generation_time_sec"])  

       if not answers:
           logger.warning("No valid answers generated, retrying one last time")  
           answers = self._generate_with_retry(full_prompt, param_space[:1])
           if not answers:
               logger.error("Retry failed, returning empty answer")  
               return "", self.token_counter

       best_answer = self._spectral_borda_voting(answers)
       logger.info("Final selected answer: %s", best_answer.strip())  

       # 保存并记录结果
       results = {
           "query": query,
           "system_prompt": self.system_prompt,
           "best_answer": best_answer,
           "token_usage": self.token_counter,
           "generation_time_sec": self.run_stats.get("generation_time_sec", None),
           "math_metrics": self.math_metrics,
           "timestamp": datetime.datetime.now().isoformat()
       }
       if self.save_results:
           self._save_results(results)
       return best_answer, self.token_counter

   def predict(self, query: str, **kwargs) -> Dict[str, Any]:
       """
       LangChain 调用接口，输入查询，返回包含答案和详细统计信息的字典。

       Args:
           query: 用户查询字符串。
           kwargs: 可选参数（如 batch_size, plot_hist 等）。

       Returns:
           包含答案、token使用情况、运行统计和数学指标的字典。
       """
       best_answer, token_usage = self.solve(query, **kwargs)
       output = {
           "answer": best_answer,
           "token_usage": token_usage,
           "run_stats": self.run_stats,
           "math_metrics": self.math_metrics,
           "timestamp": datetime.datetime.now().isoformat()
       }
       logger.info("predict call completed")  
       return output

   def detailed_report(self) -> None:
       """
       输出详细报告，包括生成参数、运行统计信息、数学指标和超参数建议。
       """
       report = (
           "===== o3 Detailed Report =====\n"  # Changed report title
           f"System Prompt: {self.system_prompt}\n"
           f"Device: {self.device}\n"
           f"Cumulative Token Count: {self.token_counter}\n"
           f"Generation Time (seconds): {self.run_stats.get('generation_time_sec', 'N/A')}\n"
           "--------------------------------------------\n"
           "Mathematical Metrics:\n"
           f"    Relative Character Entropy: {self.math_metrics.get('relative_entropy', 'N/A')}\n"
           f"    Complexity Value: {self.math_metrics.get('complexity_value', 'N/A')}\n"
           "--------------------------------------------\n"
           "Generation results will be saved in the results directory (if saving is enabled).\n"  
           "============================================\n"
       )
       logger.info(report)
       print(report)

   def run_example(self) -> None:
       """
       示例方法：展示如何调用该类进行推理，并输出最终结果与详细报告。
       """
       example_query = "证明不存在最大的素数"
       logger.info("Running example inference, question: %s", example_query)  
       output = self.predict(example_query, batch_size=8, plot_hist=True)
       print("Example inference result:")
       print(json.dumps(output, indent=4, ensure_ascii=False))
       self.detailed_report()

if __name__ == "__main__":
   system_prompt = "您是由流明智能开发的大型推理模型 Open-o3."
   reasoner = o3(system_prompt=system_prompt, enable_tda=True, save_results=True)   
   reasoner.run_example()
