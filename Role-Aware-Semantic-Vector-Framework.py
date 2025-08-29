import os
import json
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random
import time
from gensim.models import KeyedVectors
import gensim.downloader as api

# --- 效能監控裝飾器 ---
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[計時] {func.__name__} 執行時間: {time.time() - start:.2f} 秒")
        return result
    return wrapper

# --- 步驟 1: 核心的「角色感知詞向量」產生器 ---
class RoleAwareEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"\n[RoleAware] 正在載入 Sentence Transformer 基底模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.pole_vectors = {}
        self.cache = {}
        print("[RoleAware] 基底模型準備完成。")

    def setup_poles(self, all_pole_words):
        """設定極點詞向量"""
        self.pole_vectors = {}
        for dim, roles in all_pole_words.items():
            self.pole_vectors[dim] = {}
            for role, poles in roles.items():
                if poles.get("pos") and poles.get("neg"):
                    self.pole_vectors[dim][role] = {
                        "pos": self.model.encode(poles["pos"]),
                        "neg": self.model.encode(poles["neg"])
                    }
        print("[RoleAware] 所有角色的極點詞向量已設定完成。")

    def _calculate_score(self, target_vector, pos_vectors, neg_vectors):
        """計算單一向度分數"""
        if len(pos_vectors) == 0 or len(neg_vectors) == 0: 
            return 0.0
        pos_sim = cosine_similarity(target_vector, pos_vectors).mean()
        neg_sim = cosine_similarity(target_vector, neg_vectors).mean()
        raw_score = pos_sim - neg_sim
        return float(np.tanh(raw_score * 2.5))

    def create_dual_embedding(self, word: str):
        """創建雙重語義向量，使用快取避免重複計算"""
        if word in self.cache:
            return self.cache[word]
            
        try:
            target_vector = self.model.encode([word])
            head_embedding, modifier_embedding = {}, {}
            
            for dim_name, roles in self.pole_vectors.items():
                # 清理維度名稱
                clean_dim_name = dim_name.split('. ')[1] if '. ' in dim_name else dim_name
                
                # 計算主體向量
                head_poles = roles.get('head', {})
                head_score = self._calculate_score(
                    target_vector, 
                    head_poles.get('pos', []), 
                    head_poles.get('neg', [])
                )
                head_embedding[clean_dim_name] = round(head_score, 3)
                
                # 計算修飾向量
                modifier_poles = roles.get('modifier', {})
                modifier_score = self._calculate_score(
                    target_vector, 
                    modifier_poles.get('pos', []), 
                    modifier_poles.get('neg', [])
                )
                modifier_embedding[clean_dim_name] = round(modifier_score, 3)
            
            # 快取結果
            self.cache[word] = (head_embedding, modifier_embedding)
            return head_embedding, modifier_embedding
            
        except Exception as e:
            print(f"[警告] 無法為詞彙 '{word}' 生成向量: {e}")
            # 返回零向量
            zero_embedding = {dim.split('. ')[1] if '. ' in dim else dim: 0.0 
                            for dim in self.pole_vectors.keys()}
            return zero_embedding, zero_embedding.copy()

# --- 步驟 2: 改進的句法分析器 ---
class SyntacticAnalyzer:
    """簡化但有效的句法分析器"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
    @timer
    def analyze_sentence(self, sentence):
        """分析句子並提取關鍵資訊"""
        doc = self.nlp(sentence)
        
        # 提取主要成分
        analysis = {
            "sentence": sentence,
            "main_verb": None,
            "subject": [],
            "objects": [],
            "modifiers": [],
            "all_tokens": []
        }
        
        for token in doc:
            token_info = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
                "is_key": False
            }
            
            # 識別關鍵成分
            if token.dep_ == "ROOT":
                analysis["main_verb"] = token.lemma_
                token_info["is_key"] = True
            elif token.dep_ in ["nsubj", "nsubjpass"]:
                analysis["subject"].append(token.lemma_)
                token_info["is_key"] = True
            elif token.dep_ in ["dobj", "pobj"]:
                analysis["objects"].append(token.lemma_)
                token_info["is_key"] = True
            elif token.dep_ in ["amod", "advmod", "compound"]:
                analysis["modifiers"].append(token.lemma_)
                token_info["is_key"] = True
                
            analysis["all_tokens"].append(token_info)
            
        return analysis

# --- 步驟 3: 語義豐富化處理器 ---
class SemanticEnricher:
    """負責將句法分析結果與語義向量結合"""
    
    def __init__(self, embedder):
        self.embedder = embedder
        
    @timer 
    def enrich_analysis(self, syntactic_analysis):
        """為句法分析結果添加語義資訊"""
        
        # 提取需要分析的關鍵詞彙
        key_words = set()
        for token in syntactic_analysis["all_tokens"]:
            if token["is_key"] and token["lemma"].isalpha():
                key_words.add(token["lemma"])
                
        if not key_words:
            return {"error": "沒有找到可分析的關鍵詞彙"}
            
        print(f"\n將分析以下關鍵詞彙: {sorted(list(key_words))}")
        
        # 生成語義向量
        semantic_profiles = {}
        for word in tqdm(list(key_words), desc="生成語義向量"):
            v_head, v_modifier = self.embedder.create_dual_embedding(word)
            semantic_profiles[word] = {
                "V_head": v_head,
                "V_modifier": v_modifier,
                "V_average": {k: round((v_head[k] + v_modifier[k]) / 2, 3) 
                            for k in v_head.keys()}
            }
            
        return {
            "syntactic_analysis": syntactic_analysis,
            "semantic_profiles": semantic_profiles,
            "summary": {
                "total_words": len(key_words),
                "main_concepts": list(key_words)
            }
        }

# --- 步驟 4: 詞彙庫管理器 ---
class VocabularyManager:
    """管理不同模式下的詞彙庫"""
    
    @staticmethod
    def load_simlex_vocab(simlex_path):
        """載入 SimLex 詞彙"""
        simlex_vocab = set()
        try:
            with open(simlex_path, 'r', encoding='utf-8') as f:
                next(f)  # 跳過標頭
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        simlex_vocab.add(parts[0].lower())
                        simlex_vocab.add(parts[1].lower())
        except FileNotFoundError:
            print(f"[警告] 找不到 SimLex 檔案: {simlex_path}")
        return simlex_vocab
    
    @staticmethod
    @timer
    def create_debug_vocab(simlex_vocab, vocab_size=500):
        """創建除錯用詞彙庫"""
        print(f"[除錯模式] 正在建立詞彙庫...")
        
        # 嘗試從 GloVe 或 Word2Vec 獲取詞彙庫
        large_vocab = set()
        
        try:
            # 嘗試載入預訓練模型獲取詞彙
            print("嘗試載入 Word2Vec 模型以獲取詞彙庫...")
            model = api.load("word2vec-google-news-300")
            large_vocab = set(model.index_to_key[:50000])  # 取前5萬個常用詞
        except:
            try:
                print("嘗試載入 GloVe 模型以獲取詞彙庫...")
                # 這裡假設有 GloVe 檔案
                with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 50000:  # 限制詞彙量
                            break
                        word = line.split()[0]
                        large_vocab.add(word.lower())
            except:
                print("[警告] 無法載入預訓練模型，使用預設詞彙")
                # 使用一些常見英文單詞作為後備
                large_vocab = set([
                    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do',
                    'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
                    'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all',
                    'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
                    'about', 'who', 'get', 'which', 'go', 'when', 'make', 'can',
                    'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people',
                    'into', 'year', 'your', 'good', 'some', 'could', 'them',
                    'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come',
                    'way', 'work', 'life', 'day', 'get', 'use', 'man', 'new',
                    'write', 'our', 'me', 'here', 'show', 'think', 'through',
                    'back', 'much', 'before', 'move', 'right', 'boy', 'old',
                    'too', 'same', 'tell', 'does', 'set', 'each', 'want'
                ])
        
        # 移除 SimLex 詞彙以避免重複
        available_for_sampling = large_vocab - simlex_vocab
        
        # 隨機抽樣
        if len(available_for_sampling) < vocab_size:
            sampled_vocab = available_for_sampling
        else:
            sampled_vocab = set(random.sample(list(available_for_sampling), vocab_size))
        
        # 合併
        final_vocab = simlex_vocab | sampled_vocab
        
        print(f"[除錯模式] 詞彙庫建立完成:")
        print(f"  - SimLex 詞彙: {len(simlex_vocab)}")
        print(f"  - 隨機抽樣詞彙: {len(sampled_vocab)}")
        print(f"  - 總詞彙數: {len(final_vocab)}")
        
        return final_vocab

# --- 步驟 5: 主程式 ---
class FrameworkManager:
    """整合所有組件的主管理器"""
    
    def __init__(self, pole_words_path, debug_mode=False, debug_vocab_size=500, simlex_path=None):
        self.debug_mode = debug_mode
        self.debug_vocab_size = debug_vocab_size
        
        # 載入極點詞
        print("--- 正在初始化框架 ---")
        with open(pole_words_path, 'r', encoding='utf-8') as f:
            all_pole_words = json.load(f)
        
        # 初始化各組件
        self.embedder = RoleAwareEmbedding()
        self.embedder.setup_poles(all_pole_words)
        
        self.nlp = spacy.load("en_core_web_md")
        self.syntactic_analyzer = SyntacticAnalyzer(self.nlp)
        self.semantic_enricher = SemanticEnricher(self.embedder)
        
        # 處理除錯模式
        if debug_mode and simlex_path:
            print(f"\n***** 除錯模式已啟用 *****")
            simlex_vocab = VocabularyManager.load_simlex_vocab(simlex_path)
            debug_vocab = VocabularyManager.create_debug_vocab(simlex_vocab, debug_vocab_size)
            
            # 預熱快取
            print("正在預熱快取...")
            for word in tqdm(list(debug_vocab)[:100], desc="預熱中"):  # 只預熱前100個詞
                self.embedder.create_dual_embedding(word)
        
        print("--- 框架初始化完成 ---")
    
    @timer
    def analyze_sentence(self, sentence):
        """完整的句子分析流程"""
        print(f"\n--- 開始分析句子 ---")
        print(f"輸入: '{sentence}'")
        
        # 階段一: 句法分析
        syntactic_result = self.syntactic_analyzer.analyze_sentence(sentence)
        
        # 階段二: 語義豐富化
        enriched_result = self.semantic_enricher.enrich_analysis(syntactic_result)
        
        return enriched_result
    
    def print_analysis_result(self, result):
        """格式化輸出分析結果"""
        print("\n" + "="*80)
        print("                    完整分析結果")
        print("="*80)
        
        if "error" in result:
            print(f"錯誤: {result['error']}")
            return
            
        # 句法分析結果
        syntax = result["syntactic_analysis"]
        print(f"\n原句: {syntax['sentence']}")
        print(f"主動詞: {syntax['main_verb']}")
        print(f"主語: {syntax['subject']}")
        print(f"賓語: {syntax['objects']}")
        print(f"修飾詞: {syntax['modifiers']}")
        
        # 語義分析結果
        print(f"\n--- 關鍵詞彙語義分析 ---")
        semantic_profiles = result["semantic_profiles"]
        
        for word, profile in semantic_profiles.items():
            print(f"\n詞彙: '{word}'")
            print("  V_head (作為主體時):")
            head_scores = profile["V_head"]
            for dim, score in head_scores.items():
                if abs(score) > 0.1:  # 只顯示顯著的分數
                    print(f"    {dim}: {score:>6.2f}")
            
            print("  V_modifier (作為修飾語時):")
            mod_scores = profile["V_modifier"]
            for dim, score in mod_scores.items():
                if abs(score) > 0.1:  # 只顯示顯著的分數
                    print(f"    {dim}: {score:>6.2f}")
        
        print("="*80)

if __name__ == '__main__':
    # --- 設定檔案路徑 ---
    POLE_WORDS_JSON_PATH = 'all_pole_words_32d.json'
    SIMLEX_FILE_PATH = 'SimLex-999.txt'
    
    # --- 設定參數 ---
    DEBUG_MODE = True
    DEBUG_VOCAB_SIZE = 500
    
    # --- 檢查檔案 ---
    if not os.path.exists(POLE_WORDS_JSON_PATH):
        print(f"[錯誤] 找不到極點詞定義檔: {POLE_WORDS_JSON_PATH}")
    else:
        try:
            # 初始化框架
            framework = FrameworkManager(
                pole_words_path=POLE_WORDS_JSON_PATH,
                debug_mode=DEBUG_MODE,
                debug_vocab_size=DEBUG_VOCAB_SIZE,
                simlex_path=SIMLEX_FILE_PATH if os.path.exists(SIMLEX_FILE_PATH) else None
            )
            
            # 測試句子
            test_sentences = [
                "The resourceful diplomat cautiously negotiated the complex peace treaty.",
                "Scientists discovered a new species in the deep ocean.",
                "The old library contains ancient books and manuscripts."
            ]
            
            # 分析每個句子
            for sentence in test_sentences:
                result = framework.analyze_sentence(sentence)
                framework.print_analysis_result(result)
                print("\n" + "-"*50 + "\n")
                
        except Exception as e:
            print(f"[錯誤] 程式執行失敗: {e}")
            import traceback
            traceback.print_exc()
