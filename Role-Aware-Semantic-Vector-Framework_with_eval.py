import os
import json
import numpy as np
from scipy.stats import spearmanr
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim.downloader as api
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

# --- 步驟 1: 自定義「角色感知詞向量」產生器 ---
class RoleAwareEmbedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"\n[RoleAware] 正在載入 Sentence Transformer 基底模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.pole_vectors = {}
        self.cache = {}
        print("[RoleAware] 基底模型準備完成。")

    def setup_poles(self, all_pole_words):
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
        if len(pos_vectors) == 0 or len(neg_vectors) == 0: return 0.0
        pos_sim = cosine_similarity(target_vector, pos_vectors).mean()
        neg_sim = cosine_similarity(target_vector, neg_vectors).mean()
        raw_score = pos_sim - neg_sim
        return float(np.tanh(raw_score * 2.5))

    def create_dual_embedding(self, word: str):
        if word in self.cache:
            return self.cache[word]
        
        target_vector = self.model.encode([word])
        head_embedding, modifier_embedding = {}, {}
        for dim_name, roles in self.pole_vectors.items():
            clean_dim_name = dim_name.split('. ')[1] if '. ' in dim_name else dim_name
            head_poles = roles.get('head', {})
            head_score = self._calculate_score(target_vector, head_poles.get('pos', []), head_poles.get('neg', []))
            head_embedding[clean_dim_name] = round(head_score, 3)
            
            modifier_poles = roles.get('modifier', {})
            modifier_score = self._calculate_score(target_vector, modifier_poles.get('pos', []), modifier_poles.get('neg', []))
            modifier_embedding[clean_dim_name] = round(modifier_score, 3)

        v_head_array = np.array(list(head_embedding.values()))
        v_modifier_array = np.array(list(modifier_embedding.values()))
        
        self.cache[word] = (v_head_array, v_modifier_array)
        return v_head_array, v_modifier_array

# --- 步驟 2: 評估函式 ---

# 2.1 通用及基準模型評估函式
def build_gensim_model(embedding_dict):
    if not embedding_dict: return None
    vector_size = len(next(iter(embedding_dict.values())))
    model = KeyedVectors(vector_size=vector_size)
    words, vectors = list(embedding_dict.keys()), np.array(list(embedding_dict.values()))
    model.add_vectors(words, vectors)
    return model

def evaluate_similarity(model, model_name, simlex_path):
    print(f"\n--- [{model_name}] 開始內部評估 (詞彙相似度) ---")
    model_scores, human_scores, oov_count = [], [], 0
    with open(simlex_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            word1, word2, score = parts[0], parts[1], float(parts[3])
            if hasattr(model, 'has_index_for') and model.has_index_for(word1) and model.has_index_for(word2):
                similarity = model.similarity(word1, word2)
                model_scores.append(similarity)
                human_scores.append(score)
            else:
                oov_count += 1
    if not model_scores: return 0.0, 0
    spearman_corr, _ = spearmanr(human_scores, model_scores)
    print(f"[{model_name}] 內部評估完成！找到 {len(model_scores)} 詞彙對，未找到 {oov_count} 對。")
    return spearman_corr

def document_to_vector_static(doc, model):
    words = [word for word in nltk.word_tokenize(doc.lower()) if word.isalpha() and hasattr(model, 'has_index_for') and model.has_index_for(word)]
    if not words: return np.zeros(model.vector_size)
    return np.mean(model[words], axis=0)

def evaluate_classification_static(model, model_name, dataset):
    print(f"\n--- [{model_name}] 開始外部評估 (文本分類) ---")
    print(f"正在為 [{model_name}] 將文件轉換為向量...")
    X = np.array([document_to_vector_static(doc, model) for doc in tqdm(dataset.data, desc=f"處理 {model_name}")])
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[{model_name}] 外部評估完成！")
    return accuracy

def evaluate_similarity_bert(model, model_name, simlex_path):
    print(f"\n--- [{model_name}] 開始內部評估 (詞彙相似度) ---")
    human_scores, model_scores = [], []
    word_pairs = []
    with open(simlex_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            word_pairs.append(parts[:2])
            human_scores.append(float(parts[3]))

    words_to_encode = list(set([word for pair in word_pairs for word in pair]))
    print(f"正在為 {len(words_to_encode)} 個 SimLex 詞彙生成 BERT 向量...")
    embeddings = model.encode(words_to_encode, show_progress_bar=True)
    word_vec_map = {word: vec for word, vec in zip(words_to_encode, embeddings)}

    for (word1, word2) in word_pairs:
        if word1 in word_vec_map and word2 in word_vec_map:
            vec1 = word_vec_map[word1]
            vec2 = word_vec_map[word2]
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            model_scores.append(similarity)

    if not model_scores: return 0.0
    spearman_corr, _ = spearmanr(human_scores, model_scores)
    print(f"[{model_name}] 內部評估完成！找到 {len(model_scores)} 詞彙對。")
    return spearman_corr

def evaluate_classification_bert(model, model_name, dataset):
    print(f"\n--- [{model_name}] 開始外部評估 (文本分類) ---")
    print(f"正在為 [{model_name}] 將文件轉換為向量 (此為情境感知模型，速度較慢)...")
    X = model.encode(dataset.data, show_progress_bar=True)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[{model_name}] 外部評估完成！")
    return accuracy

# 2.2 角色感知策略評估函式 (採用即時生成)

MODIFIER_DEPS = {"amod", "advmod", "compound", "det", "acl", "advcl", "poss", "case", "npadvmod", "nummod", "quantmod"}
HEAD_DEPS = {"ROOT", "nsubj", "dobj", "pobj", "attr", "acomp", "nsubjpass", "csubj", "csubjpass", "xcomp", "appos"}

def ensure_word_embedded(word, embedder, head_embeds, mod_embeds, avg_embeds):
    """一個輔助函式，確保一個詞的向量已存在，若不存在則即時生成。"""
    if word not in head_embeds:
        v_head, v_mod = embedder.create_dual_embedding(word)
        head_embeds[word] = v_head
        mod_embeds[word] = v_mod
        avg_embeds[word] = (v_head + v_mod) / 2.0

def document_to_vector_role_aware(doc_text, embedder, head_embeds, mod_embeds, avg_embeds, nlp):
    """(即時生成版) 根據依存關係，動態選擇V_head/V_modifier，若詞彙未見過則即時生成向量。"""
    doc = nlp(doc_text.lower())
    doc_vectors = []
    
    for token in doc:
        word = token.lemma_
        if token.is_stop or not token.is_alpha:
            continue
        
        # 確保這個詞的向量已經被計算過
        ensure_word_embedded(word, embedder, head_embeds, mod_embeds, avg_embeds)
            
        selected_vector = None
        if token.dep_ in HEAD_DEPS:
            selected_vector = head_embeds[word]
        elif token.dep_ in MODIFIER_DEPS:
            selected_vector = mod_embeds[word]
        else:
            selected_vector = avg_embeds[word] # Fallback
        
        doc_vectors.append(selected_vector)

    if not doc_vectors:
        return np.zeros(32) # Fallback to a zero vector of correct dimension
        
    return np.mean(doc_vectors, axis=0)

def evaluate_classification_role_aware(strategy, embedder, dataset, nlp):
    """角色感知策略的分類評估主函式"""
    strategy_name = "策略 A (平均融合)" if strategy == 'A' else "角色感知策略 (依存規則)"
    print(f"\n--- [{strategy_name}] 開始外部評估 (文本分類) ---")
    print("正在將文件轉換為向量 (採用即時生成策略)...")

    head_embeds, mod_embeds, avg_embeds = {}, {}, {}
    X = []

    for doc_text in tqdm(dataset.data, desc=f"處理 {strategy_name}"):
        if strategy == 'B': # 策略 B 使用依存規則
            vec = document_to_vector_role_aware(doc_text, embedder, head_embeds, mod_embeds, avg_embeds, nlp)
        else: # 策略 A 只使用平均向量
            doc_vectors_A = []
            for token in nlp(doc_text.lower()):
                word = token.lemma_
                if token.is_stop or not token.is_alpha:
                    continue
                ensure_word_embedded(word, embedder, head_embeds, mod_embeds, avg_embeds)
                doc_vectors_A.append(avg_embeds[word])
            
            if not doc_vectors_A:
                vec = np.zeros(32)
            else:
                vec = np.mean(doc_vectors_A, axis=0)
        X.append(vec)
    
    X = np.array(X)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[{strategy_name}] 外部評估完成！共處理了 {len(head_embeds)} 個獨立詞彙。")
    return accuracy

def evaluate_similarity_role_aware(embedder, simlex_path):
    """(即時生成版) 策略 A 在 SimLex-999 上的評估"""
    model_name = "策略 A (平均融合)"
    print(f"\n--- [{model_name}] 開始內部評估 (詞彙相似度) ---")
    
    simlex_vocab = set()
    with open(simlex_path, 'r', encoding='utf-8') as f:
        next(f); [simlex_vocab.update(line.strip().split('\t')[:2]) for line in f]
    
    print(f"正在為 {len(simlex_vocab)} 個 SimLex 詞彙即時生成向量...")
    avg_embeds = {}
    for word in tqdm(list(simlex_vocab), desc="生成 SimLex 向量"):
        v_head, v_mod = embedder.create_dual_embedding(word)
        avg_embeds[word] = (v_head + v_mod) / 2.0

    model_A = build_gensim_model(avg_embeds)
    if not model_A: return 0.0
    return evaluate_similarity(model_A, model_name, simlex_path)

# --- 步驟 3: 主程式執行區塊 ---
if __name__ == '__main__':
    BASE_PATH = 'E:/wordsim/' 
    POLE_WORDS_JSON_PATH = os.path.join(BASE_PATH, 'all_pole_words_32d.json')
    SIMLEX_FILE_PATH = os.path.join(BASE_PATH, 'SimLex-999', 'SimLex-999.txt')
    GLOVE_FILE_PATH = os.path.join(BASE_PATH, 'glove.6B.100d.txt')
    
    # 在即時生成模式下，DEBUG_MODE 會限制處理的文檔數量，而不是詞彙量
    DEBUG_MODE = True
    DEBUG_DOC_COUNT = 500 # 除錯模式下處理的文件數量
    
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')

    if not all(os.path.exists(p) for p in [POLE_WORDS_JSON_PATH, SIMLEX_FILE_PATH, GLOVE_FILE_PATH]):
        print("\n錯誤：找不到必要的檔案！請確認以上檔案路徑設定正確。")
    else:
        if DEBUG_MODE:
            print("\n" + "="*50 + "\n 警告：即將進入智慧除錯模式 (僅處理部分文件)\n" + "="*50 + "\n")
        else:
            print("\n" + "="*50 + "\n 警告：即將進入完整評估模式 (可能需要較長時間)\n" + "="*50 + "\n")
        
        # --- 載入所有基準模型 ---
        print("--- 正在載入所有基準模型 (首次執行會自動下載) ---")
        glove_word2vec_file = GLOVE_FILE_PATH + '.word2vec'
        if not os.path.exists(glove_word2vec_file):
            print(f"正在將 GloVe 格式轉換為 Word2Vec 格式...")
            glove2word2vec(GLOVE_FILE_PATH, glove_word2vec_file)
        glove_model = KeyedVectors.load_word2vec_format(glove_word2vec_file, binary=False)
        print(f"[GloVe] 模型載入成功！")
        
        word2vec_model = api.load("word2vec-google-news-300")
        print(f"[Word2Vec] 模型載入成功！")

        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"[BERT] 模型載入成功！")
        
        # --- 建立我們的角色感知模型產生器 ---
        with open(POLE_WORDS_JSON_PATH, 'r', encoding='utf-8') as f:
            all_pole_words = json.load(f)
        embedder = RoleAwareEmbedding()
        embedder.setup_poles(all_pole_words)

        # --- 準備評估資料集 ---
        print("\n--- 正在準備評估資料集 ---")
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        dataset_full = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
        
        dataset_eval = dataset_full
        if DEBUG_MODE:
            print(f"***** 智慧除錯模式已啟用，僅評估 {DEBUG_DOC_COUNT} 份文件 *****")
            dataset_eval.data = dataset_full.data[:DEBUG_DOC_COUNT]
            dataset_eval.target = dataset_full.target[:DEBUG_DOC_COUNT]

        nlp = spacy.load("en_core_web_md")
        
        # --- 開始執行所有評估 ---
        
        # -- 內部評估 (詞彙相似度) --
        sim_glove = evaluate_similarity(glove_model, "GloVe", SIMLEX_FILE_PATH)
        sim_w2v = evaluate_similarity(word2vec_model, "Word2Vec", SIMLEX_FILE_PATH)
        sim_bert = evaluate_similarity_bert(bert_model, "BERT", SIMLEX_FILE_PATH)
        sim_A = evaluate_similarity_role_aware(embedder, SIMLEX_FILE_PATH)
        
        # -- 外部評估 (文本分類) --
        class_glove = evaluate_classification_static(glove_model, "GloVe", dataset_eval)
        class_w2v = evaluate_classification_static(word2vec_model, "Word2Vec", dataset_eval)
        class_bert = evaluate_classification_bert(bert_model, "BERT", dataset_eval)
        # 我們的策略 A 和 B 使用新的即時生成評估函式
        class_A = evaluate_classification_role_aware('A', embedder, dataset_eval, nlp)
        class_B = evaluate_classification_role_aware('B', embedder, dataset_eval, nlp)
        
        # --- 最終報告 ---
        print("\n\n" + "="*70)
        print("          角色感知詞向量 vs. 各大基準模型 完整評估報告 (即時生成版)")
        print("="*70)
        if DEBUG_MODE:
            print("          ***** 目前為智慧除錯模式 (結果僅供參考) *****")
        
        print("\n--- 內部評估 (Intrinsic) - SimLex-999 詞彙相似度 ---")
        print("指標: Spearman's Rho (越高越好)")
        print(f"  - GloVe (100d):                   {sim_glove:.4f}")
        print(f"  - Word2Vec (300d):                {sim_w2v:.4f}")
        print(f"  - BERT (s-bert, 384d):            {sim_bert:.4f}")
        print(f"  - 策略 A (平均融合, 32d):       {sim_A:.4f}")
        
        print(f"\n--- 外部評估 (Extrinsic) - 20 Newsgroups 分類 ({len(dataset_eval.data)} 份文件) ---")
        print("指標: Accuracy (越高越好)")
        print(f"  - GloVe (100d):                   {class_glove:.4f}")
        print(f"  - Word2Vec (300d):                {class_w2v:.4f}")
        print(f"  - BERT (s-bert, 384d):            {class_bert:.4f}")
        print(f"  - 策略 A (平均融合, 32d):       {class_A:.4f}")
        print(f"  - 角色感知策略 (依存規則, 32d): {class_B:.4f}")
        print("="*70)
