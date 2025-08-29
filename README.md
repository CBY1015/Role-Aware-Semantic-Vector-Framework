# Role-Aware-Semantic-Vector-Framework
A Role-Aware Semantic Vector Framework that analyzes text by combining syntactic structure with a new method for deep, contextual word embedding. Inspired by neuroscience research, this approach uses "pole words" to map out multi-dimensional semantic space, generating distinct embeddings for words as "head" and "modifier" roles.

# 角色感知語義向量框架 (Role-Aware Semantic Vector Framework)

本專案提出了一種新穎的自然語言理解方法，將傳統的句法分析與獨特的**角色感知語義嵌入**技術相結合。本框架不再將詞彙視為靜態實體，而是根據其在句子中的角色（作為**主體/賓語**或**修飾語**），為同一個詞生成不同的語義剖面。

其核心靈感來自神經科學，特別是 Huth 等人於 2016 年發表的語義地圖研究。我們的目標是在不依賴昂貴且複雜的 fMRI 資料的情況下，在計算機中重現多維語義空間的概念。透過使用精心挑選的「極點詞」作為錨點，本框架能夠有效率地計算詞彙在「情感」、「價值」、「抽象程度」等多個維度上的語義分數。

---

## 核心特色

* **角色感知詞向量**：為同一個詞生成獨特的 `V_head`（主體）和 `V_modifier`（修飾語）向量，捕捉其不同的語義細微差異。
* **句法-語義整合**：結合句法分析（使用 `spaCy`）與深度語義分析，提供對句子的全面視角。

---

## 工作原理

本框架遵循一個清晰的三階段流程：

1.  **句法分析**：使用 `spaCy` 將輸入句子分解為核心成分（主語、動詞、賓語、修飾語）。
2.  **角色感知嵌入**：上一步驟中識別出的關鍵詞將由 `RoleAwareEmbedding` 模組進行處理。此模組使用「極點詞」在多個預定義維度上，為每個詞計算雙重語義向量（`V_head` 和 `V_modifier`）。
3.  **語義豐富化**：將句法和語義分析的結果合併，提供一份對句子結構及其詞彙在上下文中的意義的詳細分析。

這個過程讓我們不僅能理解句子的內容，更能從獨特的「角色感知」視角，理解每個詞是如何為其整體意義做出貢獻的。



```
```
