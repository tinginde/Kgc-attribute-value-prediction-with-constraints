# 專案檔案結構說明

這份文件詳細說明此專案中每個檔案和目錄的用途。

## 專案概述
本專案用於知識圖譜中的數值屬性預測，透過加入算術約束條件來改善預測準確度。

---

## 根目錄檔案

### README.md
- 專案的主要說明文件
- 包含專案介紹、方法論、結果和參考文獻

### .gitignore
- Git 版本控制的忽略規則設定

### Drawing_lr.ipynb
- 用於繪製學習率相關的視覺化圖表
- 包含學習曲線的繪圖分析

### check_result.ipynb
- 檢查和驗證模型預測結果
- 用於結果的後處理分析

### for_thesis_data_exploration.ipynb
- 論文用的數據探索筆記本
- 包含數據分析和視覺化

---

## LiterallyWikidata/ 目錄

此目錄包含 LiterallyWikidata 數據集的預處理相關檔案。

### Jupyter Notebooks

#### Lit48_Numericaltriples_Preprocessing_ver02.ipynb
- LiterallyWikidata 數據集的數值三元組預處理（第二版）
- 數據清理和轉換

#### Lit48_Numericaltriples_Preprocessing_ver03.ipynb
- LiterallyWikidata 數據集的數值三元組預處理（第三版）
- 更新版的數據預處理流程

### Python 檔案

#### pretrainEmb.py
- 載入和處理預訓練的知識圖譜嵌入向量
- 將實體 ID 映射到預訓練的 ComplEx 嵌入

### files_needed/ 子目錄

#### 數據文件
- **attribute.txt**: 屬性列表
- **gdp_related.txt**: GDP 相關屬性列表
- **pop_related.txt**: 人口相關屬性列表
- **list_ent_ids.txt**: 實體 ID 列表
- **list_rel_ids.txt**: 關係 ID 列表

#### Python 檔案
- **debug.py**: 除錯和測試用腳本

#### preprocess/ 子目錄
- **preprocess_num_attr.py**: 數值屬性的預處理腳本

#### pretrained_kge/ 子目錄
- 存放預訓練的知識圖譜嵌入模型

---

## baseline/ 目錄

包含所有基準模型的實現（主要是回歸模型）。

### Jupyter Notebooks

#### ML2021Spring - HW1.ipynb
- 基於台大機器學習課程的基礎回歸模型實現

#### iml_death.ipynb
- 死亡年份屬性的預測實驗

#### iml_gdp.ipynb
- GDP 屬性的預測實驗

#### iml_height.ipynb
- 身高屬性的預測實驗（最大的筆記本，包含詳細實驗）

#### iml_long.ipynb
- 經度屬性的預測實驗

### Python 檔案 - 一般模型

#### meanpredictor.py
- 基線模型：使用平均值進行預測

#### ml2021spring_hw1.py & ml2021spring_hw1_ver2.py
- 基礎機器學習模型（來自 ML2021Spring 課程）

#### ml_datasplit.py
- 數據分割工具

#### ml_height.py
- 專門用於身高預測的模型

#### ml_load_model.py & ml_load_model_var.py
- 載入已訓練模型進行預測

#### ml_minmax.py & ml_minmax_v.py
- 使用 MinMax 標準化的模型

#### ml_newstdv.py & ml_newstdv_kaiming.py
- 使用新標準化方法的模型（包含 Kaiming 初始化）

#### ml_nohidden.py
- 無隱藏層的簡單神經網路

#### ml_onlye.py & ml_onlye_cons.py
- 僅使用實體嵌入的模型（有無約束條件版本）

#### ml_threelayer.py & ml_threelayer_attless.py
- 三層神經網路模型

#### ml_twohlayer.py
- 兩個隱藏層的神經網路模型

### Python 檔案 - 單屬性預測（無約束）

以 `iml_var_` 開頭的檔案是針對特定屬性的預測模型：

- **iml_var_area.py**: 面積預測
- **iml_var_birth.py**: 出生年份預測
- **iml_var_death.py**: 死亡年份預測
- **iml_var_gdp.py**: GDP 預測
- **iml_var_height.py**: 身高預測
- **iml_var_lat.py**: 緯度預測
- **iml_var_longi.py**: 經度預測
- **iml_var_pop.py**: 人口預測
- **iml_var_ppp_per.py**: 人均購買力平價預測
- **iml_var_workend.py**: 工作結束年份預測
- **iml_var_workstart.py**: 工作開始年份預測

### Python 檔案 - 單屬性預測（含約束）

以 `iml_var_` 開頭並以 `_cons` 結尾的檔案是包含約束條件的版本：

- **iml_var_area_cons.py**: 面積預測（含約束）
- **iml_var_birth_cons.py**: 出生年份預測（含約束）
- **iml_var_death_cons.py**: 死亡年份預測（含約束）
- **iml_var_gdp_cons.py**: GDP 預測（含約束）
- **iml_var_gdp_cons_norm.py**: GDP 預測（含約束和標準化）
- **iml_var_height_cons.py**: 身高預測（含約束）
- **iml_var_lat_cons.py**: 緯度預測（含約束）
- **iml_var_longi_cons.py**: 經度預測（含約束）
- **iml_var_pop_cons.py**: 人口預測（含約束）
- **iml_var_ppp_per_cons.py**: 人均購買力平價預測（含約束）
- **iml_var_workend_cons.py**: 工作結束年份預測（含約束）
- **iml_var_workstart_cons.py**: 工作開始年份預測（含約束）

### 子目錄

#### iml_var/
- 存放變量模型的輸出和中間結果

#### iml_cons/
- 存放含約束條件模型的輸出

#### iml_cons_gdp/
- GDP 約束條件模型的特定輸出

#### iml_cons_pt/
- 使用預訓練嵌入的約束條件模型輸出

#### iml_cons_realv/
- 使用真實值的約束條件模型輸出

#### iml_cons_realv_pt/
- 使用真實值和預訓練嵌入的約束條件模型輸出

#### exp_pretrained_test/
- 預訓練模型測試實驗
- **saved_model/**: 存放已訓練的模型檔案

---

## MTKGNN/ 目錄

包含多任務知識圖譜神經網路（Multi-Task Knowledge Graph Neural Network）的實現。

### KGMTL4Rec/ 子目錄

基於 Dadoun et al. (2021) 的 KGMTL4Rec 模型的改編版本。

#### 核心模型檔案

##### Model.py
- 主要模型架構
- 包含 ER_MLP（實體-關係多層感知器）和 KGMTL（知識圖譜多任務學習）類別

##### Model_onlyATT.py
- 僅使用注意力機制的模型變體

#### 數據處理檔案

##### Data_Processing_copy_less.py
- 數據處理腳本（簡化版）

##### Data_Processing_nonorm.py
- 不進行標準化的數據處理

#### 評估檔案

##### Evaluation.py
- 模型評估指標和函數
- 包含各種性能評估方法

#### __init__.py
- Python 套件初始化檔案

### src/ 子目錄

包含各種主要執行腳本。

#### main.py
- 標準 MT-KGNN 模型的主執行檔

#### main_ast.py
- 使用 AST（可能是某種特定配置）的版本

#### main_ast_testnorm.py
- 測試標準化的 AST 版本

#### main_cons.py
- 含約束條件的版本

#### main_constraints.py
- 約束條件模型的主執行檔

#### main_constraints_withage.py
- 包含年齡約束的版本

#### main_onlyATT.py
- 僅使用注意力機制的執行檔

#### modelload.py
- 載入已訓練模型的工具

---

## 檔案命名慣例

1. **iml_var_XXX.py**: 針對特定屬性 XXX 的回歸模型（無約束）
2. **iml_var_XXX_cons.py**: 針對特定屬性 XXX 的回歸模型（含約束）
3. **main_XXX.py**: MT-KGNN 模型的不同配置版本
4. **Model_XXX.py**: 模型架構的不同變體

---

## 主要技術架構

### 1. 基線方法 (baseline/)
- 使用深度神經網路的回歸模型
- 輸入：實體嵌入 + 相關屬性值
- 輸出：預測的數值屬性

### 2. 多任務方法 (MTKGNN/)
- 基於知識圖譜的多任務學習
- 同時學習多個屬性的預測
- 使用圖神經網路架構

### 3. 約束條件整合
- 在損失函數中加入算術約束
- 使用相關屬性作為特徵
- 例如：工作結束年 ≥ 工作開始年

---

## 數據流程

1. **數據預處理** (LiterallyWikidata/)
   - 載入 LiterallyWikidata 數據集
   - 清理和標準化數值屬性
   - 準備訓練/驗證/測試集

2. **嵌入準備**
   - 使用預訓練的 ComplEx 嵌入
   - 將實體映射到嵌入空間

3. **模型訓練**
   - Baseline: 單獨訓練每個屬性
   - MT-KGNN: 多任務聯合訓練

4. **評估**
   - 計算 MAE、MSE、RMSE 等指標
   - 視覺化學習曲線
   - 分析預測結果

---

## 實驗組織

### 基線實驗
- 針對 12 個數值屬性分別建模
- 比較有無約束條件的效果
- 測試不同的網路架構和標準化方法

### MT-KGNN 實驗
- 多任務學習方法
- 測試不同的模型配置
- 評估約束條件對多任務學習的影響

---

## 如何使用

### 執行基線模型
```bash
# 預測特定屬性（例如身高）
python baseline/iml_var_height.py

# 預測特定屬性（含約束）
python baseline/iml_var_height_cons.py
```

### 執行 MT-KGNN 模型
```bash
# 標準版本
python MTKGNN/src/main.py

# 含約束條件版本
python MTKGNN/src/main_constraints.py
```

### 數據預處理
```bash
# 處理預訓練嵌入
python LiterallyWikidata/pretrainEmb.py

# 預處理數值屬性
python LiterallyWikidata/files_needed/preprocess/preprocess_num_attr.py
```

---

## 依賴套件

主要使用的 Python 套件：
- PyTorch: 深度學習框架
- NumPy: 數值計算
- Pandas: 數據處理
- Matplotlib: 視覺化
- scikit-learn: 數據預處理和評估

---

## 參考文獻

1. Dadoun et al. (2021) - KGMTL4Rec 模型
2. Gesese et al. (2021) - LiterallyWikidata 數據集
