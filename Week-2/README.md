# Week — Image Classification with CNNs: MNIST & Flowers

### MNIST Dataset / MNISTデータセット

![MNIST Examples](mnist_examples.png)

### Flower Dataset / 花のデータセット

![Flower Examples](flower_examples.png)

# 今週の演習 — CNNによる画像分類：MNISTと花画像データセット

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Notebooks](https://img.shields.io/badge/Notebooks-English%20%2B%20Japanese-green)](#tutorial-notebooks--演習ノートブック)

## Overview / 概要

This week's hands-on tutorials introduce **Convolutional Neural Networks (CNNs)** for image classification. We start with the relatively simple **MNIST handwritten-digit dataset**, apply the same ideas to a more challenging **five-class flower dataset**, and then explore **transfer learning with pretrained ResNet18 / ResNet50**.

今週は、**畳み込みニューラルネットワーク（CNN）**を使った画像分類を学びます。まず、比較的簡単な**MNIST手書き数字データセット**でCNNの基本を理解し、次に**5種類の花画像の分類**に挑戦します。最後に、**事前学習済みResNet18 / ResNet50を使った転移学習**を体験します。

**Audience / 対象：** Second-year undergraduate students (学部2年生)  
**Language / 言語：** English and Japanese (英語・日本語)  
**Framework / フレームワーク：** Python, PyTorch, Torchvision, Jupyter Notebook  
**Recommended duration / 目安時間：** 2–3 hours for MNIST + basic Flowers; ResNet is an optional demonstration or follow-up exercise.  
**所要時間：** MNISTと基本的な花分類で2〜3時間。ResNetは追加のデモまたは発展課題を想定しています。

## Learning objectives / 学習目標

After completing these tutorials, students should be able to / 演習後、次のことができるようになることを目指します：

1. Describe how images are represented as tensors (pixels, height, width, channels). / 画像のテンソル表現（画素・高さ・幅・チャネル）を説明できる。
2. Explain convolution, ReLU, max pooling, and feature maps. / 畳み込み・ReLU・最大プーリング・特徴マップの役割を説明できる。
3. Build, train, and evaluate a CNN with PyTorch. / PyTorchを使ってCNNを構築・学習・評価できる。
4. Interpret loss curves, accuracy, confusion matrices, and incorrect predictions. / 損失曲線・正解率・混同行列・誤分類を読み取れる。
5. Understand image augmentation and why flower images are harder than MNIST. / データ拡張と、花の分類がMNISTより難しい理由を理解する。
6. Explain the difference between feature extraction and fine-tuning with a pretrained ResNet. / 事前学習済みResNetにおける特徴抽出とファインチューニングの違いを説明できる。

## Tutorial notebooks / 演習ノートブック

| File / ファイル | Topic / 内容 | Highlights / 主な演習 |
|---|---|---|
| [`01_MNIST_CNN_EN_JA.ipynb`](01_MNIST_CNN_EN_JA.ipynb) | **MNIST — CNN fundamentals / CNNの基礎** | Convolution by hand, pooling, CNN training, learned kernels, intermediate feature maps, probability predictions / 畳み込みの手計算、プーリング、学習、フィルタと特徴マップの可視化 |
| [`02_Flowers_CNN_EN_JA.ipynb`](02_Flowers_CNN_EN_JA.ipynb) | **Flowers — CNN from scratch / CNNを一から学習** | RGB images, five classes, data augmentation, training/evaluation, errors and feature maps / RGB画像、5クラス、データ拡張、学習・評価、誤分類 |
| [`03_Flowers_Transfer_Learning_ResNet18_ResNet50_EN_JA.ipynb`](03_Flowers_Transfer_Learning_ResNet18_ResNet50_EN_JA.ipynb) | **Flowers — transfer learning / 転移学習** | Pretrained ResNet18/50, frozen backbone, fine-tuning `layer4`, Grad-CAM, saving weights / 事前学習済みモデル、重みの固定、`layer4`の再学習、Grad-CAM、モデル保存 |

### Suggested folder structure / 推奨フォルダ構成

Place all three notebooks and `requirements.txt` in the **same folder**. Dataset directories are created automatically when you run the notebooks.

3つのノートブックと`requirements.txt`を**同じフォルダ**に置いてください。データ用のフォルダは実行時に自動作成されます。

```text
CNN-Week/
├── README.md
├── requirements.txt
├── 01_MNIST_CNN_EN_JA.ipynb
├── 02_Flowers_CNN_EN_JA.ipynb
├── 03_Flowers_Transfer_Learning_ResNet18_ResNet50_EN_JA.ipynb
├── data/                      # MNIST (created automatically)
├── flower_data/               # Flower images (created automatically)
└── trained_flower_models/     # ResNet checkpoints (created by Notebook 3)
```

> **Git note / Gitについて:** The datasets and trained model files may be large. Consider adding `data/`, `flower_data/`, `trained_flower_models/`, `.ipynb_checkpoints/`, and `__pycache__/` to `.gitignore` rather than committing them. / 大きなデータや学習済みモデルは、GitHubへアップロードせず`.gitignore`で除外することを推奨します。

## 1. Environment setup / 環境構築

Python **3.11** and a dedicated Anaconda/Miniconda environment are recommended. The supplied `requirements.txt` pins specific versions, including **PyTorch 2.5.1** and **Torchvision 0.20.1**.

Python **3.11**と専用のAnaconda/Miniconda環境を推奨します。`requirements.txt`にはPyTorch **2.5.1**、Torchvision **0.20.1**などのバージョンを指定しています。

```bash
# 1. Create and activate the environment / 環境を作成して有効化
conda create -n cnn-teaching python=3.11 -y
conda activate cnn-teaching

# 2. Install the required libraries / 必要なライブラリをインストール
python -m pip install -r requirements.txt

# 3. Check dependencies / 依存関係を確認
python -m pip check

# 4. Register a Jupyter kernel / Jupyterカーネルを登録
python -m ipykernel install --user --name cnn-teaching --display-name "Python (CNN Teaching)"

# 5. Start JupyterLab / JupyterLabを起動
jupyter lab
```

Open a notebook, select **Python (CNN Teaching)**, and run its cells **from top to bottom**. / ノートブックを開き、**Python (CNN Teaching)** を選択し、**上から順にセルを実行**してください。

### Quick verification / 動作確認

```python
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Apple MPS available:", torch.backends.mps.is_available())

x = torch.randn(2, 3, 96, 96)
conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
print("Example convolution shape:", conv(x).shape)  # [2, 16, 96, 96]
```

**Hardware note / 実行環境の注意:** Notebooks 1 and 2 automatically select **CUDA or CPU**. Notebook 3 additionally supports **Apple MPS**. On an Apple Silicon Mac, Notebooks 1 and 2 will use the CPU unless you explicitly add MPS device selection. An NVIDIA GPU or Apple GPU is helpful for ResNet, but a GPU is not required to learn the fundamentals.

**ハードウェアについて：** ノートブック1・2はCUDAまたはCPUを自動選択します。ノートブック3はApple MPSにも対応しています。Apple Silicon搭載Macでノートブック1・2をGPU実行したい場合は、デバイス選択処理を変更してください。

## 2. Notebook 1 — MNIST / MNISTによる基礎学習

**Dataset / データセット：** 70,000 grayscale handwritten-digit images, 10 classes (0–9), each **1 × 28 × 28**. / 手書き数字70,000枚、10クラス（0〜9）、**1 × 28 × 28**のグレースケール画像。

**Data split / データ分割：** 55,000 training, 5,000 validation, 10,000 test images. / 学習55,000枚、検証5,000枚、テスト10,000枚。

The model in the notebook follows this exact structure / ノートブック内のモデル構成：

```text
Input / 入力                         1 × 28 × 28
  ↓ Conv2d(1 → 16, 3×3, padding=1) 16 × 28 × 28
  ↓ ReLU                             16 × 28 × 28
  ↓ MaxPool2d(2)                     16 × 14 × 14
  ↓ Conv2d(16 → 32, 3×3, padding=1)32 × 14 × 14
  ↓ ReLU                             32 × 14 × 14
  ↓ MaxPool2d(2)                     32 × 7 × 7
  ↓ Flatten                          1,568 features
  ↓ Linear(1,568 → 64) + ReLU        64 features
  ↓ Linear(64 → 10)                  10 class scores (logits)
```

**Try / 実験してみよう：** Change `FILTERS = 16` to `8` or `32`; compare training time, number of parameters, validation accuracy, and learned feature maps. The default is `EPOCHS = 3`. / フィルタ数を変更し、学習時間・パラメータ数・検証正解率・特徴マップを比較しましょう。

## 3. Notebook 2 — Flowers CNN from scratch / 花分類用CNNを一から学習

**Dataset / データセット：** TensorFlow's public [`flower_photos`](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) sample dataset, approximately **3,670 images** of **daisy, dandelion, roses, sunflowers, and tulips**. / 5種類の花画像（デイジー、タンポポ、バラ、ヒマワリ、チューリップ）約3,670枚。

**Data split / データ分割：** Stratified 70% train / 15% validation / 15% test, with `SEED = 42`. / クラス比率を維持した70%・15%・15%の分割。

**Preprocessing / 前処理：** Resize to **96 × 96 RGB**, convert to tensor, normalize with mean/std `(0.5, 0.5, 0.5)`; training images also receive random horizontal flips and rotations of up to ±15°. / 96 × 96のRGB画像に変換し、正規化を行います。学習画像には左右反転と±15°の回転を適用します。

```text
Input / 入力                         3 × 96 × 96
  ↓ Conv2d(3 → 16, 3×3, padding=1)  16 × 96 × 96
  ↓ ReLU + MaxPool2d(2)              16 × 48 × 48
  ↓ Conv2d(16 → 32, 3×3, padding=1)32 × 48 × 48
  ↓ ReLU + MaxPool2d(2)              32 × 24 × 24
  ↓ Conv2d(32 → 64, 3×3, padding=1)64 × 24 × 24
  ↓ ReLU + MaxPool2d(2)              64 × 12 × 12
  ↓ AdaptiveAvgPool2d((1,1))         64 × 1 × 1
  ↓ Flatten                          64 features
  ↓ Linear(64 → 5)                   5 class scores (logits)
```

**Try / 実験してみよう：** Compare the original CNN with and without data augmentation; try `EPOCHS = 3` versus 8–12 (outside class, if needed). Inspect confusion matrices and misclassified flowers. / データ拡張の有無、エポック数の違いを比較し、混同行列や誤分類画像を確認しましょう。

## 4. Notebook 3 — Pretrained ResNet / 事前学習済みResNet

Instead of training all features from scratch, we reuse weights pretrained on **ImageNet**. Use **ResNet18** for a quick lesson; switch to **ResNet50** for an additional comparison.

画像特徴を一から学習するのではなく、**ImageNet**で事前学習された重みを再利用します。授業では軽量な**ResNet18**を推奨し、**ResNet50**は比較・発展課題に利用します。

| Model / モデル | Replacement classifier / 新しい分類層 | Notes / 特徴 |
|---|---|---|
| ResNet18 | `nn.Linear(512, 5)` | Faster and lighter / 高速・軽量 |
| ResNet50 | `nn.Linear(2048, 5)` | Deeper and more expensive / より深く計算量が大きい |

**Input / 入力：** Training uses random **224 × 224** crops and ImageNet normalization; validation and test use the corresponding official pretrained-weight transforms. / 学習時は224 × 224のランダムクロップとImageNet用正規化、検証・テスト時は公式の前処理を使用します。

### Why two training phases? / なぜ2段階で学習するのか？

- **Phase A — Feature extraction / 特徴抽出：** Freeze the pretrained ResNet backbone; train **only `fc`**. / 事前学習済みの層を固定し、最後の分類層`fc`だけを学習します。
- **Phase B — Fine-tuning / ファインチューニング：** Continue from Phase A; unfreeze **`layer4` and `fc`**, keeping earlier layers frozen. / 第1段階のモデルを引き継ぎ、`layer4`と`fc`を学習可能にします。

**Important / 重要：** Both phases use the **same training split**; the classifier continues training because the features change during fine-tuning. Validation data is used to select the model, and the held-out test set is reserved for final evaluation. / 2つの段階では**同じ学習データ**を使用します。特徴抽出部分の変化に合わせて分類層も再調整します。モデル選択には検証データを使い、テストデータは最終評価にのみ使います。

Notebook 3 defaults to `MODEL_NAME = 'resnet18'`. Changing it to `'resnet50'` requires rerunning all cells from the beginning. On CPU, `QUICK_DEMO` automatically limits the training/validation samples and shortens the number of epochs; the test split remains unchanged. A quick demonstration is **not** an exhaustive model comparison.

ノートブック3は初期設定でResNet18を使用します。CPUでは短縮モードが有効になり、学習・検証に使う画像枚数やエポック数が少なくなります（テストデータの分割は維持します）。

## 5. Visualization / 可視化

The notebooks generate visualizations from the executed models, including / 各ノートブックでは次の図を作成します：

- Sample images and class labels / 画像例とラベル
- Convolution and pooling outputs / 畳み込み・プーリングの出力
- Learned convolution kernels and intermediate feature maps / 学習済みカーネル・中間特徴マップ
- Training and validation loss/accuracy curves / 学習・検証の損失と正解率
- Confusion matrices, class scores, and incorrect predictions / 混同行列・クラス別スコア・誤分類
- **Optional:** Grad-CAM heatmaps in the ResNet notebook / **発展：** ResNetのGrad-CAM可視化

> **Interpreting class scores / スコアの解釈：** The final `Linear` layer produces **logits**, not probabilities. `torch.softmax(logits, dim=1)` converts them to class probabilities for display. `nn.CrossEntropyLoss` accepts logits directly; do not apply softmax before it. / 最終層は確率ではなくロジットを出力します。確率表示にはSoftmaxを使いますが、`CrossEntropyLoss`の入力にはSoftmaxを適用しません。

## 6. Suggested lesson schedule / 推奨授業スケジュール

| Approx. time / 時間 | Activity / 内容 |
|---|---|
| 0–20 min | Image tensors, convolution, ReLU, max pooling / 画像テンソル、畳み込み、ReLU、プーリング |
| 20–70 min | MNIST CNN: run, evaluate, visualize / MNISTのCNN学習・評価・可視化 |
| 70–80 min | Break / 休憩 |
| 80–135 min | Flowers CNN: RGB, augmentation, classification / 花分類：RGB、データ拡張、分類 |
| 135–160 min | ResNet18 demonstration or pretrained-model discussion / ResNet18デモ・転移学習の説明 |
| 160–170 min | Discussion and recap / 考察・まとめ |

For a strict **2-hour session**, treat the ResNet notebook as follow-up work. Run dataset downloads and any long training in advance to avoid waiting during class. / **2時間の授業**の場合、ResNetは発展課題とし、データのダウンロードや時間のかかる学習は授業前に済ませてください。

## 7. Discussion questions / 考察課題

1. Why do CNNs work well for image recognition? / CNNが画像認識に向いているのはなぜでしょうか？
2. Why can the number of channels increase while height/width decrease? / 高さ・幅が小さくなる一方、チャネル数が増えるのはなぜでしょうか？
3. Why is flower classification harder than MNIST? / 花の分類がMNISTより難しいのはなぜでしょうか？
4. What signs of overfitting appear in the learning curves? / 学習曲線から過学習をどのように判断できますか？
5. Why do we replace the pretrained ResNet classification layer? / ResNetの最後の分類層を置き換えるのはなぜでしょうか？
6. What changes between feature extraction and fine-tuning? / 特徴抽出とファインチューニングでは何が変わるのでしょうか？

## 8. Troubleshooting / よくある問題

| Issue / 問題 | Suggested action / 対処方法 |
|---|---|
| `ModuleNotFoundError` | Activate `cnn-teaching`, install `requirements.txt`, and select the correct Jupyter kernel. / 環境・カーネルを確認してください。 |
| Dataset download fails / ダウンロード失敗 | Check the Internet connection; the first run requires dataset downloads. / ネット接続を確認してください。 |
| ResNet weights fail to download / ResNetの重み取得失敗 | Run once with Internet access; pretrained ImageNet weights must be cached. / インターネット接続環境で一度実行してください。 |
| Training is slow / 学習が遅い | Try ResNet18 rather than ResNet50; use a GPU, keep CPU `QUICK_DEMO` enabled, or reduce epochs for a demonstration. / ResNet18や短縮モードを使ってください。 |
| Japanese characters show as boxes / 日本語が四角になる | Configure a Japanese-compatible Matplotlib font, e.g. `Hiragino Sans` on macOS if available. / 日本語対応フォントを設定してください。 |
| `CUDA available: False` | This is normal on CPU-only machines and on Macs without NVIDIA CUDA. Apple Silicon may support MPS. / CPUやMacでは正常な場合があります。 |

## References / 参考資料

- [PyTorch — Official tutorials](https://docs.pytorch.org/tutorials/)
- [Torchvision — MNIST dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [TensorFlow — Flowers classification tutorial (Japanese)](https://www.tensorflow.org/tutorials/images/classification?hl=ja)
- [PyTorch — Transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Deep Residual Learning for Image Recognition (He et al., 2016)](https://arxiv.org/abs/1512.03385)

---

**Note / 注意：** Accuracy and running time are not fixed; they depend on the random seed, device, data augmentation, and number of epochs. The tutorials are for education and experimentation, not benchmark claims. / 正解率や実行時間は環境・学習条件によって変わります。本教材は学習・実験を目的としています。
