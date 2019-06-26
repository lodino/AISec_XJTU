# 该目录下是推荐给小组成员的阅读资料
  (请各位成员根据自己的研究方向和兴趣点学习，:star:为推荐优先阅读的资料)

## 1.对抗样本攻/防
（对抗样本攻/防方向对应的文章很多，该列表覆盖内容极其有限）

### 攻击

1. [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) :star: 

   对抗样本开山之作

1. [Explaining and Harnessing Adversarial Examples (ICLR 2015)](https://arxiv.org/abs/1412.6572) :star:

   提出对抗样本现象的一种线性解释，提出了著名的FGSM算法及对抗训练防御策略

1. [Transferability in Machine Learning: from Phenomena to Black-Box Attacks uring Adversarial Samples](https://arxiv.org/abs/1602.02697) :star: 

   利用对抗样本的可传递性，训练替身模型(substitute model)生成对抗样本，对黑盒模型进行欺骗，对几个商业应用进行了测试 

1. [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697) :star:

   上一篇文章的延续 

1. [Delving into Transferable Adversarial Examples and Black-Box Attacks (ICLR 2017)](https://arxiv.org/pdf/1611.02770.pdf)

   系统性地在大规模神经网络和数据集上对可传递性对抗样本进行了评估

1. [Towards evaluating the robustness of neural networks (S&P '17)](<https://ieeexplore.ieee.org/abstract/document/7958570/>) :star:

   著名的C&W算法

1. [Ensemble Adversarial Training: Attacks and Defenses (ICLR 2018)](https://arxiv.org/abs/1705.07204)

   提出单步攻击方法R+FGSM

1. [The limitations of Deep Learning in Adversarial Settings (EuroS&P '16)](https://ieeexplore.ieee.org/abstract/document/7467366)

   基于Jacobian矩阵的对抗样本攻击算法 (Figure 2对模型威胁进行了归纳，值得关注)

1. [Adversarial examples are not easily detected: Bypassing ten detection methods (AISec '17)](<https://arxiv.org/abs/1705.07263>) 

   证明对抗样本的检测是困难的

1. [Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](<https://arxiv.org/abs/1801.00553>) :star:

   关于视觉系统对抗样本的综述

1. [Adversarial examples for generative models (IEEE SPW 2018)](https://ieeexplore.ieee.org/abstract/document/8424630) :star:

   针对生成模型的对抗样本

1. [Audio adversarial examples: Targeted attacks on speech-to-text (2018 IEEE SPW Best Paper)](https://ieeexplore.ieee.org/abstract/document/8424625)

   对于DeepSpeech语音识别模型的对抗样本攻击

1. [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533) 

   将对抗样本进行打印，成功欺骗一个手机上的Inception分类模型

1. [Synthesizing Robust Adversarial Examples (ICML 2018)](https://arxiv.org/abs/1707.07397)

   3D打印的对抗样本，可以从多个角度欺骗视觉识别系统

1. [Accessorize to a Crime: Real and Stealthy Attacks on State-of-the-Art Face Recognition (CCS '16)](https://dl.acm.org/citation.cfm?id=2978392)

   戴上含有对抗扰动信息的镜框，可以造成人脸识别系统的误识别

1. [Robust Physical-World Attacks on Deep Learning Models (CVPR 2018)](https://arxiv.org/abs/1707.08945) :star:

   物理世界对抗样本，可以造成路标误识别等效果。考虑了打印质量、拍摄角度等信息

1. [Automatically Evading Classifiers (NDSS '16)](https://pdfs.semanticscholar.org/5e4f/a9397c18062b970910f8ee168d3297cf098f.pdf)

   PDF恶意代码检测逃逸攻击(Evasion Attack)

1. [Adversarial Examples that Fool both Computer Vision and Time-Limited Humans (NIPS '18)](http://papers.nips.cc/paper/7647-adversarial-examples-that-fool-both-computer-vision-and-time-limited-humans.pdf)

   尝试让人类辨别错误的对抗样本

1. [Motivating the Rules of the Game for Adversarial Example Research](https://arxiv.org/pdf/1807.06732.pdf) :star:

   我们该在对抗样本研究中该遵循怎样的"游戏规则"？

1. [The Relationship Between High-Dimensional Geometry and Adversarial Examples](https://arxiv.org/pdf/1801.02774.pdf)

   利用高维同心球来分析对抗样本的作用机理

1. [The Space of Transferable Adversarial Examples](https://arxiv.org/abs/1704.03453)

   分析了可传递性对抗样本的分布空间及传递性机理


### 防御
1. [Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks (S&P '16)](<https://ieeexplore.ieee.org/abstract/document/7546524>)

   蒸馏法防御

1. [Defensive Distillation is Not Robust to Adversarial Examples](https://arxiv.org/pdf/1607.04311.pdf)

   蒸馏法不够强 (Nicolas Carlini和David Wagner不少工作都是去指出现有方法不足的...)

1. [Mitigating Evasion Attacks to Deep Neural Networks via Region-based Classification (ACSAC 2017)](https://arxiv.org/pdf/1709.05583.pdf) 

   通过检测对抗样本邻域的决策结果来进行防御

1. [Decision Boundary Analysis of Adversarial Examples (ICLR 2018)](https://openreview.net/forum?id=BkpiPMbA-&noteId=BkpiPMbA-)

   指出基于邻域检测的方法可以被攻破

1. [Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks (NDSS '18)](https://arxiv.org/abs/1704.01155)

   通过特征压缩来进行防御

1. [Adversarial Example Defense: Ensembles of Weak Defenses are not Strong (WOOT '17)](https://www.usenix.org/conference/woot17/workshop-program/presentation/he)

   证明Feature Squeezing方法不够强 

1. [Magnet: a two-pronged defense against adversarial examples (CCS '17)](https://arxiv.org/pdf/1705.09064.pdf)

   利用多个神经网络进行对抗信息过滤

1. [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples (ICML '18 Best Paper)](https://arxiv.org/abs/1802.00420) :star:

   提出的攻击方法，对ICLR 2018年7篇基于梯度混淆方法中，6篇被完全攻破，1篇被部分攻破
   
(可以看到基本所有的防御方法都可以被另一种的新的攻击方法攻破，这就是adaptive attacker的假设，该部分内容推荐学习"On Evaluating Adversarial Robustness" :thinking:)

### 攻防测试

1. [Making machine learning robust against adversarial inputs](https://dl.acm.org/citation.cfm?id=3134599) :star:

   在对抗攻击与防御的军备竞赛中，我们正处在一种什么阶段，未来应该朝什么方向发展？

1. [On Evaluating Adversarial Robustness](https://arxiv.org/abs/1902.06705) :star:

   当前的模型鲁棒性评估方法是否可信？目前许多防御方法的评估过程是否合理？我们是否接近模型鲁棒性的下界？
   该文章对此进行了很好的讨论，而且该文章还在不断更新和完善中

1. [DEEPSEC: A Uniform Platform for Security Analysis of Deep Learning Model (IEEE S&P '19)](https://nesa.zju.edu.cn/download/DEEPSEC%20A%20Uniform%20Platform%20for%20Security%20Analysis%20of%20Deep%20Learning%20Model.pdf])

   一个对于深度学习的安全分析平台，设计攻击、防御、评估等环节

1. [A critique of the DeepSec Platform for Security Analysis of Deep Learning Models](https://arxiv.org/abs/1905.07112)

   DeepSec在实验结果呈现中存在哪些问题？（推荐阅读，结合前面三个文章来思考在模型安全性评估方向应该如何设计实验、得出结论）

## 2.模型可解释性

* __Feature Visualization:__ maximization activation, code inversion
(研究模型学习到的特征)
* __Attribution__
(研究模型对特定样本的决策行为)

1. [Visualizing and Understanding Convolutional Networks (ECCV '14)](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) :star:

   模型可视化解释的经典之作，利用反卷积和反池化方法尝试对模型进行解释

1. [Understanding Deep Image Representations by Inverting Them (CVPR '15)](https://arxiv.org/abs/1412.0035)

   对模型解释进行逆向解码(Feature Visualization)

1. [Visualizing Deep Convolutional Neural Networks Using Natural Pre-Images](https://arxiv.org/abs/1512.02017)

   上一个工作的推进(Feature Visualization)

1. [Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks](https://arxiv.org/abs/1602.03616) :star:

   利用activation maximization的方法寻找最大激活样本，对模型学习到的特征进行可视化解释
   
1. [Interpretable Explanations of Black Boxes by Meaningful Perturbation (ICCV '17)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Fong_Interpretable_Explanations_of_ICCV_2017_paper.pdf)

   对于黑盒模型的解释技术(Feature Visualization)

1. [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (ICCV '17)](http://202.117.4.101/cache/12/03/openaccess.thecvf.com/dba50ecdd225389aee5cedd39d57f649/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) :star:

   归因解释的经典文章(Attribution)

1. [Feature Visualization](https://distill.pub/2017/feature-visualization/) :star:

   对Feature Visualization进行了很漂亮的展示 

1. [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) :star:

   很酷炫的模型可解释性demo(上面的加强版)

1. [Interpretable Deep Learning under Fire (USENIX '20)](https://arxiv.org/abs/1812.00891)

   欺骗模型解释方法的对抗样本

1. [LEMNA: Explaining Deep Learning based Security Applications (CCS '18)](http://people.cs.vt.edu/gangwang/ccs18.pdf)

   基于模型可解释性的安全分析

## 3.模型/数据隐私

* __Model Inversion__: 窃取隐私数据
* __Model Extraction__: 窃取模型参数

1. [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures (CCS '15)](https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf) :star:

   展示了如何推测用户隐私信息(decision tree)，复原人脸训练数据(neural network)

1. [Membership Inference Attacks Against Machine Learning Models (IEEE S&P '17)](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf) :star:

   利用训练一系列shadow model进行成员推测攻击(Membership Inference Attack)

1. [Stealing Machine Learning Models via Prediction APIs (USENIX Security '16)](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf) :star:

   通过发送轮询数据可以推测模型参数，实现模型的窃取

1. [Stealing Hyperparameters in Machine Learning (IEEE S&P '18)](https://ieeexplore.ieee.org/abstract/document/8418595) :star:

   偷取模型训练时的超参数

1. [The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks (USENIX Security '19)](https://arxiv.org/pdf/1802.08232.pdf) :star:

   生成序列模型会泄露训练数据信息

1. [How You Act Tells a Lot: Privacy-Leaking Attack on Deep Reinforcement Learning (AAMAS '19)](https://dl.acm.org/citation.cfm?id=3331715)

   针对强化学习模型的隐私窃取

## 4.模型水印/后门

1. [BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain](https://arxiv.org/pdf/1708.06733.pdf)

   提出了网络模型后门的概念

1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082.pdf)

   模型水印的早期工作

1. [Trojaning Attack on Neural Networks (NDSS '18)](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech)

   并不是添加任意trigger，提出了削弱特定神经元最大激活值的trigger生成方式

1. [Protecting Intellectual Property of Deep Neural Networks with Watermarking (ASIACCS '18)](https://gzs715.github.io/pubs/WATERMARK_ASIACCS18.pdf)

   提出了三种添加模型水印的算法

1. [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring (USENIX Security '18)](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf) :star:

   注意文章中对于watermark特性的定义，值得好好学习

1. [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks (IEEE S&P '19)](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a530/19skfH8dcqc)

   提出了深度神经网络模型后门检测和去除方法(示例给的是CNN)


## 5.模型安全测试
1. [DeepXplore: Automated Whitebox Testing of Deep Learning Systems (SOSP '17 Best Paper)](https://arxiv.org/pdf/1705.06640) :star:

   借鉴了软件测试中模糊测试的概念，提出了神经元覆盖率引导的深度学习系统测试框架（白盒）

1. [DeepGauge: multi-granularity testing criteria for deep learning systems (ASE '18)](https://arxiv.org/pdf/1803.07519.pdf)

   应该是借鉴了DeepXplore的思想，对神经元覆盖率等指标进行了细化

1. [Testing deep neural networks](https://arxiv.org/pdf/1803.04792.pdf)

   受软件测试覆盖率指标启发，该文章提出了4种测试指标（白盒）

1. [DeepCT: Tomographic Combinatorial Testing for Deep Learning Systems (SANER '19)](https://ieeexplore.ieee.org/document/8668044)

   借鉴了软件测试中的组合测试概念

1. [Deepmutation: Mutation testing of deep learning systems (ISSRE '18)](https://arxiv.org/abs/1805.05206)

   借鉴了软件测试中的变异测试概念

1. [Formal Security Analysis of Neural Networks using Symbolic Intervals (USENIX Security '18)](http://www.cs.columbia.edu/~suman/docs/reluval.pdf)

   关于模型输出安全范围的形式化分析（借鉴了符号执行？）

1. [Concolic Testing for Deep Neural Networks (ASE '18)](http://www.kroening.com/papers/ase2018.pdf)

   借鉴了符号执行中的Concolic Execution

1. [TensorFuzz: Debugging Neural Networks with Coverage-Guided Fuzzing](https://arxiv.org/pdf/1807.10875.pdf)

   基于模糊测试的神经网络调试方法，发现神经网络中的数值计算错误
   
## 6.智能系统安全问题

### 概述性文章
1. [SoK: Security and Privacy in Machine Learning (IEEE EuroS&P '18)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8406613) :star: :star:

   系统性地介绍了机器学习中的安全与隐私问题(SoK:Systematization of Knowledge)，强烈推荐！

1. [A Berkeley View of Systems Challenges for AI](https://arxiv.org/abs/1712.05855) :star:

   Berkeley关于AI系统应用问题的白皮书

### 语音识别攻击

1. [DolphinAttack: Inaudible Voice Commands (CCS '17 Best Paper)](https://acmccs.github.io/papers/p103-zhangAemb.pdf)

   利用超声频段传播语音指令，使人无法察觉

1. [CommanderSong: A Systematic Approach for Practical Adversarial Voice Recognition (USENIX Security '18)](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-yuan.pdf)

   将攻击扰动隐藏到音乐中，可以经过物理信道进行攻击(play over the air)

1. [Practical Hidden Voice Attacks against Speech and Speaker Recognition Systems (NDSS '19)](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_08-1_Abdullah_paper.pdf) :star:

   对黑盒语音系统的攻击，其中利用到了信号处理的一些特性。在真实语音识别应用中，对Over-the-Air和Over-the-Line两种攻击场景都进行了测试。

1. [Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding (NDSS '19)](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_08-2_Schonherr_paper.pdf)

   欺骗人的听觉感知机制的语音对抗样本(注意不同于DolphinAttack)

1. [Characterizing Audio Adversarial Examples Using Temporal Dependency](https://arxiv.org/pdf/1809.10875.pdf)

   包含对几种语音攻击方法进行了分析

## 7.样本生成/检测
1. [Generative Adversarial Nets](http://202.117.4.101/cache/1/03/papers.nips.cc/c9eaa01bb80324b24e05e5a696fa4ab0/5423-generative-adversarial-nets.pdf) :star:
   
   Goodfellow等提出的GAN模型 

1. [生成式对抗网络GAN的研究进展与展望](http://www.aas.net.cn/CN/abstract/abstract19012.shtml)

   中文GAN综述

1. [Autoencoders, Unsupervised Learning, and Deep Architectures](http://proceedings.mlr.press/v27/baldi12a/baldi12a.pdf)

   Autoencoder学习资料

1. [Slides--Unsupervised Learning: Autoencoders by Yunsheng Bai](http://yunshengb.com/wp-content/uploads/2018/04/0412018_unsupervised_learning_autoencoders.pdf) :star:

   不错的Autoencoder模型回顾

1. [Face2Face: Real-Time Face Capture and Reenactment of RGB Videos (CVPR 2016)](https://ieeexplore.ieee.org/document/7780631) :star:

   视频换脸

1. [Synthesizing Obama: learning lip sync from audio](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf) :star:

   生成虚假的奥巴马讲话视频(主要生成唇部动作)

1. [Exposing Deep Fakes Using Inconsistent Head Poses](https://ieeexplore.ieee.org/abstract/document/8683164)

   基于头部姿态检测Deepfake

1. [In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking](https://www.albany.edu/faculty/mchang2/files/2018_12_WIFS_EyeBlink_FakeVideos.pdf)

   LSTM+CNN检测眨眼

1. [Exposing DeepFake Videos By Detecting Face Warping Artifacts](https://arxiv.org/abs/1811.00656)

   利用CNN检测Deepfake

1. [MesoNet: a Compact Facial Video Forgery Detection Network](https://ieeexplore.ieee.org/abstract/document/8630761)

   利用CNN检测Deepfake

1. [Deepfake Video Detection Using Recurrent Neural Networks](https://ieeexplore.ieee.org/abstract/document/8639163)

   利用RNN检测Deepfake

1. [ON THE GENERALIZATION OF GAN IMAGE FORENSICS](https://arxiv.org/pdf/1902.11153.pdf)

   CNN检测虚假图片
(还有类似的工作就不一一列出了，基于CNN/RNN方法的一个主要问题是:如果虚假样本生成方法未知，或者攻击者已知模型检测方法，这些提出的方法还是否奏效？)

1. [Protecting World Leaders Against Deep Fakes (CVPR Workshops 2019)](http://202.117.4.101/cache/4/03/openaccess.thecvf.com/1061984c4cca6e3b3df8d5e2f50b6803/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)

   基于面部关键部位检测
   
1. [FaceSwap-GitHub](https://github.com/deepfakes/faceswap)

   视频换脸项目

## 8.人机交互行为
   (正在添加中...)
