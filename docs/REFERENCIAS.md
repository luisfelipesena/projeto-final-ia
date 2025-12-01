# Referências Científicas - Projeto YouBot Autônomo

**Projeto:** Sistema Autônomo de Coleta e Organização de Objetos com YouBot
**Aluno:** Luis Felipe Cordeiro Sena
**Disciplina:** MATA64 - Inteligência Artificial - UFBA
**Professor:** Luciano Oliveira
**Semestre:** 2025.2

---

## Índice
1. [Referências Essenciais (Top 10)](#1-referências-essenciais-top-10)
2. [Inteligência Artificial e Deep Learning](#2-inteligência-artificial-e-deep-learning)
3. [Redes Neurais para LIDAR](#3-redes-neurais-para-lidar)
4. [CNNs para Detecção de Objetos](#4-cnns-para-detecção-de-objetos)
5. [Lógica Fuzzy](#5-lógica-fuzzy)
6. [Robótica Móvel e Navegação](#6-robótica-móvel-e-navegação)
7. [KUKA YouBot e Cinemática](#7-kuka-youbot-e-cinemática)
8. [SLAM e Localização](#8-slam-e-localização)
9. [Manipulação e Grasping](#9-manipulação-e-grasping)
10. [Sistemas Neuro-Fuzzy](#10-sistemas-neuro-fuzzy)
11. [Frameworks e Ferramentas](#11-frameworks-e-ferramentas)
12. [Bases de Dados e Recursos](#12-bases-de-dados-e-recursos)

---

## 1. Referências Essenciais (Top 10)

### Citar obrigatoriamente na apresentação:

**[1] GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A.** *Deep Learning*. MIT Press, 2016.
- **Aplicação:** Fundamentos teóricos de redes neurais e CNNs
- **Disponível:** https://www.deeplearningbook.org/

**[2] ZADEH, L. A.** Fuzzy Sets. *Information and Control*, v. 8, n. 3, p. 338-353, 1965.
- **Aplicação:** Base teórica de lógica fuzzy para controle

**[3] MAMDANI, E. H.; ASSILIAN, S.** An experiment in linguistic synthesis with a fuzzy logic controller. *International Journal of Man-Machine Studies*, v. 7, n. 1, p. 1-13, 1975.
- **Aplicação:** Controlador fuzzy Mamdani para navegação do robô

**[4] QI, C. R.; SU, H.; MO, K.; GUIBAS, L. J.** PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, p. 652-660, 2017.
- **Aplicação:** Arquitetura base para processar dados LIDAR

**[5] REDMON, J.; DIVVALA, S.; GIRSHICK, R.; FARHADI, A.** You Only Look Once: Unified, Real-Time Object Detection. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, p. 779-788, 2016.
- **Aplicação:** Detecção em tempo real de cubos coloridos

**[6] BISCHOFF, R.; HUGGENBERGER, U.; PRASSLER, E.** KUKA youBot - a mobile manipulator for research and education. *IEEE International Conference on Robotics and Automation (ICRA)*, p. 1-4, 2011.
- **Aplicação:** Especificações técnicas da plataforma YouBot

**[7] THRUN, S.; BURGARD, W.; FOX, D.** *Probabilistic Robotics*. MIT Press, 2005.
- **Aplicação:** Navegação probabilística e mapeamento

**[8] TAHERI, H.; QIAO, B.; GHAEMINEZHAD, N.** Kinematic model of a four mecanum wheeled mobile robot. *International Journal of Computer Applications*, v. 113, n. 3, p. 6-9, 2015.
- **Aplicação:** Cinemática das rodas mecanum do YouBot

**[9] SAFFIOTTI, A.** The uses of fuzzy logic in autonomous robot navigation. *Soft Computing*, v. 1, n. 4, p. 180-197, 1997.
- **Aplicação:** Navegação autônoma com lógica fuzzy

**[10] CRAIG, J. J.** *Introduction to Robotics: Mechanics and Control*. 3. ed. Pearson, 2005.
- **Aplicação:** Cinemática inversa do braço 5-DOF

---

## 2. Inteligência Artificial e Deep Learning

### Livros-Texto Fundamentais

**RUSSELL, S.; NORVIG, P.** *Artificial Intelligence: A Modern Approach*. 4. ed. Pearson, 2021.
- Capítulos 18-20: Machine Learning e Deep Learning
- Base teórica sobre agentes inteligentes

**GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A.** *Deep Learning*. MIT Press, 2016.
- Capítulo 6: Deep Feedforward Networks (MLP)
- Capítulo 9: Convolutional Networks (CNN)
- Capítulo 11: Practical Methodology

**MURPHY, R. R.** *Introduction to AI Robotics*. 2. ed. MIT Press, 2019.
- Integração de IA em sistemas robóticos

---

## 3. Redes Neurais para LIDAR

### Processamento de Nuvens de Pontos

**QI, C. R.; SU, H.; MO, K.; GUIBAS, L. J.** PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *IEEE CVPR*, p. 652-660, 2017.
- **Aplicação direta:** Processar varreduras LIDAR para detecção de obstáculos
- **Conceito-chave:** Redes que processam point clouds não ordenados

**QI, C. R.; YI, L.; SU, H.; GUIBAS, L. J.** PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS*, 2017.
- **Aplicação:** Versão melhorada com hierarquia espacial

**ZHOU, Y.; TUZEL, O.** VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. *IEEE CVPR*, p. 4490-4499, 2018.
- **Aplicação:** Conversão de LIDAR para voxels + CNN 3D
- **Vantagem:** End-to-end learning para detecção

**YAN, Y.; MAO, Y.; LI, B.** SECOND: Sparsely Embedded Convolutional Detection. *Sensors*, v. 18, n. 10, 3337, 2018.
- **Aplicação:** Detecção eficiente em point clouds esparsos
- **Performance:** Mais rápido que VoxelNet

**CHEN, X.; MA, H.; WAN, J.; LI, B.; XIA, T.** Multi-View 3D Object Detection Network for Autonomous Driving. *IEEE CVPR*, p. 1907-1915, 2017.
- **Aplicação:** Fusão LIDAR + câmera RGB
- **Conceito:** Multi-view feature fusion

---

## 4. CNNs para Detecção de Objetos

### Arquiteturas Fundamentais

**LECUN, Y.; BOTTOU, L.; BENGIO, Y.; HAFFNER, P.** Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, v. 86, n. 11, p. 2278-2324, 1998.
- **LeNet:** Primeira CNN bem-sucedida
- **Conceito:** Convolution, pooling, backpropagation

**KRIZHEVSKY, A.; SUTSKEVER, I.; HINTON, G. E.** ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*, v. 25, p. 1097-1105, 2012.
- **AlexNet:** Marco do deep learning moderno
- **Conceito:** ReLU, dropout, data augmentation

**SIMONYAN, K.; ZISSERMAN, A.** Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*, 2015.
- **VGGNet:** Blocos convolucionais profundos
- **Aplicação:** Transfer learning para detecção de cores

**HE, K.; ZHANG, X.; REN, S.; SUN, J.** Deep Residual Learning for Image Recognition. *IEEE CVPR*, p. 770-778, 2016.
- **ResNet:** Skip connections para treinar redes profundas
- **Aplicação:** Backbone para detecção de objetos

### Detecção em Tempo Real

**REDMON, J.; DIVVALA, S.; GIRSHICK, R.; FARHADI, A.** You Only Look Once: Unified, Real-Time Object Detection. *IEEE CVPR*, p. 779-788, 2016.
- **YOLO:** Single-shot detector para tempo real
- **Aplicação direta:** Detectar cubos coloridos com baixa latência
- **FPS:** ~45 frames/segundo

**REDMON, J.; FARHADI, A.** YOLO9000: Better, Faster, Stronger. *IEEE CVPR*, 2017.
- **YOLOv2:** Melhorias em precisão e velocidade
- **Batch normalization, anchor boxes**

**REDMON, J.; FARHADI, A.** YOLOv3: An Incremental Improvement. *arXiv:1804.02767*, 2018.
- **YOLOv3:** Multi-scale predictions
- **Aplicação:** Detectar cubos de diferentes distâncias

**LIU, W.; ANGUELOV, D.; ERHAN, D.; et al.** SSD: Single Shot MultiBox Detector. *ECCV*, p. 21-37, 2016.
- **SSD:** Melhor para objetos pequenos (cubos!)
- **Múltiplas escalas de feature maps**

**REN, S.; HE, K.; GIRSHICK, R.; SUN, J.** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS*, v. 28, p. 91-99, 2015.
- **Faster R-CNN:** Alta precisão (trade-off com velocidade)
- **RPN:** Region Proposal Networks

**LIN, T. Y.; GOYAL, P.; GIRSHICK, R.; HE, K.; DOLLÁR, P.** Focal Loss for Dense Object Detection. *IEEE ICCV*, p. 2980-2988, 2017.
- **RetinaNet:** Lida com class imbalance
- **Focal loss:** Foca em exemplos difíceis

### Classificação de Cores

**KHAN, R.; VAN DE WEIJER, J.; KHAN, F. S.; et al.** Discriminative Color Descriptors. *IEEE CVPR*, p. 2866-2873, 2013.
- **Aplicação:** Classificação robusta de cores RGB
- **Histogramas de cores invariantes**

---

## 5. Lógica Fuzzy

### Fundamentos Teóricos

**ZADEH, L. A.** Fuzzy Sets. *Information and Control*, v. 8, n. 3, p. 338-353, 1965.
- **Artigo seminal:** Introdução dos conjuntos fuzzy
- **Aplicação:** Base para todas as regras fuzzy do projeto

**MAMDANI, E. H.; ASSILIAN, S.** An experiment in linguistic synthesis with a fuzzy logic controller. *Int. Journal of Man-Machine Studies*, v. 7, n. 1, p. 1-13, 1975.
- **Controlador Mamdani:** Tipo mais usado em robótica
- **Aplicação direta:** Controle de velocidade e direção

**TAKAGI, T.; SUGENO, M.** Fuzzy identification of systems and its applications to modeling and control. *IEEE Trans. SMC*, SMC-15(1), p. 116-132, 1985.
- **Controlador Sugeno:** Alternativa ao Mamdani
- **Saídas lineares:** Mais eficiente computacionalmente

### Navegação de Robôs

**SAFFIOTTI, A.** The uses of fuzzy logic in autonomous robot navigation. *Soft Computing*, v. 1, n. 4, p. 180-197, 1997.
- **Review completo:** Aplicações de fuzzy em navegação
- **Aplicação:** Estratégias para evitar obstáculos

**BEOM, H. R.; CHO, H. S.** A sensor-based navigation for a mobile robot using fuzzy logic and reinforcement learning. *IEEE Trans. SMC*, v. 25, n. 3, p. 464-477, 1995.
- **Fusão de sensores:** LIDAR + outros sensores
- **Fuzzy + RL:** Aprendizado adaptativo

**ANTONELLI, G.; CHIAVERINI, S.; FUSCO, G.** A fuzzy-logic-based approach for mobile robot path tracking. *IEEE Trans. Fuzzy Systems*, v. 15, n. 2, p. 211-221, 2007.
- **Path tracking:** Seguir trajetórias planejadas
- **Aplicação:** Aproximação suave de objetivos

**OMRANE, H.; MASMOUDI, M. S.; MASMOUDI, M.** Fuzzy Logic Based Control for Autonomous Mobile Robot Navigation. *Computational Intelligence and Neuroscience*, 2016, 9548482.
- **Evitação dinâmica:** Obstáculos móveis
- **Regras fuzzy:** Distância, ângulo, velocidade

### Livros de Referência

**ROSS, T. J.** *Fuzzy Logic with Engineering Applications*. 3. ed. Wiley, 2010.
- **Capítulo 6:** Fuzzy Control Systems
- **Exemplos práticos** de implementação

**PASSINO, K. M.; YURKOVICH, S.** *Fuzzy Control*. Addison-Wesley, 1998.
- **Teoria e prática:** Design de controladores fuzzy

---

## 6. Robótica Móvel e Navegação

### Livros-Texto

**SIEGWART, R.; NOURBAKHSH, I. R.; SCARAMUZZA, D.** *Introduction to Autonomous Mobile Robots*. 2. ed. MIT Press, 2011.
- **Capítulos 3-4:** Cinemática e localização
- **Capítulo 5:** Navegação e path planning
- **Referência completa** para robôs móveis

**THRUN, S.; BURGARD, W.; FOX, D.** *Probabilistic Robotics*. MIT Press, 2005.
- **Capítulo 5:** Robot Motion (cinemática)
- **Capítulo 6:** Robot Perception (sensores)
- **Capítulo 8:** Localization
- **Capítulo 10:** SLAM

**CORKE, P. I.** *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*. 2. ed. Springer, 2017.
- **Código prático:** MATLAB/Python
- **Visualizações:** Simulações e plots

### Artigos de Navegação

**TZAFESTAS, S. G.** Mobile robot control and navigation: A global overview. *Journal of Intelligent & Robotic Systems*, v. 91, n. 1, p. 35-58, 2018.
- **Survey:** Estado da arte em controle e navegação

**OLIVEIRA, L.; SANTOS, V.** Deep learning for mobile robotics: A survey. *Robotics and Autonomous Systems*, v. 133, 103642, 2020.
- **Deep learning:** Aplicações modernas em robótica móvel

---

## 7. KUKA YouBot e Cinemática

### Rodas Mecanum

**MUIR, P. F.; NEUMAN, C. P.** Kinematic modeling of wheeled mobile robots. *Journal of Robotic Systems*, v. 4, n. 2, p. 281-340, 1987.
- **Fundamento teórico:** Cinemática de robôs com rodas
- **Modelo geral:** Diferentes configurações

**TAHERI, H.; QIAO, B.; GHAEMINEZHAD, N.** Kinematic model of a four mecanum wheeled mobile robot. *Int. Journal of Computer Applications*, v. 113, n. 3, p. 6-9, 2015.
- **Aplicação direta:** Modelo cinemático do YouBot
- **Equações:** Velocidades das rodas → velocidade do robô

**DIEGEL, O.; BADVE, A.; BRIGHT, G.; et al.** Improved mecanum wheel design for omni-directional robots. *Australasian Conf. on Robotics and Automation*, p. 117-121, 2002.
- **Design:** Otimizações mecânicas
- **Controle:** Estratégias para movimento omnidirecional

### KUKA YouBot

**BISCHOFF, R.; HUGGENBERGER, U.; PRASSLER, E.** KUKA youBot - a mobile manipulator for research and education. *IEEE ICRA*, p. 1-4, 2011.
- **Artigo oficial:** Especificações completas
- **Aplicação educacional:** Ideal para pesquisa

**KECSKEMÉTHY, A.; WEINBERG, A.** An improved elasto-kinetostatic model of the KUKA YouBot and its force control. *IFToMM Symposium*, p. 471-479, 2014.
- **Modelo detalhado:** Elasticidade e cinemática
- **Controle de força:** Manipulação delicada

---

## 8. SLAM e Localização

### Fundamentos de SLAM

**DURRANT-WHYTE, H.; BAILEY, T.** Simultaneous localization and mapping: Part I. *IEEE Robotics & Automation Magazine*, v. 13, n. 2, p. 99-110, 2006.
- **Tutorial fundamental:** Conceitos básicos de SLAM
- **Formulação matemática:** EKF-SLAM

**BAILEY, T.; DURRANT-WHYTE, H.** Simultaneous localization and mapping (SLAM): Part II. *IEEE Robotics & Automation Magazine*, v. 13, n. 3, p. 108-117, 2006.
- **Continuação:** Estado da arte em 2006

### Visual SLAM

**MUR-ARTAL, R.; MONTIEL, J. M. M.; TARDÓS, J. D.** ORB-SLAM: A versatile and accurate monocular SLAM system. *IEEE Trans. Robotics*, v. 31, n. 5, p. 1147-1163, 2015.
- **Monocular SLAM:** Usando apenas uma câmera
- **Features ORB:** Oriented FAST and Rotated BRIEF

**MUR-ARTAL, R.; TARDÓS, J. D.** ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. *IEEE Trans. Robotics*, v. 33, n. 5, p. 1255-1262, 2017.
- **Extensão:** Múltiplos tipos de câmera
- **Open-source:** Código disponível

### LIDAR SLAM

**HESS, W.; KOHLER, D.; RAPP, H.; ANDOR, D.** Real-time loop closure in 2D LIDAR SLAM. *IEEE ICRA*, p. 1271-1278, 2016.
- **Google Cartographer:** SLAM 2D eficiente
- **Loop closure:** Correção de drift

**ZHANG, J.; SINGH, S.** LOAM: Lidar Odometry and Mapping in Real-time. *Robotics: Science and Systems (RSS)*, v. 2, p. 9, 2014.
- **LIDAR 3D:** Odometria precisa
- **Real-time:** Baixa latência

---

## 9. Manipulação e Grasping

### Cinemática de Manipuladores

**CRAIG, J. J.** *Introduction to Robotics: Mechanics and Control*. 3. ed. Pearson, 2005.
- **Capítulos 2-3:** Cinemática direta e inversa
- **Capítulo 4:** Jacobiano e velocidades
- **Aplicação direta:** Braço 5-DOF do YouBot

**SICILIANO, B.; SCIAVICCO, L.; VILLANI, L.; ORIOLO, G.** *Robotics: Modelling, Planning and Control*. Springer, 2009.
- **Livro completo:** Modelagem e controle
- **Mobile manipulators:** Capítulo específico

### Planejamento de Grasps

**BOHG, J.; MORALES, A.; ASFOUR, T.; KRAGIC, D.** Data-driven grasp synthesis—A survey. *IEEE Trans. Robotics*, v. 30, n. 2, p. 289-309, 2014.
- **Survey completo:** Métodos de grasp synthesis
- **Analítico vs data-driven**

**MAHLER, J.; LIANG, J.; NIYAZ, S.; et al.** Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics. *RSS*, 2017.
- **Deep learning:** Planejamento de grasps
- **Dados sintéticos:** Treinamento eficiente

### Visual Servoing

**CHAUMETTE, F.; HUTCHINSON, S.** Visual servo control. Part I: Basic approaches. *IEEE Robotics & Automation Magazine*, v. 13, n. 4, p. 82-90, 2006.
- **Image-based vs position-based:** Duas abordagens
- **Aplicação:** Ajuste fino para pegar cubos

**CORKE, P.; HUTCHINSON, S.** A new partitioned approach to image-based visual servo control. *IEEE Trans. Robotics and Automation*, v. 17, n. 4, p. 507-515, 2001.
- **Hybrid approach:** Combina vantagens das duas abordagens

---

## 10. Sistemas Neuro-Fuzzy

### Arquiteturas Híbridas

**JANG, J. S.** ANFIS: Adaptive-network-based fuzzy inference system. *IEEE Trans. SMC*, v. 23, n. 3, p. 665-685, 1993.
- **ANFIS:** Fuzzy + backpropagation
- **Aplicação:** Ajuste automático de funções de pertinência

**LIN, C. T.; LEE, C. G.** *Neural Fuzzy Systems: A Neuro-Fuzzy Synergism to Intelligent Systems*. Prentice Hall, 1996.
- **Livro-texto:** Sistemas neuro-fuzzy
- **Teoria e aplicações**

**TUNSTEL, E.; LIPPINCOTT, T.; JAMSHIDI, M.** Behavior hierarchy for autonomous mobile robots: Fuzzy-behavior modulation and evolution. *Int. Journal of Intelligent Automation and Soft Computing*, v. 3, n. 1, p. 37-49, 1997.
- **Hierarquia de comportamentos:** Fuzzy modulation
- **Evolução:** Genetic algorithms + fuzzy

### Aplicações em Robótica

**ER, M. J.; DENG, C.** Obstacle avoidance of a mobile robot using hybrid learning approach. *IEEE Trans. Industrial Electronics*, v. 51, n. 3, p. 677-686, 2004.
- **Híbrido:** Neural + fuzzy para obstáculos
- **Online learning:** Adaptação em tempo real

---

## 11. Frameworks e Ferramentas

### Deep Learning Frameworks

**ABADI, M.; AGARWAL, A.; BARHAM, P.; et al.** TensorFlow: Large-scale machine learning on heterogeneous systems. *arXiv:1603.04467*, 2016.
- **TensorFlow:** Framework do Google
- **Disponível:** https://www.tensorflow.org/

**PASZKE, A.; GROSS, S.; MASSA, F.; et al.** PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*, v. 32, p. 8024-8035, 2019.
- **PyTorch:** Framework mais usado em pesquisa
- **Disponível:** https://pytorch.org/

### Machine Learning Clássico

**PEDREGOSA, F.; VAROQUAUX, G.; GRAMFORT, A.; et al.** Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825-2830, 2011.
- **scikit-learn:** Biblioteca completa de ML
- **scikit-fuzzy:** Extensão para lógica fuzzy

### Simulação

**MICHEL, O.** Cyberbotics Ltd. Webots™: Professional mobile robot simulation. *Int. Journal of Advanced Robotic Systems*, v. 1, n. 1, p. 39-42, 2004.
- **Webots:** Simulador usado no projeto
- **Documentação:** https://cyberbotics.com/doc/

**KOENIG, N.; HOWARD, A.** Design and use paradigms for Gazebo, an open-source multi-robot simulator. *IEEE IROS*, v. 3, p. 2149-2154, 2004.
- **Gazebo:** Alternativa ao Webots
- **Comparação:** Princípios de design

---

## 12. Bases de Dados e Recursos

### Datasets

**DENG, J.; DONG, W.; SOCHER, R.; et al.** ImageNet: A large-scale hierarchical image database. *IEEE CVPR*, p. 248-255, 2009.
- **ImageNet:** 14 milhões de imagens
- **Transfer learning:** Modelos pré-treinados

**GEIGER, A.; LENZ, P.; URTASUN, R.** Are we ready for autonomous driving? The KITTI vision benchmark suite. *IEEE CVPR*, p. 3354-3361, 2012.
- **KITTI:** LIDAR + câmera para carros autônomos
- **Benchmark:** Avaliação de algoritmos

### Repositórios Open-Source

1. **ORB-SLAM2:** https://github.com/raulmur/ORB_SLAM2
2. **Google Cartographer:** https://github.com/cartographer-project/cartographer
3. **YOLO (Darknet):** https://github.com/AlexeyAB/darknet
4. **Ultralytics YOLOv8:** https://github.com/ultralytics/ultralytics
5. **PyTorch:** https://github.com/pytorch/pytorch
6. **scikit-fuzzy:** https://github.com/scikit-fuzzy/scikit-fuzzy
7. **Webots:** https://github.com/cyberbotics/webots

### Bases Acadêmicas

- **IEEE Xplore:** https://ieeexplore.ieee.org/
- **Google Scholar:** https://scholar.google.com/
- **arXiv (cs.RO):** https://arxiv.org/list/cs.RO/recent
- **ScienceDirect:** https://www.sciencedirect.com/
- **ACM Digital Library:** https://dl.acm.org/
- **Springer:** https://link.springer.com/

### Termos de Busca

- "mobile robot fuzzy control"
- "CNN object detection robotics real-time"
- "LIDAR neural network obstacle detection"
- "mecanum wheel kinematics control"
- "SLAM without GPS indoor"
- "visual servoing object grasping"
- "autonomous navigation fuzzy logic"
- "youbot kuka mobile manipulator"
- "point cloud deep learning"
- "neuro-fuzzy hybrid control"

---

## Estratégia de Citação na Apresentação

### Slides Recomendados

**Slide 1: Fundamentação Teórica - Deep Learning**
- Goodfellow et al. (2016): Conceitos de CNNs
- LeCun et al. (1998): Arquitetura convolucional
- Krizhevsky et al. (2012): Transfer learning

**Slide 2: Processamento LIDAR**
- Qi et al. (2017): PointNet para point clouds
- Zhou & Tuzel (2018): VoxelNet para detecção 3D
- Diagrama da arquitetura adaptada

**Slide 3: Detecção de Cubos Coloridos**
- Redmon et al. (2016): YOLO para tempo real
- Liu et al. (2016): SSD para objetos pequenos
- Demonstração visual da detecção

**Slide 4: Lógica Fuzzy para Controle**
- Zadeh (1965): Conjuntos fuzzy
- Mamdani & Assilian (1975): Controlador fuzzy
- Saffiotti (1997): Navegação autônoma
- Diagrama das regras fuzzy

**Slide 5: Plataforma YouBot**
- Bischoff et al. (2011): Especificações do YouBot
- Taheri et al. (2015): Cinemática mecanum
- Modelo 3D do robô com sensores

**Slide 6: Navegação e Mapeamento**
- Thrun et al. (2005): Robótica probabilística
- Se usar SLAM: Durrant-Whyte & Bailey (2006)
- Mapa construído pelo robô

**Slide 7: Manipulação**
- Craig (2005): Cinemática inversa do braço
- Bohg et al. (2014): Planejamento de grasps
- Sequência de pegada do cubo

**Slide 8: Arquitetura do Sistema**
- Diagrama completo: Sensores → Percepção → Controle → Atuação
- Citações em cada módulo

### Formato de Legendas

```
Figura 1: Arquitetura PointNet (QI et al., 2017) adaptada para
processamento de varreduras LIDAR 2D do YouBot.

Figura 2: Sistema de inferência fuzzy Mamdani (MAMDANI; ASSILIAN, 1975)
para controle de velocidade linear e angular.

Figura 3: Modelo cinemático do YouBot com rodas mecanum
(BISCHOFF et al., 2011; TAHERI et al., 2015).

Figura 4: Pipeline de detecção YOLO (REDMON et al., 2016) para
identificação de cubos coloridos em tempo real.
```

### Citações em Texto (sem código!)

- Use diagramas de blocos com equações matemáticas
- Gráficos de performance (precisão, tempo)
- Vídeos do robô executando tarefas
- Visualizações 3D do ambiente mapeado
- Tabelas comparativas de abordagens

**IMPORTANTE:** Não mostre código-fonte (perda de 3-10 pontos)

---

## BibTeX para LaTeX (opcional)

```bibtex
@book{goodfellow2016deep,
  title={Deep Learning},
  author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  year={2016},
  publisher={MIT Press}
}

@article{zadeh1965fuzzy,
  title={Fuzzy sets},
  author={Zadeh, Lotfi A},
  journal={Information and control},
  volume={8},
  number={3},
  pages={338--353},
  year={1965}
}

@article{mamdani1975fuzzy,
  title={An experiment in linguistic synthesis with a fuzzy logic controller},
  author={Mamdani, Ebrahim H and Assilian, Sedrak},
  journal={International journal of man-machine studies},
  volume={7},
  number={1},
  pages={1--13},
  year={1975}
}

@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={652--660},
  year={2017}
}

@inproceedings{redmon2016yolo,
  title={You only look once: Unified, real-time object detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={779--788},
  year={2016}
}

@inproceedings{bischoff2011kuka,
  title={KUKA youBot-a mobile manipulator for research and education},
  author={Bischoff, Rainer and Huggenberger, Ulrich and Prassler, Erwin},
  booktitle={IEEE International Conference on Robotics and Automation},
  pages={1--4},
  year={2011}
}

@article{taheri2015kinematic,
  title={Kinematic model of a four mecanum wheeled mobile robot},
  author={Taheri, Hamid and Qiao, Bing and Ghaeminezhad, Nurallah},
  journal={International Journal of Computer Applications},
  volume={113},
  number={3},
  pages={6--9},
  year={2015}
}

@book{thrun2005probabilistic,
  title={Probabilistic robotics},
  author={Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter},
  year={2005},
  publisher={MIT press}
}

@book{craig2005introduction,
  title={Introduction to robotics: mechanics and control},
  author={Craig, John J},
  edition={3},
  year={2005},
  publisher={Pearson}
}

@article{saffiotti1997fuzzy,
  title={The uses of fuzzy logic in autonomous robot navigation},
  author={Saffiotti, Alessandro},
  journal={Soft Computing},
  volume={1},
  number={4},
  pages={180--197},
  year={1997}
}
```

---

**Observação Final:** Este documento contém referências criterosas e peer-reviewed, organizadas por relevância ao projeto. Priorize as Top 10 na apresentação e use as demais para embasamento técnico durante implementação.
