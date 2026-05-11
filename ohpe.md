# Occluded human pose estimation based on part-aware discrete diffusion priors

Hongyu Xiao a,b ,∗, Hui He a,b, Yifan Xie c , Yi Zheng a,b 

a Faculty of Arts and Sciences, Beijing Normal University, Zhuhai, Guangdong, China 

b Advanced Institute of Natural Sciences, Beijing Normal University, Zhuhai, Guangdong, China 

c University of Glasgow, Glasgow, Scotland, UK 

# A R T I C L E I N F O

Keywords: 

Human pose estimation 

Human pose prior 

Hierarchical human pose prior 

Conditional discrete diffusion model 

Multimodal conditional 

VQ-VAE 

# A B S T R A C T

In this work, we focus on reconstructing human poses from RGB images, with particular attention given to the ambiguity issues caused by complex scenes such as occlusions. The main challenges we face are twofold: how to reconstruct a complete pose based on limited visible cues and how to handle the uncertainty of occluded parts. To address these issues, our primary approach is to leverage human prior knowledge to ensure the physical plausibility of the reconstructed pose and simulate occluded scenarios through the forward process of the diffusion model, followed by recovering the occluded parts through the reverse process. Specifically, we first train hierarchical encoders, codebooks, and decoders to learn rich pose prior knowledge and then incorporate these priors into a discrete diffusion model with multimodal guidance. We train the network to gradually predict clean discrete pose tokens that are consistent with prior knowledge and ultimately decode them into complete body poses. Extensive experimental results on the COCO and 3DMPB datasets demonstrate that our method achieves state-of-the-art performance compared with previous approaches. The code will be publicly available. 

# 1. Introduction

Human pose estimation, which recognizes and locates human body keypoints from RGB images, is a fundamental computer vision task. It has extensive applications in fields such as medical rehabilitation [1–4], augmented reality [5], sports analysis [6,7], and autonomous driving [8]. In recent years, although deep learning-based pose estimation methods [9–21] have made significant progress on standard datasets, existing methods still face challenges in real-world scenarios, especially under severe occlusion or complex lighting conditions: how to reconstruct a complete pose based on limited visible cues, and how to effectively handle the uncertainty of occluded poses. 

Deep learning-based methods often enhance feature representation capabilities by improving network architectures [22–24] or by incorporating long-range dependency modeling based on Transformer [25–30]. However, these methods often lack explicit modeling of the intrinsic physical constraints of poses, leading to the generation of unreasonable poses in occluded scenarios. As shown in Fig. 1, existing deep learningbased methods [17,25] often predict results that do not conform to the laws of kinematics in occluded scenarios. Some studies have built human priors based on VQ-VAE [31] to constrain the rationality of poses, but they tend to output single deterministic predictions [32–35], which are insufficient in considering the diversity of occluded poses. 

In this work, we propose a framework that integrates part-aware hierarchical priors and multimodal conditional discrete diffusion models to address the above issues from the two dimensions of structured modeling and multimodal uncertainty. Human priors are defined here as explicit encodings of the topological structure of human poses, kinematic constraints of joints, and part-global dependencies. To promote the ability of the network to estimate complete body poses from partially visible observations in occluded images, we first learn a hierarchical prior to encode body parts with multiple codebooks, which captures the dependencies between different body limbs and improves robustness in occluded scenes. In addition, we introduce a multimodal conditional discrete diffusion model to consider the uncertainty and indeterminacy of pose estimation under occlusions. The model uses a reverse diffusion mechanism to gradually predict clean discrete prior tokens in latent space, which is then decoded to a complete body pose with the pretrained decoder. To further increase the accuracy of pose prediction and reduce its uncertainty and ambiguity, we integrate interactive information closely related to pose prediction, extracted by the CLIP image encoder and CLIP text encoder, on top of the image conditional features. This information collectively forms a multimodal condition that effectively guides the learning process of the discrete diffusion model. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/3118f8ae24d21205bdccd0b31824016e2603c8a438965a7d97340c2aab5dd588.jpg)



Fig. 1. Deep learning-based human pose estimation [17,25] tends to predict unrealistic poses under occlusions. Our method integrates human prior knowledge to constrain pose distributions and enhance adaptability through a discrete diffusion process, maintaining high accuracy and robustness even in heavily occluded scenes with complex lighting.


Specifically, we first design a hierarchical discrete prior model based on VQ-VAE-2 [36] to learn the dependencies of ?? body parts (e.g., head, arms, legs, and full body). As shown in Fig. 2, ?? decoders map different local parts to ?? token features, with each token feature encoding a substructure of body pose, which are then quantified into ?? discrete indices by the regional codebooks. The quantified features of substructures are subsequently concatenated with global features and then fed into a global codebook. The features from the global codebook are decoded into a complete body pose. During training, the encoders, codebooks, and decoder are optimized jointly as learnable parameters by minimizing reconstruction and commitment loss [36, 37]. After training, the encoders and codebooks can learn dependencies of each body part and can be used as a prior in the pose estimation task [33,34,38]. To leverage the discrete prior for occluded human pose estimation, we further propose a multimodal conditional diffusion model to gradually predict clean indices that are consistent with the trained prior encoders from a single RGB image. We adopt a discrete diffusion model for the denoising process, which has been proven to be effective for discrete representation learning [39–41]. During the training phase, an occluded pose is encoded by the prior encoder to obtain ?? discrete indices. We then randomly mask some dimensions of the indices according to a sampled timestep and force the diffusion model to recover clean indices from the noisy indices guided by image features. In the reverse process, we start from pure noise and take several timesteps to predict the clean indices with the trained diffusion model. Subsequently, we utilize a pretrained CLIP model to extract interactive information closely related to human pose estimation and combine it with image features extracted by the Swin-Base backbone network, thereby providing the model with a deeper contextual understanding capability. Finally, we use the trained decoder of the prior to generate the complete body pose from the denoised indices. Our method has been extensively evaluated on COCO and 3DMPB datasets, achieving comparable results to those of the state-of-the-art methods, providing a new perspective for addressing human posture estimation issues in occluded scenes. Our contributions are summarized as follows: 

• We propose a framework to consider the uncertainty of human poses in occlusion scenarios and produce multiple plausible body poses from partially observed images. 

• We propose a hierarchical part-aware discrete prior to learn skeletal topologies to recover complete body poses conditioned on visible parts. 

• We develop a discrete diffusion model to leverage learned body prior knowledge and multimodal combination conditions to assist in occluded human pose estimation with latent code denoising. 

# 2. Related work

# 2.1. Single-person pose estimation

Single-person pose estimation predicts body poses from a single image. Recent advances in the use of neural networks have improved accuracy through data augmentation [11,42,43], model architecture [17, 44], postprocessing [45,46], and supervised representation [47]. Key contributions include keypoint masking for occlusions [22], adversarial augmentation [48], and generating challenging samples with a semantic body part pool [49]. Model architectures such as preserving high-resolution features [17], AutoPose [50], and stacked hourglass networks [19,51] improved feature fusion. DARTS [52] optimized architecture search. Postprocessing methods such as distribution-aware decoding [53,54] enhance keypoint accuracy. Refinement models [55] and supervised methods, such as CPM [56] and binary heatmaps [57], reduce errors and improve precision [58]. Although these works have achieved good results in common situations, the results are not good for partially occluded images. In contrast, our prior learns real-world prototypes to acquire pose prior knowledge, thereby restricting pose configurations to physically plausible ranges and effectively enhancing accuracy in occluded scenes. 

# 2.2. Multiperson pose estimation

Multiperson pose estimation (MPPE) aims to locate the keypoints of each person’s body accurately in an image. Two primary strategies are employed in this domain: two-stage and one-stage approaches. The two-stage approach can be further divided into top-down and bottom-up methods. The top-down method first uses a human detector to identify individuals and then predicts keypoints for each detected instance, exemplified by RMPE [59], CPN [60], Simple Baselines [61], HRNet [17] and AlphaPose [62]. Among them, CPN [60] and HR-Net [17] enhance recognition capabilities through innovative network designs. The bottom-up approach, on the other hand, first detects all possible body parts in an image and then groups them into individual skeletons. Research in this area includes OpenPose [63], HigherHR-Net [64], and DERK [65]. The PAFs technology of OpenPose [63] leads the trend, and higher HRNET [64] addresses the problem of scale change. One-stage approaches, such as SPM [66], Directpose [67], GroupPose [26], and PETR [68], directly predict poses from images, bypassing complex intermediate processing and resulting in simplified training and computational efficiency. Occlusion in MPPE remains a significant challenge. Although existing methods achieve satisfactory results on standard datasets, they often predict inaccurate poses for complex postures and cluttered backgrounds in occluded scenes. Our framework explicitly considers occlusion by adopting a masking and replacing strategy during training. By utilizing the forward and reverse processes of the diffusion model, the framework learns to inpaint an incomplete pose, enhancing its understanding of complex scenes and effectively improving robustness in occluded scenarios. 

The accuracy of human pose estimation is enhanced by integrating knowledge from anatomy, physiology, and kinematics, focusing mainly on learning the unconditional distribution of plausible poses [69, 70] and conditional priors under specific conditions [32,33,38,47,71]. Some studies [32,33,38,70] utilize VAE [72] to construct priors, enhancing the model’s ability to handle noise interference and incomplete observational data. GFPose [73] proposed a pose prior learning framework based on the score diffusion model, but its reverse sampling process may limit real-time applications. UVPrior [34] makes reasonable inferences about invisible body parts based on visible information. Pose-NDF [74] improves the ability to generate realistic new poses and accurately reconstruct poses from noisy or partial observations. Recent studies such as PCT [35] leverage VQ-VAE [31] to quantize poses, transform pose estimation into a classification problem and enhance model performance in occluded situations. However, PCT encodes learned poses into a single Codebook, limiting the integration of semantic information representing different substructures. In contrast, our prior effectively captures the hierarchical features and local details of poses by distinguishing between the regional and global layers of pose representation. We use multiple encoders and corresponding codebooks to learn and memorize representative pose prototypes, integrating these prototypes into a comprehensive and robust prior representation. This layered structure allows for more efficient retrieval of relevant prototypes in uncertain or ambiguous scenes, thereby enhancing the model’s robustness in occluded scenarios. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/764acc37f0e94a7f0b3c2072c2db659d88ad36a9106378682da89f5b5fee9c42.jpg)



Fig. 2. Overview of our method. Our framework consists of two components: the hierarchical discrete prior model and the multimodal conditional discrete diffusion model. The prior model encompasses regional and global layers, capturing rich prior knowledge of human poses through hierarchical encoders, codebooks, and decoders. The discrete diffusion model integrates image features extracted by Swin-Base with text–image features extracted by CLIP, forming multimodal conditional inputs. In the forward propagation of the diffusion model, noise is gradually introduced via an occlusion-replacement strategy until pure noise is achieved. In the reverse process, we train a denoiser to leverage multimodal conditional features, randomly sampled time steps t, and human pose priors to progressively recover clear pose discrete indices, ultimately reconstructing the complete human pose through the decoder.


# 3. Method

# 3.1. Preliminaries

# 3.1.1. VQ-VAE

VQ-VAE (vector quantization variational autoencoder) [31] is a deep learning model that combines vector quantization technology to learn the discrete latent representation of data. It is composed of three model components: an encoder, a quantizer, and a decoder. The encoder is responsible for mapping the input data ?? to a representation $z _ { e } ( x )$ in the continuous latent space. The discretization component maps $z _ { e } ( x )$ to the nearest embedding vector $\boldsymbol { e } _ { k }$ , forming a discrete latent variable ??. The discretization process can be expressed as: 

$$
z _ {q} (x) = e _ {k}, \quad \text { where } \quad k = \operatorname{argmin} _ {j} \left\| z _ {e} (x) - e _ {j} \right\| _ {2}. \tag {1}
$$

Here, ?? is the index that minimizes the Euclidean distance between $z _ { e } ( x )$ and $e _ { j } .$ . The decoder uses the discrete latent variable ?? to reconstruct the input data ??. 

The total loss of VQ-VAE consists of three parts: reconstruction loss, embedding space quantization loss, and commitment loss. The total loss can be formulated as: 

$$
\mathcal {L} = \log p (x | z _ {q} (x)) + \| \mathrm{sg} [ z _ {e} (x) ] - e \| _ {2} ^ {2} +
$$

$$
\beta \| z _ {e} (x) - \operatorname{sg} [ e ] \| _ {2} ^ {2}. \tag {2}
$$

The reconstruction loss, log ??(??|????(??)), measures the ability of the decoder to reconstruct the input data ?? on the basis of the discretized latent variable $z _ { q } ( x ) . \ p ( x | z _ { q } ( x ) )$ represents the probability of reconstructing data ?? given the latent variable $z _ { q } ( x )$ . The quantization loss in the embedding space, sg $\dot { z } _ { e } ( x ) ] - e | | _ { 2 } ^ { 2 } ;$ , uses the squared Euclidean distance to move the embedding vector $\overline { { \boldsymbol { e } } } _ { i }$ closer to the encoder output $z _ { e } ( x )$ . This loss term is only used to update the embedding space, enabling the embedding vectors to better represent the latent space. The commitment loss, $\beta \| z _ { e } ( x ) - s \mathbf { g } [ e ] \| _ { 2 } ^ { 2 } .$ , ensures that the encoder commits to its output in the embedding space, preventing the output from growing unboundedly. ?? is a hyperparameter used to balance this term with other loss terms. $s g$ denotes the stop-gradient operation, which is defined as an identity operation during forward propagation, and its derivative is 0 during backward propagation, thus preventing gradients from updating its operand. This means that the embedding vector ?? does not directly receive gradients from the reconstruction loss but is updated through the embedding vector loss. 

# 3.1.2. Diffusion model

The diffusion model (denoising diffusion probability model, DDPM) [75] is a generative model, with its key lying in its forward and reversed-phases. In the forward process, starting from a data distribution $q ( x _ { 0 } ) ,$ a series of increasingly noisy latent variables $x _ { 1 } , \dots , x _ { T }$ are generated by gradually adding Gaussian noise. Its forward process can be expressed as: 

$$
q (x _ {1: T} | x _ {0}) = \prod_ {t = 1} ^ {T} q (x _ {t} | x _ {t - 1}),
$$

$$
q (x _ {t} | x _ {t - 1}) = \mathcal {N} (x _ {t}; \sqrt {1 - \beta_ {t}} x _ {t - 1}, \beta_ {t}). \tag {3}
$$

Its reverse process can be expressed as: 

$$
p _ {\theta} (x _ {0}: T) = p (x _ {T}) \prod_ {t = 1} ^ {T - 1} p _ {\theta} (x _ {t - 1} | x _ {t}),
$$

$$
p _ {\theta} (x _ {t - 1} | x _ {t}) = \mathcal {N} (x _ {t - 1}; \mu_ {\theta} (x _ {t}, t), \sum_ {\theta} (x _ {t}, t)), \tag {4}
$$

where $\mu _ { \theta }$ and $\Sigma _ { \theta }$ are the mean and variance functions parameterized by a neural network. 

The forward process transforms data into a noise distribution by gradually adding noise, which can be achieved through a reparameterization trick [72]. This technique enables the separation of noise generation from model parameters, facilitating gradient computation. The reverse process is a parameterized Markov chain whose parameters are learned by minimizing the loss function. The variational lower bound (VLB) [76] serves as the core loss function. By minimizing VLB, the model can learn a reverse process that approximates the true data distribution as closely as possible, which is defined as: 

$$
\mathcal {L} = \sum_ {t = 1} ^ {T} \mathbb {E} _ {q (x _ {t} | x _ {0})} [ \log p _ {\theta} (x _ {t - 1} | x _ {t}) ] -
$$

$$
D _ {K L} (q (x _ {T} | x _ {0}) | | p (x _ {T})). \tag {5}
$$

 is the objective function optimized during the training process. The training objective can be formulated as follows: 

$$
\mathcal {L} _ {\text { simple }} (\theta) = \mathbb {E} _ {t, x _ {0}, \epsilon} [ \| \epsilon - \epsilon_ {\theta} (\sqrt {\overline {{\alpha}} _ {t}} x _ {0} + \sqrt {1 - \overline {{\alpha}} _ {t}} \epsilon , t) \| ^ {2} ]. \tag {6}
$$

where $\epsilon _ { \theta }$ is a network that predicts noise and where $\overline { { \alpha _ { t } } }$ is the complement of the cumulative noise variance. The sampling strategy to recover a sample $x _ { t - 1 }$ closer to the original data $x _ { 0 }$ from the noisy data $x _ { t }$ can be expressed as: 

$$
x _ {t - 1} = \frac {1}{\sqrt {\alpha_ {t}}} (x _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \overline {{\alpha_ {t}}}}} \epsilon_ {\theta} (x _ {t}, t)) + \sigma_ {t} z. \tag {7}
$$

where $z \sim \mathcal { N } ( 0 , I )$ is random noise. 

# 3.2. Part-aware occluded pose prior

When reconstructing 2D human poses from RGB images, complex scenes (such as occlusions) often lead to unreasonable predictions. Introducing human prior knowledge can improve accuracy. Some studies use VQ-VAE technology to learn prior knowledge through the collaboration of encoders, codebooks, and decoders. However, these methods are still limited in reconstructing fine-grained pose substructures, especially in multi-person or heavily occluded scenes. To address this challenge, we first propose an innovative part-aware pose prior to learn human pose prior knowledge for the occlusion problem. 

Our prior divides a complete pose into several layers (e.g., head, arms, legs, and global) to capture the detailed hierarchical features of human poses. With sufficient training data, it constructs a comprehensive and robust pose representation, which has two advantages: (1) It can accurately capture subtle changes in local poses, enhancing the model’s adaptability to complex scenes such as occlusions. (2) Iterative refinement can further improve the accuracy of predictions by adjusting the output results. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/0830d325cd041f4cdcf28fd607791959c5bd1d8bb39fc88fe382d495555864e8.jpg)


Specifically, a raw pose $P \in \mathbb { R } ^ { p \times 2 }$ is fed into a set of encoders to obtain token features $F = ( F _ { 1 } , \dots , F _ { x } , \dots , F _ { N } )$ , where ?? is the number of keypoints. $F _ { x }$ represents a series of token features generated by the ?? − ??ℎ encoder, 

$$
F _ {x} = (f _ {0}, \dots , f _ {i}, \dots , f _ {M}) = g _ {E n c x} (P _ {x}), \tag {8}
$$

where $f _ { i }$ denotes a substructure of posture composed of several interrelated joints, $g _ { E n c x }$ represents the ??−??ℎ learnable encoder, which consists of multiple MLP-Mixer blocks [77], and ?? represents the human pose. Moreover, we adopt ?? codebooks $C = ( C _ { 1 } , \dots , C _ { x } , \dots , C _ { N } ) , C _ { x } =$ $( c _ { 1 } , \ldots , c _ { x } , \ldots , c _ { V } ) ^ { T } .$ , where ?? is the number of codebook entries. In the embedding space, tokens are quantized through nearest neighbor lookup: 

$$
z _ {x} (f _ {i} = v \mid P _ {x}) = \left\{ \begin{array}{l l} 1 & \text { if } \quad v = \arg \min _ {j} \| f _ {i} - c _ {x j} \| _ {2} \\ 0 & \text { otherwise. } \end{array} \right. \tag {9}
$$

The term $z _ { x } ( f _ { i } )$ represents the index corresponding to the codebook entry, and the set of quantized tokens is represented as $K = ( K _ { 1 } , \ldots ,$ $K _ { x } , \ldots , K _ { N } ) ,$ , where $K _ { x }$ denotes the quantization result from the ?? − ??ℎ codebook. The token features $F _ { N } ,$ secured by the global encoder, merge with the tokens quantized by local codebooks postprocessing through a projection transformation layer. This ensemble is subsequently fed into the global codebook to yield the comprehensive pose quantization tokens $K _ { N } = ( c _ { z ( f _ { 0 } ) } , c _ { z ( f _ { 1 } ) } , \ldots , c _ { z ( f _ { M } ) } ) = ( k _ { 0 } , k _ { 1 } , \ldots , k _ { M } )$ . Ultimately, this series of quantized tokens is fed into the pose decoder to accurately reconstruct the original pose. 

$$
\overline {{{P}}} = g _ {D e c} (k _ {0}, k _ {1}, \dots , k _ {M}). \tag {10}
$$

$g _ { D e c }$ represents a learnable encoder that consists of an MLP-Mixer block. We train the encoders,codebooks and decoder jointly by minimizing the following loss over the training set: 

$$
\mathcal {L} = \operatorname{smooth} _ {\mathcal {L} _ {1}} (\overline {{{P}}}, P) + \beta \sum_ {i = 1} ^ {M} \left\| k _ {i} - \mathrm{sg} \left[ c _ {e (k _ {i})} \right] \right\| _ {2} ^ {2}, \tag {11}
$$

where ???? denotes the stopping gradient and where ?? is a hyperparameter. 

To address the potential issue of vanishing gradients effectively during the quantization process, we have adopted the optimization strategy proposed in [31], and we refine the update of the codebook via the exponential moving average (EMA) of previous token features. Our approach innovatively introduces ?? codebooks, each equipped with ?? trainable embedding vectors, thus establishing ?? distinct embedding spaces. This unique design endows our model with the robust ability to precisely extract subtle substructures of human poses from each codebook and integrate them, constructing a comprehensive and detailed prior knowledge system of keypoints, denoted as ?? . Furthermore, by leveraging the quantization mechanism of the global codebook coupled with the exceptional decoding capabilities of the decoder, we achieve efficient integration of local and global features. This not only enriches the dimensionality of feature expression but also ensures that the final ?? quantized token features provide a solid foundation for accurate pose recovery. 

# 3.3. Discrete diffusion model for occluded posel

Incorporating human prior knowledge is crucial for constraining pose predictions within a physically plausible range and enhancing the reliability of reconstructed poses. However, the occluded parts may correspond to multiple poses, increasing the complexity of the reconstruction. To address this, we draw on the latest advancements in diffusion models in image generation to design a multimodal conditional discrete diffusion model to address the issue of pose prediction uncertainty in occluded scenes. Specifically, we use previously trained human priors to obtain a discrete pose representation $K = ( k _ { 0 } , \ldots , k _ { x } , \ldots , k _ { M } ) ;$ , where $k _ { x }$ represents a posture substructure composed of joints, and integrate image features extracted by Swin-Base with visual and textual features extracted by the CLIP model to form multimodal conditional features. These priors and conditional features are input together into the denoiser network. In the forward process, we randomly sample a time step ?? in ?? to add noise to $K ,$ , resulting in $K _ { t } .$ . In the reverse process, we train the network to gradually denoise and restore $K .$ . In the testing phase, we start from pure noise $K _ { T } \mathrm { : }$ , use the well-trained denoiser, and, based on multimodal conditions, gradually denoise over ?? time steps to obtain ${ \overline { { K } } } ,$ and then use the pretrained pose decoder to obtain the estimated pose ${ \overline { { P } } } .$ The overall goal is to maximize the conditional transition distribution $q ( x | y )$ . Next, we provide a detailed introduction to this multimodal conditional discrete diffusion process and explore how to train the denoiser network to reverse this process. 

Discrete Diffusion Model. Our discrete diffusion model features two processes: forward and reverse processes. In the forward process, the initial token $k _ { 0 }$ is transformed into pure noise $k _ { t }$ after a series of noise-adding steps $t \in \{ 1 , 2 , \ldots , s , \ldots , T \}$ . 

Previous works [78] have shown that the probability of a token transitioning from $k _ { s - 1 }$ to $k _ { s }$ can be estimated via a state transition matrix $[ A _ { s } ] _ { m n } = q ( k _ { s } = m | k _ { s - 1 } = n ) \in \mathbb { R } ^ { \| C \| \times \| C \| } ;$ ; the detailed design of the matrix ?? will be thoroughly introduced in the following section. Therefore, the forward process can be formulated as: 

$$
q (k _ {s} | k _ {s - 1}) = u ^ {T} (k _ {s}) A _ {s} u (k _ {s - 1}), \tag {12}
$$

where $u ( \cdot )$ is a one-hot vector with a length of $V , k _ { s }$ is a multinomial distribution, and its probabilities are determined by the state $A _ { s } u ( k _ { s - 1 } )$ at time ????−1. $k _ { s - 1 } .$ 

Owing to the properties of Markov chains, we are able to ignore the intermediate processes and directly calculate the probability from $k _ { 0 }$ to any time step $k _ { s } \colon$ 

$$
q (k _ {s} | k _ {0}) = u ^ {T} (k _ {s}) \overline {{A}} _ {s} u (k _ {0}), \text { with } \overline {{A _ {s}}} = A _ {s} \dots A _ {1}. \tag {13}
$$

Furthermore, another significant feature is that by conditioning on $k _ { 0 } ,$ , we are able to effectively trace and analyze the posterior distribution of the diffusion process: 

$$
\begin{array}{l} q (k _ {s - 1} | k _ {s}, k _ {0}) = \frac {q (k _ {s} | k _ {s - 1} , k _ {0}) q (k _ {s - 1} | k _ {0})}{q (k _ {s} | k _ {0})} \\ = \frac {(u ^ {T} (k _ {s}) A _ {s} u (k _ {s - 1})) (u ^ {T} (k _ {s - 1} \overline {{A}} _ {s - 1} u (k _ {0})))}{u ^ {T} (k _ {s} \overline {{A}} u (k _ {0}))}. \tag {14} \\ \end{array}
$$

This implies that we can perform computations and processing more efficiently. The transition matrix $A _ { t }$ plays a crucial role in the discrete diffusion model and needs to be carefully designed to ensure that the reverse process network can more easily extract useful signals from the noise. 

Obscured-and-Replace State Transition Matrix. When addressing human pose estimation in complex occlusion scenarios, we encounter challenges such as self-occlusion, object occlusion, and interpersonal occlusion, which often render key substructures of the pose invisible. To address this issue, we devised an intricate occlusion simulation mechanism that enhances the model’s learning for pose reconstruction under visibility obstruction by integrating occlusion effects within the model. Prior to this, the pose ?? is transformed into ?? tokens $K =$ $( k _ { 0 } , k _ { 1 } , \ldots , | , k _ { N } ) )$ , drawing on masking language modeling approaches similar to those in [79]. We introduce a special Obs token to represent the occlusion of pose substructures, thereby giving each token $( V + 1 )$ states. Furthermore, recognizing that a single occluded area may correspond to various poses, we propose a token replacement strategy to increase the model’s diversity and adaptability in handling occlusions. We define the diffusion of the Obs token as follows: each token has a probability of $\psi _ { s }$ being replaced by the Obs token and a probability of $M \omega _ { s }$ being uniformly diffused, leaving a probability $\mu _ { s } = 1 - M \omega _ { s } - \psi _ { s }$ to remain unchanged, whereas the Obs token always maintains its own state. Therefore, we can construct the transition matrix $A _ { s } \in \mathbb { R } ^ { ( V + 1 ) \times ( V + 1 ) }$ as follows: 

$$
A _ {s} = \left[ \begin{array}{c c c c c} \mu_ {s} + \omega_ {s} & \omega_ {s} & \omega_ {s} & \dots & 0 \\ \omega_ {s} & \mu_ {s} + \omega_ {s} & \omega_ {s} & \dots & 0 \\ \omega_ {s} & \omega_ {s} & \mu_ {s} + \omega_ {s} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ \psi_ {s} & \psi_ {s} & \psi_ {s} & \dots & 1 \end{array} \right].
$$

The introduction of special Obs tokens and random replacement tokens offers significant benefits for model learning: (1) It enhances the model’s ability to remember noisy tokens, thereby simplifying the reverse inference process. (2) As research in [39,78] has shown, the introduction of a small amount of uniform noise is equally important alongside occlusion tokens. (3) The use of random replacement tokens encourages the model to focus not only on occlusion tokens but also on understanding contextual information. (4) The cumulative transition matrix $\overline { { A } } _ { s }$ and probability $q ( k _ { s } | k _ { 0 } )$ in Eq. (13) can be calculated with 

$$
\overline {{{A}}} _ {s} u (k _ {0}) = \overline {{{\mu}}} _ {s} u (k _ {0}) + (\overline {{{\psi}}} _ {s} - \overline {{{\omega}}} _ {s}) u (V + 1) + \overline {{{\omega}}} _ {s}. \tag {15}
$$

where $\begin{array} { r } { \overline { { \mu } } _ { s } = \prod _ { i = 1 } ^ { s } \mu _ { i } , \overline { { \psi } } _ { s } = 1 - \prod _ { i = 1 } ^ { s } ( 1 - \psi _ { i } ) _ { : } } \end{array}$ , and $\overline { { \omega } } _ { s } = ( 1 - \overline { { \mu } } _ { s } - \overline { { \psi } } _ { s } ) / V$ . Research [39] shows that $\mu _ { s } , \omega _ { s }$ and $\psi _ { s }$ can be precalculated and saved, which reduces the computational cost of $q ( k _ { s } | k _ { 0 } )$ from $O ( t V ^ { 2 } )$ to $O ( V )$ . 

Learning the reverse process. Based on the properties of the Markov chain, we can bypass the intermediate steps and directly calculate the probability of reaching any time step $k _ { s }$ from $k _ { 0 } .$ . During the forward process, we randomly sample a time step ?? within the range of 0 to $T ,$ and using Eq. (12), we can obtain the state $k _ { s } .$ . In the reverse process, we trained a network $g _ { \theta } ( k _ { s - 1 } | k _ { s } , y )$ to predict the posterior probability distribution $q ( k _ { s - 1 } | k _ { s } , k _ { 0 } )$ , with the network being trained to the variational lower bound (VLB) [76]: 

$$
\mathcal {L} _ {v l b} = - \log g _ {\theta} (k _ {0} | k _ {1}, y) + \sum_ {s = 1} ^ {T} D _ {K L} [ q (k _ {s - 1} | k _ {s}, k _ {0}) | |
$$

$$
g _ {\theta} (k _ {s - 1} | k _ {s}, y) ] + D _ {K L} (q (k _ {T} | k _ {0}) | | p (k _ {T})). \tag {16}
$$

where $q ( k _ { T } )$ is the prior distribution of the timestep ?? . For the proposed Obscured-and-Replace state transition, the prior is 

$$
q (k _ {T}) = [ \overline {{{\omega}}} _ {T}, \overline {{{\omega}}} _ {T}, \dots , \overline {{{\psi}}} _ {T} ] ^ {T}. \tag {17}
$$

In addition, at each reverse step, we adopt the reparameterization method proposed in [39] to predict the noiseless token distribution $g _ { \theta } ( \overline { { k } } _ { 0 } | k _ { s } , y )$ and then calculate $g _ { \theta } ( k _ { s - 1 } | k _ { s } , y ) \mathrm { ~ a s ~ }$ 

$$
g _ {\theta} (k _ {s - 1} | k _ {s}, y) = \sum_ {\overline {{{{k}}}} _ {0} = 1} ^ {V} q (k _ {s - 1} | k _ {s}, \overline {{{{k}}}} _ {0}) g _ {\theta} (\overline {{{{k}}}} _ {0} | k _ {s}, y). \tag {18}
$$

Based on Eq. (16), an auxiliary loss is introduced to encourage the network to predict $g _ { \theta } ( \overline { { k } } _ { 0 } ^ { \ L } , y ) . \longleftarrow $ 

$$
\underline {{{\mathcal {L}}}} _ {k 0} = - \log g _ {\theta} (k _ {0} | k _ {s}, y). \tag {19}
$$

After the denoising phase is complete, we obtain $\overline { { k } } _ { 0 }$ and apply the cross-entropy loss ${ \mathcal L } _ { t k n } ~ = ~ C E ( \overline { { k } } _ { 0 } , k _ { 0 } )$ to ensure the accuracy of the prediction. Subsequently, we feed ?? into a pretrained pose decoder to generate the estimated pose ${ \overline { { P } } } .$ To increase the precision of the estimated pose, we introduce a pose reconstruction loss $\begin{array} { r l } { \mathcal { L } _ { R e c o n } } & { { } = } \end{array}$ smooth (?? , ?? ), which is designed to minimize the discrepancy between the predicted pose $\overline { P }$ and the actual pose ?? . Ultimately, we integrate all the loss terms and define the total loss as 

$$
\mathcal {L} _ {\text { all }} = \eta \mathcal {L} _ {k 0} + \mathcal {L} _ {v l b} + \mathcal {L} _ {t k n} + \mathcal {L} _ {\text { Recon }}. \tag {20}
$$

The hyperparameter ?? is used to adjust the weight of the auxiliary loss ??0. $\mathcal { L } _ { k 0 } .$ 

Inference process. Unlike the training phase, where the true pose tokens $k _ { 0 }$ are available, during the inference phase, the ground truth values of $k _ { 0 }$ are not present. In this scenario, we initialize $\underbrace { k _ { T } }$ to a pure noise state of all zeros. The denoiser then utilizes the multimodal conditions, the time step ?? , and the previously predicted tokens to predict each token progressively through a reverse pass. This process starts from $k _ { T } ,$ progressively denoising back to $k _ { 0 } ,$ then readding noise to $k _ { T - 1 } ,$ repeating the denoising back to $k _ { 0 } ,$ and so on until the entire sequence from $k _ { 1 }$ to $k _ { 0 }$ is completed. Even in the low-resolution latent space, this step-by-step sampling method is quite time-consuming. To address this issue, we employ a reparameterization trick that allows us to skip some steps in the diffusion model, thereby accelerating the inference speed. Specifically, a time interval ???? is set, and instead of sequentially sampling from $k _ { T }$ to $k _ { 0 } ,$ we opt for a leapfrog sampling approach, such as transitioning directly from $k _ { T }$ to $k _ { T - \varDelta t }$ and then to $k _ { T - 2 , 4 t }$ , reducing the number of necessary sampling points and enhancing inference efficiency. This method takes advantage of the reverse transition distribution, 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/f84afacae7c3f00de6098d5d94d56ac656383cd4d87c141c0885ed3e49216b07.jpg)


$$
g _ {\theta} (k _ {t - \Delta t} \mid k _ {t}, y) = \sum_ {\overline {{k}} _ {0} = 1} ^ {V} q (k _ {t - \Delta t} \mid k _ {t}, \overline {{k}} _ {0}) g _ {\theta} (\overline {{k}} _ {0} \mid k _ {t}, y). \tag {21}
$$

In this way, we not only improved the efficiency of sampling but also made the impact on prediction accuracy virtually negligible. 

Model architecture. We design a conditional discrete diffusion model to estimate the conditional probability distribution accurately $g _ { \theta } ( k _ { 0 } | k _ { s } , y )$ . As shown in Fig. 2, the model framework consists of two key components: one part focuses on generating conditional tokens, and the other is dedicated to pose denoising. The model first constructs multimodal conditional tokens through three parallel feature extr ons: based on the Swin-Base backbone to extract fine-grained ima ocal features, combined with high-level semantic features output by the CLIP image encoder and cross-modal semantic embeddings generated by the CLIP text encoder, aligned by projection and fused into a unified conditional token. The pose denoiser adopts a modular design, composed of multiple basic modules, each carefully integrating AdaLNorm operations [80], self-attention mechanisms, cross-attention mechanisms, and feedforward networks (FNN). By dynamically adjusting the network’s sensitivity to the diffusion time step through AdaL-Norm, its core formula is given by ???????? $N ( f , t ) = a _ { t } { \cdot } L a y e r N o r m ( f ) { + } b _ { t } ;$ , where $f$ represents the intermediate activation values and where $a _ { t }$ and $b _ { t }$ are parameters obtained through a linear projection of the timestep embedding. The denoising module further integrates selfattention layers with cross-attention layers; the former captures the global structural dependencies among pose tokens, with queries (??), keys (??), and values (?? ) all derived from pose tokens. The latter uses multimodal conditional tokens as queries $( Q ) ,$ and noisy pose tokens as keys (??) and values (?? ), achieving semantically guided feature interaction, and ultimately enhances non-linear expressive capabilities through the feedforward network. The model iteratively denoises the noisy tokens $k _ { t }$ to gradually restore them to pure original noise-free tokens $k _ { 0 } ,$ and inputs them into the decoder pretrained by the hierarchical prior model to reconstruct a complete pose that conforms to human prior constraints. The model explicitly models the occlusion noise distribution, enhances semantic consistency through the CLIP multimodal alignment characteristics, and simultaneously endows the network with adaptive discrimination capability to noise intensity through the AdaLNorm mechanism, showing significant advantages in occlusion robustness and cross-modal conditional driving. 

# 4. Experiments

# 4.1. Datasets and metrics

COCO [81] is a commonly used dataset for 2D human pose estimation, encompassing over 200,000 human images, each annotated with 17 keypoints. We use standard splits for training and testing. 

3DMPB [34] contains 2D and 3D pose annotations, which is a challenging benchmark for multiperson pose estimation tasks. It comprises over 10,000 images capturing basketball scenarios, with each scene featuring 1 to 4 individuals, encompassing numerous human-human interactions and occlusions during basketball gameplay. 

OcMotion [47] focuses on occlusion scenarios and contains 300,000 images. The 2D pose annotations of the dataset are generated by advanced detectors and have been manually verified. The dataset simulates real-world situations where human bodies are occluded by objects or other people and supports the study of multi-person interaction scenarios. 

Human3.6M [82] is a large-scale 3D human pose estimation dataset, containing over 3.6 million frames of image data. It features 11 professional actors performing various daily activities such as walking, discussing, and waving in natural environments, with precise 3D joint location annotations provided. 

Evaluation metrics. We adopt average precision (AP) based on object key similarity (OKS) [81] as the main metric and specifically analyze the performance of the AP at IoU thresholds of 50% and 75% (i.e., AP50 and AP75). 

# 4.2. Implementation details

All the experiments are conducted on an NVIDIA A40 PCIe GPU. The model is fully implemented with PyTorch. 

Hierarchical Discrete Prior Model. The pose encoder uniformly adopts an architecture that integrates two linear projection layers and four MLP-mixer blocks [77] to deeply fuse key body part features. The decoder is composed of two linear projection layers and one MLP-mixer block. We designed four codebooks with dimensions of $2 0 4 8 \times 5 1 2$ to encode the features of the full body, head, arms, and legs into 34, 10, 12, and 12 tokens, respectively. During training, the model used a learning rate of 0.0008, a weight decay of 0.05, and a batch size of 32, iterating for 300 epochs on the COCO dataset. The optimization strategy employed the Adam optimizer. 

Conditional Discrete Diffusion Model. The parameters of the Swin-Base and CLIP models are fixed. The pose denoiser is composed of 19 transformer blocks, each equipped with self-attention, crossattention, and a feed-forward network (FFN). The channel count of each block is 1024. The FFN contains two linear layers, which expand the dimension to 4096 in the middle layer. The length of the conditional tokens is set to 34, with an embedding dimension of 512. In the training setup, the number of time steps (?? ) is set to 100, the weight of the auxiliary loss $\eta$ is 0.0005, and the learning rate is 0.0008. The strategy for the transition matrix starts with $\omega _ { s }$ and $\psi _ { s }$ linearly increasing from 0.9 and 0.1, respectively. The entire network optimization uses AdamW with parameters $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 6$ . Our strategy for generating pose text descriptions uses a format like ‘‘the {body part} of the human is occluded’’ to describe the pose in the input image. Specifically, we analyze keypoint information from the datasets, including their coordinates, visibility, and connection descriptions, to produce the text prompts. If some keypoints are not visible, we aggregate these points and their connection information; for instance, if the left shoulder and left elbow are both invisible, we describe it as ‘‘The left hand of the human is occluded’’. Conversely, if all keypoints are visible, the overall description is ‘‘The body of the human is unoccluded’’. For datasets lacking text descriptions, we use a large language model to generate prompt templates such as ‘‘the {body part (e.g., head, arm, torso, and leg)} of the human is occluded(or unoccluded)’’, and then use ChatPose to produce the text prompts [88,89]. 

# 4.3. Results on COCO, 3DMPB, and OcMotion

Comparison with state-of-the-art methods. We compared our method with the current state-of-the-art methods on the COCO dataset, and the experimental results are shown in Table 1. Our method achieves higher prediction accuracy than both deep learning-based methods [17, 25,29,44,60,61,85] and prior-based methods with single deterministic outputs [34,35]. PRTR [85] uses self-attention to directly regress the locations of human keypoints but has limited capability in extracting global features from input images. CPN [60] fully utilizes the network’s feature extraction ability by designing GlobalNet and RefineNet. SimpleBaseline [61] directly regresses keypoint coordinates through a deep convolutional network. HRNet [17] enhances feature representation through multi-scale fusion. TokenPose [25] improves pose estimation accuracy by learning constraints between keypoints and appearance cues in the image. TransPose [29] improves keypoint localization accuracy by capturing long-range dependencies between keypoints. Compared with these methods, our method achieves AP improvements of 7.4%, 6.3%, 4.7%, 4.2%, 3.7%, and 3.7%, respectively. When human poses are obscured and there are limited visual cues available, relying solely on feature enhancement and long-distance dependencies to address this issue proves to be inadequate. 


Table 1 Comparisons with state-of-the-art models on coco val2017. Our method achieves comparable performance with others. † indicates that the Gaussian label smoothing is adopted. - means the numbers are not provided in the original papers.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Params</td><td rowspan="2">GFLOPs</td><td colspan="10">COCO val2017</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AP(M)</td><td>AP(L)</td><td>AR</td><td>AR.5</td><td>AR.75</td><td>AR(M)</td><td>AR(L)</td></tr><tr><td>CPN [60]</td><td>ResNet50</td><td>102M</td><td>6.2G</td><td>72.1</td><td>91.4</td><td>80.0</td><td>68.7</td><td>77.2</td><td>78.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SimpleBaseline [61]</td><td>ResNet152</td><td>68.6M</td><td>35.6G</td><td>73.7</td><td>91.9</td><td>81.1</td><td>70.3</td><td>80.0</td><td>79.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Pose2UV [34]</td><td>ResNet50</td><td>-</td><td>-</td><td>73.9</td><td>91.4</td><td>81.2</td><td>73.0</td><td>76.4</td><td>77.0</td><td>92.6</td><td>83.1</td><td>73.4</td><td>82.5</td></tr><tr><td>TokenPose-B [25]</td><td>HRNet-W32</td><td>13.5M</td><td>5.7G</td><td>74.7</td><td>89.8</td><td>81.4</td><td>71.3</td><td>81.4</td><td>80.0</td><td>92.8</td><td>82.1</td><td>72.7</td><td>81.6</td></tr><tr><td>PCT [35]</td><td>Swin-Base</td><td>-</td><td>15.2G</td><td>76.5</td><td>92.5</td><td>84.7</td><td>77.7</td><td>91.2</td><td>84.7</td><td>94.7</td><td>87.1</td><td>76.9</td><td>85.3</td></tr><tr><td>SimBa (SimCC†) [83]</td><td>ResNet50</td><td>-</td><td>20.2G</td><td>72.7</td><td>91.2</td><td>80.1</td><td>69.2</td><td>79.0</td><td>78.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HRNet (SimCC†) [83]</td><td>HRNet-W48</td><td>-</td><td>14.6G</td><td>75.4</td><td>92.4</td><td>82.7</td><td>71.9</td><td>81.3</td><td>80.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DPIT-B [84]</td><td>HRNet-W32-s</td><td>20.8M</td><td>-</td><td>73.6</td><td>91.4</td><td>81.2</td><td>70.4</td><td>79.5</td><td>78.9</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HRNet [17]</td><td>HRNet-W48</td><td>63.6M</td><td>14.6G</td><td>74.2</td><td>92.4</td><td>82.4</td><td>70.9</td><td>79.7</td><td>79.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PRTR [85]</td><td>ResNet-50</td><td>41.5M</td><td>18.8G</td><td>71.0</td><td>89.3</td><td>78.0</td><td>66.4</td><td>78.8</td><td>78.0</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TransPose [29]</td><td>HRNet-W48</td><td>17.3M</td><td>17.5G</td><td>74.7</td><td>91.9</td><td>82.2</td><td>71.4</td><td>80.7</td><td>79.9</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ViTPose [44]</td><td>ViTPose-Base</td><td>86M</td><td>17.9G</td><td>75.8</td><td>90.7</td><td>83.2</td><td>-</td><td>-</td><td>81.1</td><td>94.6</td><td>87.7</td><td>-</td><td>-</td></tr><tr><td>HRFormer-B [44]</td><td>HRFormer-B</td><td>43.2M</td><td>12.2G</td><td>75.6</td><td>90.8</td><td>82.8</td><td>71.7</td><td>82.6</td><td>80.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>Swin-Base</td><td>38.7M</td><td>14.3G</td><td>78.4</td><td>93.6</td><td>85.9</td><td>77.2</td><td>81.3</td><td>80.8</td><td>94.8</td><td>87.5</td><td>77.5</td><td>85.9</td></tr></table>


Table 2 Comparisons with state-of-the-art models on 3DMPB dataset. Our method achieves comparable performance with others. - means the numbers are not provided in the original papers.


<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Params</td><td rowspan="2">GFLOPs</td><td colspan="8">3DMPB</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AP(L)</td><td>AR</td><td>AR.5</td><td>AR.75</td><td>AR(L)</td></tr><tr><td>Pose2UV [34]</td><td>ResNet50</td><td>-</td><td>-</td><td>57.7</td><td>92.1</td><td>64.7</td><td>59.8</td><td>63.5</td><td>93.7</td><td>72.9</td><td>65.7</td></tr><tr><td>TokenPose [25]</td><td>HRNet-W32</td><td>13.5M</td><td>5.7G</td><td>58.4</td><td>92.6</td><td>66.0</td><td>60.6</td><td>64.0</td><td>93.9</td><td>73.6</td><td>66.3</td></tr><tr><td>PCT [35]</td><td>Swin-Base</td><td>-</td><td>15.2G</td><td>60.3</td><td>93.2</td><td>69.5</td><td>62.4</td><td>66.0</td><td>94.6</td><td>76.1</td><td>68.4</td></tr><tr><td>OpenPose [86]</td><td>VGG-19</td><td>-</td><td>-</td><td>52.9</td><td>91.9</td><td>62.1</td><td>58.5</td><td>60.5</td><td>92.9</td><td>70.4</td><td>65.6</td></tr><tr><td>HRNet [17]</td><td>HRNet-W32</td><td>28.5M</td><td>16.0G</td><td>59.1</td><td>93.4</td><td>66.9</td><td>64.4</td><td>64.3</td><td>93.8</td><td>73.2</td><td>65.6</td></tr><tr><td>AlphaPose [62]</td><td>ResNet50</td><td>99.0M</td><td>5.9G</td><td>58.8</td><td>92.3</td><td>65.9</td><td>61.9</td><td>61.6</td><td>93.7</td><td>70.5</td><td>63.8</td></tr><tr><td>ViTPose [44]</td><td>ViTPose-Base</td><td>86M</td><td>17.9G</td><td>61.4</td><td>93.3</td><td>70.5</td><td>63.7</td><td>67.2</td><td>95.5</td><td>78.4</td><td>69.6</td></tr><tr><td>Ours</td><td>Swin-Base</td><td>38.7M</td><td>14.3G</td><td>62.2</td><td>93.7</td><td>71.2</td><td>64.1</td><td>68.5</td><td>94.6</td><td>79.2</td><td>70.8</td></tr></table>


Table 3 Experiments were conducted on the large-scale OcMotion occlusion dataset, where the proposed model achieved higher prediction accuracy than the current state-of-the-art models.


<table><tr><td rowspan="2">Method</td><td colspan="6">OcMotion</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AR</td><td>AR.5</td><td>AR.75</td></tr><tr><td>Pose2UV [34]</td><td>79.5</td><td>94.1</td><td>85.5</td><td>84.5</td><td>93.7</td><td>78.5</td></tr><tr><td>TokenPose [25]</td><td>76.6</td><td>93.2</td><td>75.1</td><td>82.0</td><td>93.2</td><td>76.8</td></tr><tr><td>PCT [35]</td><td>79.9</td><td>94.5</td><td>82.9</td><td>83.5</td><td>92.6</td><td>78.1</td></tr><tr><td>OpenPose [86]</td><td>73.1</td><td>91.5</td><td>71.5</td><td>79.0</td><td>91.5</td><td>71.0</td></tr><tr><td>HRNet [17]</td><td>77.3</td><td>93.8</td><td>76.3</td><td>81.9</td><td>93.2</td><td>76.6</td></tr><tr><td>TransPose [29]</td><td>76.8</td><td>93.1</td><td>74.2</td><td>78.8</td><td>92.9</td><td>77.7</td></tr><tr><td>ViTPose [44]</td><td>78.3</td><td>93.9</td><td>82.1</td><td>82.6</td><td>93.8</td><td>77.0</td></tr><tr><td>HRFormer-B [87]</td><td>77.9</td><td>94.0</td><td>84.3</td><td>82.6</td><td>94.1</td><td>77.9</td></tr><tr><td>Ours</td><td>83.8</td><td>94.7</td><td>87.9</td><td>85.0</td><td>94.8</td><td>80.8</td></tr></table>

Pose2UV [34] directly recovers human meshes from RGB images by introducing UV priors. PCT [35] transforms pose estimation into a classification task based on VQ-VAE. Both of them tend to output a single deterministic result and lack the capability to handle the diversity of occluded poses. Compared to them, our method improves by 4.5 and 1.7 percentage points respectively. 

Specifically, compared with CPN [60], our method reduces the number of parameters by 62%, thanks to our streamlined encoder, decoder, and denoiser network architecture. Compared with Simple-Baseline [61], our computational cost (GFLOPs) is reduced by 59.8%, mainly due to the reparameterization technique used in our conditional diffusion model. 

To verify the capability of our method for pose estimation in multiperson interaction scenarios, we conducted experiments on the 3DMPB dataset, and the results are shown in Table 2. Our method outperforms Pose2UV [34], TokenPose [25], PCT [35], HRNet [17], and 


Table 4 The speed of all models is recorded on a single A40 GPU with a batch size of 32.


<table><tr><td>Model</td><td>Backbone</td><td>Params (M)</td><td>Speed (fps)</td><td>Image Size</td></tr><tr><td>ViTPose [44]</td><td>ViT-B</td><td>86</td><td>181.6</td><td>256 * 192</td></tr><tr><td>HRNet [17]</td><td>HRNet-W32</td><td>29</td><td>120.8</td><td>384 * 288</td></tr><tr><td>TransPose [29]</td><td>HRNet-W48</td><td>17.3</td><td>90.7</td><td>256 * 192</td></tr><tr><td>TokenPose [25]</td><td>HRNet-W48</td><td>28</td><td>84.6</td><td>256 * 192</td></tr><tr><td>HRFormer-B [87]</td><td>HRFormer-B</td><td>-</td><td>40.3</td><td>384 * 288</td></tr><tr><td>Ours</td><td>Swin-Base</td><td>38.7</td><td>85.7</td><td>256 * 256</td></tr></table>

ViTPose [44]. Additionally, we added comparative experiments with OpenPose [63] and AlphaPose [62]. OpenPose achieves high accuracy by utilizing global contextual information. AlphaPose demonstrates excellent performance on the COCO-wholebody dataset through techniques such as symmetric keypoint regression. Compared to them, our method improves the AP by 9.3 and 3.4 percentage points, respectively, which verifies the robustness of our method in complex multi-person interaction scenarios. 

Furthermore, we conducted experiments on the large-scale occlusion dataset OcMotion, and the results are shown in Table 3. Our method still significantly outperforms deep learning-based methods such as OpenPose [63], HRNet [17], TokenPose [25], AlphaPose [62], and ViTPose [44]. Compared to OpenPose, our method achieved an improvement of 10.7% in AP, further extending our advantage. The data indicates that methods based on VQ-VAE, such as Pose2UV [34] and PCT [35], generally perform better than deep learning-based approaches, further demonstrating that human prior knowledge can significantly enhance the robustness of the model. 

Figs. 3 and 5 intuitively present a comparison of our method with TokenPose, Pose2UV, HRNet, and ViTPose on the COCO val2017 and 3DMPB datasets. Even in severely occluded scenarios, our method is capable of predicting a reasonable configuration of occluded joints that are coordinated with visible joints, validating the strong occlusion modeling capability of our method. Additionally, our method has a greater ability to resolve ambiguities caused by other interfering individuals. In particular, our method demonstrates a significant advantage in the estimation of lower body joints, likely because these areas are more susceptible to occlusion, which poses a greater challenge in pose estimation. Our approach effectively manages this complexity, offering more accurate reconstruction of the lower body posture. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/a6540868613660b6d4fac8c91aa1905311299253fa995376347074f7e791fc1f.jpg)



Fig. 3. The results of human pose estimation in complex scenes. From top to bottom, the rows show the original images, TokenPose prediction results, Pose2UV prediction results, HRNet prediction performance, ViTPose prediction results, and prediction outcomes of our method. When dealing with complex situations such as occlusions, other methods tend to produce unrealistic predictions, whereas our method can provide more reasonable pose estimations. These images are all selected from the COCO val2017 dataset.


Inference time. We conducted inference experiments on an A40 GPU with a batch size of 32, comparing TransPose [29], ViTPose [44], HRNet [17], TokenPose [25], HRFormer-B [87] and our method. The results are recorded in Table 4. Our method has an inference speed 2.13 times faster than HRFormer-B and achieves a 5.9% higher AP on the OcMotion dataset. Despite the slower inference speed due to the extensive sampling and evaluation across multiple time steps required by the discrete diffusion model, our method significantly outperforms 

other methods in terms of accuracy and robustness. On the OcMotion dataset, our model achieves 7.0%, 5.5%, 6.5%, 7.2% and 5.9% higher AP than TransPose [29] , ViTPose [44], HRNet [17], TokenPose [25] and HRFormer [87], respectively. Qualitative experiments show that our method can accurately reconstruct human poses that conform to real-world scenarios, even under severe occlusions. In scenarios where high precision is required, performance is crucial, and the trade-off between speed and performance is acceptable. 

Qualitative results. Figs. 6 present the poses estimated by our method, showing its effectiveness in dealing with complex scenarios such as multiperson interactions or severe occlusions. When extensive occlusions make it difficult to discern the true pose, our approach still provides reasonable pose estimates, even though there may be some deviation from the actual pose. These results highlight the significant role of combining human prior knowledge and contextual information from the conditional diffusion model in improving the accuracy of pose estimation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/4ec219fdb6708282cc223259d43429db563747bacccc8b392897fd4122b8a06a.jpg)



Fig. 4. Qualitative comparison results under occluded and non-occluded conditions on the Human3.6M dataset. The first row shows the reconstruction of unoccluded images, and the second to fifth rows show the reconstruction results of occluded images. For other methods, the pose reconstruction often deviates from the ground truth when there is severe occlusion. Our method mostly shows results consistent with those of unoccluded cases. When there are differences, the reconstruction results are still realistic and reliable..


# 4.4. Artificial occlusion experiment

We conducted occlusion experiments on the Human3.6M dataset, randomly overlaying objects (such as sofas, backpacks, containers, and animals) on images to create pairs of occluded and non-occluded images. We compared HRNet, ViTPose, AlphaPose, and our model, with some results shown in Fig. 4. The reconstruction results for nonoccluded images were similar and are only displayed in the first row. Rows 2 to 4 show the pose reconstructions on occluded images by AlphaPose, HRNet, ViTPose, and our method, respectively. Our model performed better under occlusion conditions, such as when the lower body was occluded by a toy bear (7th column), where other methods deviated from the true pose; when the upper body was occluded (4th column), AlphaPose and HRNet had significant pose deviations, and ViTPose was inferior in detail (head) compared to our method. Although there were differences between the reconstructed poses under severe occlusion and non-occlusion conditions (7th and 8th columns), our method still maintained a high level of robustness. 

# 4.5. Ablation study

We conducted a series of ablation studies on the key components of our model to evaluate their specific impact on overall performance. These components include the strategy of hierarchical priors, the conditional discrete diffusion approach, the composition of hierarchical 


Table 5 Ablation studies on COCO and 3DMPB. To comprehensively test the performance of each component, we conducted ablation studies on COCO val2017 and 3DMPB.


<table><tr><td rowspan="2">Method</td><td colspan="3">COCO val2017</td><td colspan="3">3DMPB</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td><td>AP</td><td>AP.5</td><td>AP.75</td></tr><tr><td>regression (RGB to Pose)</td><td>74.7</td><td>89.8</td><td>81.4</td><td>58.4</td><td>92.6</td><td>66.0</td></tr><tr><td>+prior</td><td>76.8</td><td>92.9</td><td>84.8</td><td>59.4</td><td>93.2</td><td>69.5</td></tr><tr><td>+prior+diffusion</td><td>77.8</td><td>92.9</td><td>85.1</td><td>61.7</td><td>93.5</td><td>70.7</td></tr><tr><td>+prior+diffusion +conditions</td><td>78.4</td><td>93.5</td><td>85.9</td><td>62.2</td><td>93.7</td><td>71.2</td></tr></table>

priors, the setting of the time step T, and the makeup of the conditional feature sequence. The following sections detail the design and results of each ablation experiment. 

Part-aware VQ-VAE. We conducted a series of experiments to demonstrate the significance of the hierarchical priors we proposed. The hierarchical human prior is a 2D human prior that provides additional information to recover complete poses from occluded and incomplete human poses, and it constrains the predicted poses within a physically plausible range, thereby increasing the accuracy of human pose estimation. In Table 5, Row 2, it can be observed that, compared with the regression method without hierarchical priors, the method employing hierarchical priors achieved a 2.1% increase in AP on the COCO dataset. Furthermore, we explored the impact of different hierarchical strategies on the experimental results. In the last row of 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/42218d62e4ff25da749db8abdc0ea03c5c9e713893e0253cb93960434419135e.jpg)



Fig. 5. The results of human pose estimation in complex scenes. From top to bottom, the rows show the original images, TokenPose prediction results, Pose2UV prediction results, HRNet prediction performance, ViTPose prediction results, and prediction outcomes of our method. When dealing with complex situations such as occlusions, other methods tend to produce unrealistic predictions, whereas our method can provide more reasonable pose estimations. These images are all selected from the 3DMPB dataset.


Table 6, we found that integrating encoders for the head, arms, and legs, along with global encoders and codebook configurations, could achieve optimal performance. This confirms that by increasing the number of local encoders, we can acquire finer and richer human prior knowledge, leading to more accurate predictive results. In Table $^ { 6 , }$ Rows 4 and $^ { 6 , }$ the inclusion of leg encoders and codebooks significantly improved performance, primarily because the lower body parts are more susceptible to visual occlusion by other objects. The integration of leg encoders and codebooks helps to enhance the model’s ability to recognize joints in the lower body, thereby improving the overall accuracy of pose estimation. 

We have observed that merely increasing the number of local encoders and codebooks does not continuously enhance model performance. As shown in Table 6, assigning different encoders and codebooks to the left and right upper and lower limbs leads to varying 

degrees of decreased prediction accuracy, especially when the limbs are divided into left and right sections, likely due to excessive partitioning that disrupts the symmetry of posture. When the number of local codebooks is increased to 17, matching the number of keypoints, the model’s performance is suboptimal. This is because assigning independent encoders and codebooks to each keypoint overlooks the interdependencies between joints, causing the model to more easily estimate impractical poses. 

Discrete diffusion model. To verify the effectiveness of the discrete diffusion model in addressing pose estimation issues under occlusion scenarios, we conducted a series of experiments. These experiments not only assessed the performance of the diffusion model in multiperson pose estimation tasks but also specifically examined its predictive capabilities when facing severe occlusion situations. The test results on 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/45f4fc540a7948f24377896210573d943b9f2e1ee74508a7a17dc3e8e9f3993a.jpg)



Fig. 6. Qualitative results when multimodal conditional features are used, with images sourced from the COCO val2017 and 3DMPB datasets.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-11/7da8be20-ae1e-46d3-92dd-4b573fdb3210/f4fefae1962ebdc1388c6e1bb095996252e448f797fd894b38889fe0fd5ec539.jpg)



Fig. 7. The qualitative results obtained via prior knowledge versus diffusion models are compared with image data sourced from the COCO val2017 dataset.



Table 6 Ablations study on the hierarchical discrete prior model on COCO val2017 dataset. AP are reported.


<table><tr><td rowspan="2">(Hierarchical Discrete Prior</td><td colspan="3">COCO val2017</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td></tr><tr><td>Global</td><td>76.2</td><td>92.3</td><td>84.6</td></tr><tr><td>Head+Global</td><td>76.8</td><td>93.5</td><td>85.8</td></tr><tr><td>Arm+Global</td><td>76.9</td><td>93.5</td><td>84.9</td></tr><tr><td>Leg+Global</td><td>77.9</td><td>93.4</td><td>85.0</td></tr><tr><td>Head+Arm+Global</td><td>77.5</td><td>93.5</td><td>85.7</td></tr><tr><td>Head+Leg+Global</td><td>78.1</td><td>93.5</td><td>85.9</td></tr><tr><td>Arm+Leg+Global</td><td>78.2</td><td>93.5</td><td>85.9</td></tr><tr><td>Head+Arm+Leg+Global</td><td>78.4</td><td>93.6</td><td>86.1</td></tr><tr><td>LeftArm+RightArm+Leg+Global</td><td>77.6</td><td>92.7</td><td>83.4</td></tr><tr><td>Arm+LeftLeg+RightLeg+Global</td><td>76.2</td><td>91.3</td><td>82.9</td></tr><tr><td>LeftArm+RightArm+LeftLeg+RightLeg+Global</td><td>75.1</td><td>90.5</td><td>83.1</td></tr></table>

the 3DMPB dataset demonstrated significant improvements, as shown in Table 5, Row 3, where the accuracy of the pose estimation noticeably increased after the discrete diffusion model was adopted. Furthermore, by comparing the visualization results in Rows 2 and 3 of Fig. 7, we further confirmed the advantages of our model in terms of diversity and detail handling in pose estimation under occlusion scenarios. The predicted results not only corresponded with the true poses but also presented a higher level of precision in detail than methods that did not utilize a diffusion model. This proves that the discrete diffusion model can effectively capture changes in individual poses within complex environments, particularly under occlusion, providing richer and more refined pose estimations. 

Timestep. We conducted an in-depth study on the significance of the number of timesteps during the model training and inference phases. Through a series of experiments on the COCO val2017 dataset, the experimental results are shown in Table 7. We observed that the predictive accuracy of the model significantly improved as the number of training steps increased from 10 to 100. However, when the number of training steps was further extended to 200, the increase in accuracy began to slow, showing a trend toward stabilization. On the basis of these experimental results, we decided to set the default number of timesteps to 100 for our future experimental research. This decision is aimed at balancing model performance with training efficiency, ensuring that while an ideal accuracy rate is achieved, the investment in training resources is also kept reasonable. 

Vision-language guidance. To delve into the advantages of multimodal combinatorial conditions in enhancing predictive performance and to analyze the specific impact of individual conditions on the results, we conducted a series of experiments. By comparing the data in Rows 1 and 2 of Table 8, we find that while single-feature conditions can achieve relatively good results, they are not sufficient to reach the desired level of prediction. Further experiments indicated that when we integrated image features extracted by Swin-Base, visual features captured by the CLIP image encoder, and linguistic features parsed by the CLIP text encoder, the accuracy of prediction was significantly improved, achieving optimal performance. This highlights the complementarity between image, visual, and linguistic features and reveals the key role of multimodal information in the process of optimizing predictions. By integrating multimodal features, the model can more effectively filter out irrelevant distractions such as background noise and enhance its ability to recognize occluded areas. 

# 5. Conclusion

In this paper, we present a novel local-aware human pose estimation framework designed to address the challenge of occlusion in 2D human pose estimation. We have developed a hierarchical human prior model that learns human pose priors through a hierarchical encoder, codebook, and decoder, enabling the reconstruction of plausible poses from partially visible pose information. For body parts with severe occlusion that may correspond to multiple poses, we introduce a discrete diffusion model, combined with occlusion and replacement strategies, to explicitly incorporate occlusion factors into the learning process. By integrating multimodal conditional features, time step T, and the learned human prior, we trained a denoising network to learn how to remove occlusions and reconstruct clean poses. Our research findings indicate that hierarchical discrete priors are more effective at capturing local pose details than single priors are. We have also demonstrated that the multimodal conditional discrete diffusion model has significant advantages in handling the diversity of occluded poses. Extensive evaluations conducted on the COCO and 3DMPB datasets have shown that our method outperforms the state-of-the-art techniques in terms of performance. 


Table 7 Ablation study on training steps and inference steps on COCO val2017 dataset. AP are reported.


<table><tr><td colspan="2"></td><td colspan="5">Training steps</td></tr><tr><td rowspan="6">Inference steps</td><td></td><td>10</td><td>25</td><td>50</td><td>100</td><td>200</td></tr><tr><td>10</td><td>75.1</td><td>75.5</td><td>75.7</td><td>75.9</td><td>75.6</td></tr><tr><td>25</td><td>-</td><td>76.1</td><td>76.4</td><td>76.8</td><td>76.6</td></tr><tr><td>50</td><td>-</td><td>-</td><td>77.2</td><td>77.9</td><td>77.6</td></tr><tr><td>100</td><td>-</td><td>-</td><td>-</td><td>78.4</td><td>77.9</td></tr><tr><td>200</td><td>-</td><td>-</td><td>-</td><td>-</td><td>78.1</td></tr></table>


Table 8 Ablation study on multimodal combination conditions on COCO val2017. AP are reported.


<table><tr><td rowspan="2">Condition</td><td colspan="3">COCO val2017</td></tr><tr><td>AP</td><td>AP.5</td><td>AP.75</td></tr><tr><td>Swin-Base</td><td>76.5</td><td>93.1</td><td>84.8</td></tr><tr><td>CLIP</td><td>77.1</td><td>93.5</td><td>85.3</td></tr><tr><td>Swin-Base+CLIP</td><td>78.4</td><td>93.6</td><td>86.1</td></tr></table>

# CRediT authorship contribution statement

Hongyu Xiao: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Hui He: Validation, Software, Methodology, Formal analysis. Yifan Xie: Visualization, Validation, Investigation. Yi Zheng: Resources, Investigation, Data curation. 

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

# Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant No. 62277007). 

# Appendix. List of abbreviations

AP: Average Precision 

IoU: Intersection over Union 

AP50: Average Precision at IoU=0.5 

AP75: Average Precision at IoU=0.75 

AR: Average Recall 

AR50: Average Recall at IoU=0.5 



AR75: Average Recall at IoU=0.75 





COCO: Common Objects in Context 





CLIP: Contrastive Language-Image Pre-training 





DDPM: Denoising Diffusion Probabilistic Models 





EMA: Exponential Moving Average 





FFN: Feed-Forward Network 





MLP-Mixer: Multi-Layer Perceptron Mixer 





MPPE: Multi-Person Pose Estimation 





OKS: Object Key Similarity 





PCT: Pose as Compositional Tokens 





RGB: Red, Green, Blue 





Swin-Base: Swin Transformer Base 





VQ-VAE: Vector Quantized Variational AutoEncoder 





VQ-VAE-2: Vector Quantized Variational AutoEncoder 2 





VLB: Variational Lower Bound 



# Data availability

Data will be made available on request. 

# References



[1] A. Latreche, R. Kelaiaia, A. Chemori, A. Kerboua, A new home-based upper-and lower-limb telerehabilitation platform with experimental validation, Arab. J. Sci. Eng. 48 (8) (2023) 10825–10840. 





[2] A. Latreche, R. Kelaiaia, A. Chemori, A. Kerboua, Reliability and validity analysis of MediaPipe-based measurement system for some human rehabilitation motions, Measurement 214 (2023) 112826. 





[3] Y. Tian, H. Fu, H. Wang, Y. Liu, Z. Xu, H. Chen, J. Li, R. Wang, RGB oralscan video-based orthodontic treatment monitoring, Sci. China Inf. Sci. 67 (1) (2024) 112107. 





[4] Y. Tian, G. Jian, J. Wang, H. Chen, L. Pan, Z. Xu, J. Li, R. Wang, A revised approach to orthodontic treatment monitoring from oralscan video, IEEE J. Biomed. Heal. Informatics (2023). 





[5] M. Urgo, F. Berardinucci, P. Zheng, L. Wang, AI-based pose estimation of human operators in manufacturing environments, in: CIRP Novel Topics in Production Engineering: Volume 1, Springer, 2024, pp. 3–38. 





[6] T. Fukushima, P. Blauberger, T. Guedes Russomanno, M. Lames, The potential of human pose estimation for motion capture in sports: a validation study, Sport. Eng. 27 (1) (2024) 19. 





[7] X. Xi, C. Zhang, W. Jia, R. Jiang, Enhancing human pose estimation in sports training: Integrating spatiotemporal transformer for improved accuracy and real-time performance, Alex. Eng. J. 109 (2024) 144–156. 





[8] B. Parsaeifard, S. Saadatnejad, Y. Liu, T. Mordan, A. Alahi, Learning decoupled representations for human pose forecasting, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2021, pp. 2294–2303. 





[9] B. Huang, J. Ju, Z. Li, Y. Wang, Reconstructing groups of people with hypergraph relational reasoning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2023, pp. 14873–14883. 





[10] B. Huang, L. Pan, Y. Yang, J. Ju, Y. Wang, Neural mocon: Neural motion control for physically plausible human motion capture, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 6417–6426. 





[11] B. Huang, T. Zhang, Y. Wang, Object-occluded human shape and pose estimation with probabilistic latent consistency, IEEE Trans. Pattern Anal. Mach. Intell. 45 (4) (2022) 5010–5026. 





[12] U. Iqbal, P. Molchanov, J. Kautz, Weakly-supervised 3d human pose learning via multi-view images in the wild, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2020, pp. 5243–5252. 





[13] J. Li, W. Su, Z. Wang, Simple pose: Rethinking and improving a bottom-up approach for multi-person pose estimation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2019, pp. 9237–9246. 





[14] A. Nadeem, A. Jalal, K. Kim, Automatic human posture estimation for sport activity recognition with robust body parts detection and entropy markov model, Multimedia Tools Appl. 80 (2021) 21465–21498. 





[15] L. Schmidtke, A. Vlontzos, S. Ellershaw, A. Lukens, T. Arichi, B. Kainz, Unsupervised human pose estimation through transforming shape templates, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2021, pp. 2484–2494. 





[16] D. Sun, S. Wang, H. Xia, C. Zhang, J. Gao, M. Mao, Human pose estimation based on cross-view feature fusion, Vis. Comput. 40 (9) (2024) 6581–6597. 





[17] K. Sun, B. Xiao, D. Liu, J. Wang, Deep high-resolution representation learning for human pose estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019, pp. 5693–5703. 





[18] Y. Tian, W. Hu, H. Jiang, J. Wu, Densely connected attentional pyramid residual network for human pose estimation, Neurocomputing 347 (2019) 13–23. 





[19] D. Wang, Stacked Dense-Hourglass Networks for Human Pose Estimation (Ph.D. thesis), University of Illinois at Urbana-Champaign, 2018. 





[20] J. Wang, W. Wang, X. Zhang, KTPose: Keypoint-based tokens in vision transformer for human pose estimation, in: 2023 IEEE International Conference on Systems, Man, and Cybernetics, SMC, 2023, pp. 323–328. 





[21] X. Wang, Y. Tian, X. Zhao, T. Yang, J. Gelernter, J. Wang, G. Cheng, W. Hu, Improving multiperson pose estimation by mask-aware deep reinforcement learning, ACM Trans. Multimed. Comput. Commun. Appl. (TOMM) 16 (3) (2020) 1–18. 





[22] L. Ke, M.C. Chang, H. Qi, S. Lyu, Multi-scale structure-aware network for human pose estimation, in: Proceedings of the European Conference on Computer Vision, ECCV, 2018, pp. 713–728. 





[23] G. Xie, J. Wang, T. Zhang, J. Lai, R. Hong, G.J. Qi, Interleaved structured sparse convolutional neural networks, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2018, pp. 8847–8856. 





[24] T. Zhang, G.J. Qi, B. Xiao, J. Wang, Interleaved group convolutions, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2017, pp. 4373–4382. 





[25] Y. Li, S. Zhang, Z. Wang, S. Yang, W. Yang, S.-T. Xia, E. Zhou, Tokenpose: Learning keypoint tokens for human pose estimation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2021, pp. 11313–11322. 





[26] H. Liu, Q. Chen, Z. Tan, J.J. Liu, J. Wang, X. Su, X. Li, K. Yao, J. Han, E. Ding, et al., Group pose: A simple baseline for end-to-end multi-person pose estimation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2023, pp. 15029–15038. 





[27] P. Sun, K. Gu, Y. Wang, L. Yang, A. Yao, Rethinking visibility in human pose estimation: Occluded pose reasoning via transformers, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 5903–5912. 





[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 5998–6008. 





[29] S. Yang, Z. Quan, M. Nie, W. Yang, Transpose: Keypoint localization via transformer, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2021, pp. 11802–11812. 





[30] C. Zheng, S. Zhu, M. Mendieta, T. Yang, C. Chen, Z. Ding, 3D human pose estimation with spatial and temporal transformers, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2021, pp. 11656–11665. 





[31] A. Van Den Oord, O. Vinyals, et al., Neural discrete representation learning, Adv. Neural Inf. Process. Syst. 30 (2017). 





[32] D. Rempe, T. Birdal, A. Hertzmann, J. Yang, S. Sridhar, L.J. Guibas, Humor: 3d human motion model for robust pose estimation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2021, pp. 11488–11499. 





[33] B. Huang, Y. Shu, T. Zhang, Y. Wang, Dynamic multi-person mesh recovery from uncalibrated multi-view cameras, in: 2021 International Conference on 3D Vision, 3DV, IEEE, 2021, pp. 710–720. 





[34] B. Huang, T. Zhang, Y. Wang, Pose2uv: Single-shot multiperson mesh recovery with deep uv prior, IEEE Trans. Image Process. 31 (2022) 4679–4692. 





[35] Z. Geng, C. Wang, Y. Wei, Z. Liu, H. Li, H. Hu, Human pose as compositional tokens, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2023, pp. 660–671. 





[36] A. Razavi, A. Van den Oord, O. Vinyals, Generating diverse high-fidelity images with vq-vae-2, Adv. Neural Inf. Process. Syst. 32 (2019) 14866–14876. 





[37] B. Huang, C. Li, C. Xu, L. Pan, Y. Wang, G.H. Lee, Closely interactive human reconstruction with proxemics and physics-guided adaption, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2024, pp. 1011–1021. 





[38] B. Huang, J. Ju, Y. Shu, Y. Wang, Simultaneously recovering multi-person meshes and multi-view cameras with human semantics, IEEE Trans. Circuits Syst. Video Technol. (2023). 





[39] S. Gu, D. Chen, J. Bao, F. Wen, B. Zhang, D. Chen, L. Yuan, B. Guo, Vector quantized diffusion model for text-to-image synthesis, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2022, pp. 10696–10706. 





[40] N. Chen, Y. Zhang, H. Zen, R.J. Weiss, M. Norouzi, W. Chan, Wavegrad: Estimating gradients for waveform generation, 2020, arXiv preprint arXiv:2009. 00713. 





[41] H. Kong, K. Gong, D. Lian, M.B. Mi, X. Wang, Priority-centric human motion generation in discrete latent space, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 14806–14816. 





[42] G. Han, C. Song, S. Wang, H. Wang, E. Chen, G. Wang, Occluded human pose estimation based on limb joint augmentation, 2024, arXiv preprint arXiv: 2410.09885. 





[43] T. Zhang, B. Huang, Y. Wang, Object-occluded human shape and pose estimation from a single color image, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7376–7385. 





[44] Y. Xu, J. Zhang, Q. Zhang, D. Tao, Vitpose: Simple vision transformer baselines for human pose estimation, Adv. Neural Inf. Process. Syst. 35 (2022) 38571–38584. 





[45] J. Huang, Z. Zhu, F. Guo, G. Huang, The devil is in the details: Delving into unbiased data processing for human pose estimation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 5700–5709. 





[46] J. Ju, B. Huang, C. Zhu, Z. Li, Y. Wang, Physics-guided human motion capture with pose probability modeling, in: Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 2023, pp. 947–955. 





[47] B. Huang, Y. Shu, J. Ju, Y. Wang, Occluded human body capture with self-supervised spatial-temporal motion prior, 2022, arXiv preprint arXiv:2207. 05375. 





[48] X. Peng, Z. Tang, F. Yang, R. Feris, D. Metaxas, Jointly optimize data augmentation and network training: Adversarial data augmentation in human pose estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2018, pp. 9286–9295. 





[49] Y. Bin, X. Cao, X. Chen, Y. Ge, Y. Tai, C. Wang, J. Li, F. Huang, C. Gao, N. Sang, Adversarial semantic data augmentation for human pose estimation, in: Proceedings of the European Conference on Computer Vision, ECCV, 2020, pp. 606–622. 





[50] X. Gong, W. Chen, Y. Jiang, Y. Yuan, Z. Wang, AutoPose: Searching multi-scale branch aggregation for pose estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2020, pp. 7356–7365. 





[51] A. Newell, K. Yang, J. Deng, Stacked hourglass networks for human pose estimation, Lecture Notes in Comput. Sci. 9912 (2016) 483–499. 





[52] H. Liu, K. Simonyan, Y. Yang, DARTS: Differentiable architecture search, Int. Conf. Learn. Represent. (ICLR) (2019) URL: https://openreview.net/pdf?id= S1eYHoC5FX. (Accessed: 26 September 2024). 





[53] F. Zhang, X. Zhu, H. Dai, M. Ye, C. Zhu, Distribution-aware coordinate representation for human pose estimation, in: Proceedings of TheIEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2020. 





[54] F. Yang, Z. Song, Z. Xiao, Y. Chen, Z. Pan, M. Zhang, M. Xue, Y. Mo, Y. Zhang, G. Guan, Train your data processor: Distribution-aware and error-compensation coordinate decoding for human pose estimation, 2020, ArXiv E-Prints, arXiv: 2007.05887. (Accessed: 26 September 2024). 





[55] M. Fieraru, A. Khoreva, L. Pishchulin, B. Schiele, Learning to refine human pose estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2018, pp. 205–214. 





[56] S.E. Wei, V. Ramakrishna, T. Kanade, Y. Sheikh, Convolutional pose machines, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2016, pp. 4724–4732. 





[57] R. Zhang, Z. Zhu, P. Li, R. Wu, H. Xia, Exploiting offset-guided network for pose estimation and tracking, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019, pp. 1–4. 





[58] X. Sun, B. Xiao, S. Liang, Y. Wei, Integral human pose regression, in: Proceedings of the IEEE International Conference on Computer Vision, ICCV, 2017, pp. 3243–3252. 





[59] H.S. Fang, S. Xie, Y.W. Tai, C. Lu, Rmpe: Regional multi-person pose estimation, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2017, pp. 2334–2343. 





[60] Y. Chen, Z. Wang, Y. Peng, Z. Zhang, G. Yu, J. Sun, Cascaded pyramid network for multi-person pose estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2018, pp. 7103–7112. 





[61] B. Xiao, H. Wu, Y. Wei, Simple baselines for human pose estimation and tracking, in: Proceedings of the European Conference on Computer Vision, ECCV, 2018, pp. 466–481. 





[62] H.S. Fang, J. Li, H. Tang, C. Xu, H. Zhu, Y. Xiu, Y.L. Li, C. Lu, Alphapose: Wholebody regional multi-person pose estimation and tracking in real-time, IEEE Trans. Pattern Anal. Mach. Intell. 45 (6) (2022) 7157–7173. 





[63] Z. Cao, T. Simon, S.E. Wei, Y. Sheikh, Realtime multi-person 2d pose estimation using part affinity fields, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2017, pp. 7291–7299. 





[64] B. Cheng, B. Xiao, J. Wang, H. Shi, T.S. Huang, L. Zhang, Higherhrnet: Scale-aware representation learning for bottom-up human pose estimation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2020, pp. 5386–5395. 





[65] Z. Geng, K. Sun, B. Xiao, Z. Zhang, J. Wang, Bottom-up human pose estimation via disentangled keypoint regression, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2021, pp. 14676–14686. 





[66] X. Nie, J. Feng, J. Zhang, S. Yan, Single-stage multi-person pose machines, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2019, pp. 6951–6960. 





[67] Z. Tian, H. Chen, C. Shen, DirectPose: Direct end-to-end multi-person pose estimation, IEEE Trans. Pattern Anal. Mach. Intell. 43 (1) (2021) 260–273. 





[68] D. Shi, X. Wei, L. Li, Y. Ren, W. Tan, End-to-end multi-person pose estimation with transformers, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2022, pp. 11069–11078. 





[69] F. Bogo, A. Kanazawa, C. Lassner, P. Gehler, J. Romero, M.J. Black, Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image, in: Proceedings of the European Conference on Computer Vision, ECCV, 2016, pp. 561–578. 





[70] M. Petrovich, M.J. Black, G. Varol, Action-conditioned 3d human motion synthesis with transformer vae, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2021, pp. 10985–10995. 





[71] G. Pavlakos, V. Choutas, N. Ghorbani, T. Bolkart, A.A. Osman, D. Tzionas, M.J. Black, Expressive body capture: 3d hands, face, and body from a single image, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019, pp. 10975–10985. 





[72] D.P. Kingma, M. Welling, Auto-encoding variational Bayes, in: 2nd International Conference on Learning Representations, ICLR, 2014. 





[73] H. Ci, M. Wu, W. Zhu, X. Ma, H. Dong, F. Zhong, Y. Wang, Gfpose: Learning 3d human pose prior with gradient fields, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2023, pp. 4800–4810. 





[74] G. Tiwari, D. Antić, J.E. Lenssen, N. Sarafianos, T. Tung, G. Pons-Moll, Posendf: Modeling human pose manifolds with neural distance fields, in: European Conference on Computer Vision, Springer, 2022, pp. 572–589. 





[75] J. Ho, A. Jain, P. Abbeel, Denoising diffusion probabilistic models, Adv. Neural Inf. Process. Syst. 33 (2020) 6840–6851. 





[76] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli, Deep unsupervised learning using nonequilibrium thermodynamics, in: International Conference on Machine Learning, 2015, pp. 2256–2265. 





[77] I.O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, et al., Mlp-mixer: An all-mlp architecture for vision, Adv. Neural Inf. Process. Syst. 34 (2021) 24261–24272. 





[78] J. Austin, D.D. Johnson, J. Ho, D. Tarlow, R. Van Den Berg, Structured denoising diffusion models in discrete state-spaces, Adv. Neural Inf. Process. Syst. 34 (2021) 17981–17993. 





[79] J. Devlin, M.W. Chang, K. Lee, K. Toutanova, BERT: Pre-training of deep bidirectional transformers for language understanding, in: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Vol.1, 2019, pp. 4171–4186. 





[80] J. Xu, X. Sun, Z. Zhang, G. Zhao, J. Lin, Understanding and improving layer normalization, Adv. Neural Inf. Process. Syst. 32 (2019). 





[81] T.Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C.L. Zitnick, Microsoft coco: Common objects in context, in: Proceedings of the European Conference on Computer Vision, ECCV, 2014, pp. 740–755. 





[82] C. Ionescu, D. Papava, V. Olaru, C. Sminchisescu, Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments, IEEE Trans. Pattern Anal. Mach. Intell. 36 (7) (2013) 1325–1339. 





[83] Y. Li, S. Yang, P. Liu, S. Zhang, Y. Wang, Z. Wang, W. Yang, S.T. Xia, SimCC: A Simple Coordinate Classification Perspective for Human Pose Estimation, Springer, Cham, 2022. 





[84] S. Zhao, K. Liu, Y. Huang, Q. Bao, D. Zeng, W. Liu, Dpit: Dual-pipeline integrated transformer for human pose estimation, in: CAAI International Conference on Artificial Intelligence, 2022, pp. 559–576. 





[85] K. Li, S. Wang, X. Zhang, Y. Xu, W. Xu, Z. Tu, Pose recognition with cascade transformers, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2021, pp. 1944–1953. 





[86] Z. Cao, G. Hidalgo, T. Simon, S.E. Wei, Y. Sheikh, OpenPose: Realtime multiperson 2D pose estimation using part affinity fields, IEEE Trans. Pattern Anal. Mach. Intell. (2018). 





[87] Y. Yuan, R. Fu, L. Huang, W. Lin, C. Zhang, X. Chen, J. Wang, Hrformer: High-resolution transformer for dense prediction, 2021, arXiv preprint arXiv: 2110.09408. 





[88] Y. Feng, J. Lin, S.K. Dwivedi, Y. Sun, P. Patel, M.J. Black, Chatpose: Chatting about 3d human pose, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 2093–2103. 





[89] C. Xu, B. Huang, C. Zhang, Z. Feng, Y. Wang, Adapting human mesh recovery with vision-language feedback, 2025, arXiv preprint arXiv:2502.03836. 

