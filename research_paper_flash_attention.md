## FlashAttention: Fast and Memory-Efficient Exact Attention

## with IO-Awareness

### Tri Daoy, Daniel Y. Fuy, Stefano Ermony, Atri Rudraz, and Christopher Réy

### yDepartment of Computer Science, Stanford University

### zDepartment of Computer Science and Engineering, University at Buffalo, SUNY

### {trid,danfu}@cs.stanford.edu,ermon@stanford.edu,atri@buffalo.edu,

### chrismre@cs.stanford.edu

### June 24, 2022

```
Abstract
Transformers are slow and memory-hungry on long sequences, since the time and memory complexity
of self-attention are quadratic in sequence length. Approximate attention methods have attempted
to address this problem by trading off model quality to reduce the compute complexity, but often do
not achieve wall-clock speedup. We argue that a missing principle is making attention algorithmsIO-
aware—accounting for reads and writes between levels of GPU memory. We proposeFlashAttention,
an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes
between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity
ofFlashAttention, showing that it requires fewer HBM accesses than standard attention, and is
optimal for a range of SRAM sizes. We also extendFlashAttentionto block-sparse attention, yielding
an approximate attention algorithm that is faster than any existing approximate attention method.
FlashAttentiontrains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup
on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3speedup on
GPT-2 (seq. length 1K), and 2.4speedup on long-range arena (seq. length 1K-4K).FlashAttention
and block-sparseFlashAttentionenable longer context in Transformers, yielding higher quality models
(0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new
capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge
(seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).
```
## 1 Introduction

```
Transformer models [ 82 ] have emerged as the most widely used architecture in applications such as natural
language processing and image classification. Transformers have grown larger [ 5 ] and deeper [ 83 ], but
equipping them with longer context remains difficult [ 80 ], since the self-attention module at their heart
has time and memory complexity quadratic in sequence length. An important question is whether making
attention faster and more memory-efficient can help Transformer models address their runtime and memory
challenges for long sequences.
Many approximate attention methods have aimed to reduce the compute and memory requirements of
attention. These methods range from sparse-approximation [ 51 , 74 ] to low-rank approximation [ 12 , 50 , 84 ],
and their combinations [ 3 , 9 , 92 ]. Although these methods reduce the compute requirements to linear or
near-linear in sequence length, many of them do not display wall-clock speedup against standard attention
and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not
correlate with wall-clock speed) and tend to ignore overheads from memory access (IO).
In this paper, we argue that a missing principle is making attention algorithmsIO-aware[ 1 ]—that is,
carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU
on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM [ 45 ], Figure 1 left). On modern
```
# arXiv:2205.14135v2 [cs.LG] 23 Jun 2022


```
FlashAttention
```
```
Memory Hierarchy with
Bandwidth & Memory Size
```
```
Attention on GPT-
```
```
PyTorch FlashAttention
```
```
Time (ms)
```
```
Matmul
```
```
Mask
```
```
Softmax
```
```
Dropout
```
```
Matmul
```
```
Fused
Kernel
```
```
Q: N x d V: N X d
```
```
KT: d x N
```
```
QK
```
```
T: N x N
```
```
sm(QKT)V: N x d
```
```
Outer Loop
```
```
Copy Block to SRAM
```
```
Copy
Outer Loop
```
```
Copy
Inner Loop
```
```
Compute Block
on SRAM
```
```
Output to HBM
```
```
Inner Loop
```
```
Inner Loop
```
```
Outer Loop
GPU
SRAM
GPU
HBM
Main Memory
(CPU DRAM)
```
```
SRAM: 19 TB/s (20 MB)
HBM: 1.5 TB/s (40 GB)
```
```
DRAM: 12.8 GB/s
(>1 TB)
```
```
0
```
```
5
```
```
10
```
```
15
```
Figure 1:Left:FlashAttentionuses tiling to prevent materialization of the large𝑁𝑁attention matrix
(dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows),FlashAttentionloops through
blocks of theKandVmatrices and loads them to fast on-chip SRAM. In each block,FlashAttention
loops over blocks ofQmatrix (blue arrows), loading them to SRAM, and writing the output of the attention
computation back to HBM.Right: Speedup over the PyTorch implementation of attention on GPT-2.
FlashAttentiondoes not read and write the large𝑁𝑁attention matrix to HBM, resulting in an 7.6
speedup on the attention computation.

```
GPUs, compute speed has out-paced memory speed [ 61 , 62 , 63 ], and most operations in Transformers are
bottlenecked by memory accesses [ 43 ]. IO-aware algorithms have been critical for similar memory-bound
operations, when reading and writing data can account for a large portion of the runtime—such as database
joins [ 71 ], image processing [ 70 ], numerical linear algebra [ 4 ], and more [ 40 , 85 ]. However, common Python
interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained control of memory
access.
We proposeFlashAttention, a new attention algorithm that computes exact attention with far fewer
memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM.
This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large
intermediate attention matrix for the backward pass. We apply two well-established techniques to address
these challenges. (i) We restructure the attention computation to split the input into blocks and make several
passes over input blocks, thus incrementally performing the softmax reduction (also known astiling). (ii) We
store the softmax normalization factor from the forward pass to quicklyrecomputeattention on-chip in the
backward pass, which is faster than the standard approach of reading the intermediate attention matrix from
HBM. We implementFlashAttentionin CUDA to achieve fine-grained control over memory access and
fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation,
our algorithm bothruns faster(up to 7.6x on GPT-2 [ 67 ], Figure 1 right) anduses less memory—linear
in sequence length—than standard attention, thanks to the massively reduced amount of HBM access.
We analyze the IO complexity [ 1 ] ofFlashAttention, proving that it requires𝑂¹𝑁^2 𝑑^2 𝑀^1 ºHBM
accesses where𝑑is the head dimension and𝑀is the size of SRAM, as compared toΩ¹𝑁𝑑 ̧𝑁^2 ºof standard
attention. For typical values of𝑑and𝑀,FlashAttentionrequires many times fewer HBM accesses
compared to standard attention (up to 9fewer, as shown in Fig. 2). Moreover, we provide a lower bound,
showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over
all SRAM sizes.
We also show thatFlashAttentioncan serve as a useful primitive for realizing the potential of
approximate attention algorithms by overcoming their issues with memory access overhead. As a proof of
concept, we implement block-sparseFlashAttention, a sparse attention algorithm that is 2-4faster than
evenFlashAttention, scaling up to sequence length of 64k. We prove that block-sparseFlashAttention
has better IO complexity thanFlashAttentionby a factor proportional to the sparsity ratio. We discuss
further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix
```

```
multiply) in Section 5. We open-sourceFlashAttentionto make it easier to build on this primitive.^1
We empirically validate thatFlashAttentionspeeds up model training and improves model quality by
modeling longer context. We also benchmark the runtime and memory footprint ofFlashAttentionand
block-sparseFlashAttentioncompared to prior attention implementations.
```
- Faster Model Training.FlashAttentiontrains Transformer models faster in wall-clock time. We
    train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1 [ 58 ], GPT
(seq. length 1K) 3faster than baseline implementations from HuggingFace [ 87 ] and Megatron-LM [ 77 ],
and long-range arena (seq. length 1K-4K) 2.4faster than baselines.
- Higher Quality Models.FlashAttentionscales Transformers to longer sequences, which improves
    their quality and enables new capabilities. We observe a 0.7 improvement in perplexity on GPT-2 and
    6.4 points of lift from modeling longer sequences on long-document classification [13].FlashAttention
    enables the first Transformer that can achieve better-than-chance performance on the Path-X [ 80 ] challenge,
    solely from using a longer sequence length (16K). Block-sparseFlashAttentionenables a Transformer
    to scale to even longer sequences (64K), resulting in the first model that can achieve better-than-chance
    performance on Path-256.
- Benchmarking Attention.FlashAttentionis up to 3faster than the standard attention implemen-
    tation across common sequence lengths from 128 to 2K and scales up to 64K. Up to sequence length of 512,
    FlashAttentionis both faster and more memory-efficient than any existing attention method, whereas
    for sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become
    faster. On the other hand, block-sparseFlashAttentionis faster than all existing approximate attention
    methods that we know of.

## 2 Background

We provide some background on the performance characteristics of common deep learning operations on
modern hardware (GPUs). We also describe the standard implementation of attention.

### 2.1 Hardware Performance

We focus here on GPUs. Performance on other hardware accelerators are similar [46, 48].
GPU Memory Hierarchy. The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of
memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU
has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM
per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [ 44 , 45 ]. The on-chip
SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute
has gotten faster relative to memory speed [ 61 , 62 , 63 ], operations are increasingly bottlenecked by memory
(HBM) accesses. Thus exploiting fast SRAM becomes more important.
Execution Model.GPUs have a massive number of threads to execute an operation (called a kernel).
Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.
Performance characteristics.Depending on the balance of computation and memory accesses, op-
erations can be classified as either compute-bound or memory-bound. This is commonly measured by the
arithmetic intensity[85], which is the number of arithmetic operations per byte of memory access.
1.Compute-bound: the time taken by the operation is determined by how many arithmetic operations there
are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner
dimension, and convolution with large number of channels.
2.Memory-bound: the time taken by the operation is determined by the number of memory accesses, while
time spent in computation is much smaller. Examples include most other operations: elementwise (e.g.,
activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).
Kernel fusion.The most common approach to accelerate memory-bound operations is kernel fusion: if
there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of
multiple times for each operation. Compilers can automatically fuse many elementwise operations [ 53 , 65 , 75 ].

(^1) FlashAttentioncode is available athttps://github.com/HazyResearch/flash-attention


```
However, in the context of model training, the intermediate values still need to be written to HBM to save
for the backward pass, reducing the effectiveness of naive kernel fusion.
```
### 2.2 Standard Attention Implementation

```
Given input sequencesQKV 2 R𝑁𝑑where𝑁is the sequence length and𝑑is the head dimension, we want
to compute the attention outputO 2 R𝑁𝑑:
```
```
S=QK> 2 R𝑁𝑁 P=softmax¹Sº 2R𝑁𝑁 O=PV 2 R𝑁𝑑
```
```
wheresoftmaxis applied row-wise.
Standard attention implementations materialize the matricesSandPto HBM, which takes𝑂¹𝑁^2 ºmemory.
Often𝑁𝑑(e.g., for GPT2,𝑁= 1024 and𝑑= 64 ). We describe the standard attention implementation
in Algorithm 0. As some or most of the operations are memory-bound (e.g., softmax), the large number of
memory accesses translates to slow wall-clock time.
This problem is exacerbated by other elementwise operations applied to the attention matrix, such as
masking applied toSor dropout applied toP. As a result, there have been many attempts to fuse several
elementwise operations, such as fusing masking with softmax [77].
In Section 3.2, we will show that the standard attention implementation performs HBM accesses quadratic
in the sequence length𝑁. We also compare the number of FLOPs and number of HBM accesses of standard
attention and of our method (FlashAttention).
```
```
Algorithm 0Standard Attention Implementation
Require:MatricesQKV 2 R𝑁𝑑in HBM.
1:LoadQKby blocks from HBM, computeS=QK>, writeSto HBM.
2:ReadSfrom HBM, computeP=softmax¹Sº, writePto HBM.
3:LoadPandVby blocks from HBM, computeO=PV, writeOto HBM.
4:ReturnO.
```
## 3 FlashAttention: Algorithm, Analysis, and Extensions

We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate
matrices for the backward pass. This yields an attention algorithm that is both memory efficient and faster in
wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses
compared to standard attention. We further show thatFlashAttentioncan serve as a useful primitive by
extending it to handle block-sparse attention.
We focus here on the forward pass for ease of exposition; Appendix B contains details for the backward.

### 3.1 An Efficient Attention Algorithm With Tiling and Recomputation

```
Given the inputsQKV 2 R𝑁𝑑in HBM, we aim to compute the attention outputO 2 R𝑁𝑑and write it to
HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in𝑁).
We apply two established techniques (tiling, recomputation) to overcome the technical challenge of
computing exact attention in sub-quadratic HBM accesses. We describe this in Algorithm 1. The main idea
is that we split the inputsQKVinto blocks, load them from slow HBM to fast SRAM, then compute the
attention output with respect to those blocks. By scaling the output of each block by the right normalization
factor before adding them up, we get the correct result at the end.
Tiling.We compute attention by blocks. Softmax couples columns ofK, so we decompose the large
softmax with scaling [51, 60, 66]. For numerical stability, the softmax of vector𝑥 2 R𝐵is computed as:
```
```
𝑚¹𝑥º:=max
𝑖
```
##### 𝑥𝑖 𝑓¹𝑥º:=

##### 

##### 𝑒𝑥^1 𝑚¹𝑥º  𝑒𝑥𝐵𝑚¹𝑥º

##### 

#####  ℓ¹𝑥º:=

##### ∑︁

```
𝑖
```
```
𝑓¹𝑥º𝑖 softmax¹𝑥º:=
```
##### 𝑓¹𝑥º

##### ℓ¹𝑥º

##### 


```
For vectors𝑥¹^1 º𝑥¹^2 º 2 R𝐵, we can decompose the softmax of the concatenated𝑥=
```
##### 

##### 𝑥¹^1 º𝑥¹^2 º

##### 

```
2 R^2 𝐵as:
```
```
𝑚¹𝑥º=𝑚¹
```
##### 

##### 𝑥¹^1 º𝑥¹^2 º

##### 

```
º=max¹𝑚¹𝑥¹^1 ºº𝑚¹𝑥¹^2 ººº 𝑓¹𝑥º=
```
```
h
𝑒𝑚¹𝑥
¹ 1 ºº𝑚¹𝑥º
𝑓¹𝑥¹^1 ºº 𝑒𝑚¹𝑥
¹ 2 ºº𝑚¹𝑥º
𝑓¹𝑥¹^2 ºº
```
```
i

```
```
ℓ¹𝑥º=ℓ¹
```
##### 

##### 𝑥¹^1 º𝑥¹^2 º

##### 

##### º=𝑒𝑚¹𝑥

```
¹ 1 ºº𝑚¹𝑥º
ℓ¹𝑥¹^1 ºº ̧𝑒𝑚¹𝑥
¹ 2 ºº𝑚¹𝑥º
ℓ¹𝑥¹^2 ºº softmax¹𝑥º=
```
##### 𝑓¹𝑥º

##### ℓ¹𝑥º

##### 

Therefore if we keep track of some extra statistics (𝑚¹𝑥ºℓ¹𝑥º), we can compute softmax one block at a time.^2
We thus split the inputsQKVinto blocks (Algorithm 1 line 3), compute the softmax values along with
extra statistics (Algorithm 1 line 10), and combine the results (Algorithm 1 line 12).
Recomputation.One of our goals is to not store𝑂¹𝑁^2 ºintermediate values for the backward pass. The
backward pass typically requires the matricesSP 2 R𝑁𝑁to compute the gradients with respect toQKV.
However, by storing the outputOand the softmax normalization statistics¹𝑚ℓº, we can recompute the
attention matrixSandPeasily in the backward pass from blocks ofQKVin SRAM. This can be seen as a
form of selective gradient checkpointing [ 10 , 34 ]. While gradient checkpointing has been suggested to reduce
the maximum amount of memory required [ 66 ], all implementations (that we know off) have to trade speed
for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to
reduced HBM accesses (Fig. 2). The full backward pass description is in Appendix B.
Implementation details: Kernel fusion. Tiling enables us to implement our algorithm in one
CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax,
optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout
in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM.

```
Algorithm 1FlashAttention
Require:MatricesQKV 2 R𝑁𝑑in HBM, on-chip SRAM of size𝑀.
1:Set block sizes𝐵𝑐=
```
##### 𝑀

```
4 𝑑
```
##### 

```
 𝐵𝑟=min
```
##### 𝑀

```
4 𝑑
```
##### 

#####  𝑑

##### 

##### .

```
2:InitializeO=¹ 0 º𝑁𝑑 2 R𝑁𝑑ℓ=¹ 0 º𝑁 2 R𝑁𝑚=¹1º𝑁 2 R𝑁in HBM.
3:DivideQinto𝑇𝑟=
```
```
l
𝑁
𝐵𝑟
```
```
m
blocksQ 1 Q𝑇𝑟of size𝐵𝑟𝑑each, and divideKVin to𝑇𝑐=
```
```
l
𝑁
𝐵𝑐
```
```
m
blocks
K 1 K𝑇𝑐andV 1 V𝑇𝑐, of size𝐵𝑐𝑑each.
4:DivideOinto𝑇𝑟blocksO𝑖O𝑇𝑟of size𝐵𝑟𝑑each, divideℓinto𝑇𝑟blocksℓ𝑖ℓ𝑇𝑟of size𝐵𝑟each,
divide𝑚into𝑇𝑟blocks𝑚 1 𝑚𝑇𝑟of size𝐵𝑟each.
5:for 1 𝑗𝑇𝑐do
6: LoadK𝑗V𝑗from HBM to on-chip SRAM.
7: for 1 𝑖𝑇𝑟do
8: LoadQ𝑖O𝑖ℓ𝑖𝑚𝑖from HBM to on-chip SRAM.
9: On chip, computeS𝑖𝑗=Q𝑖K𝑇𝑗 2 R𝐵𝑟𝐵𝑐.
10: On chip, compute𝑚 ̃𝑖𝑗 =rowmax¹S𝑖𝑗º 2R𝐵𝑟,P ̃𝑖𝑗 =exp¹S𝑖𝑗𝑚 ̃𝑖𝑗º 2R𝐵𝑟𝐵𝑐 (pointwise), ℓ ̃𝑖𝑗 =
rowsum¹P ̃𝑖𝑗º 2R𝐵𝑟.
11: On chip, compute𝑚new𝑖 =max¹𝑚𝑖𝑚 ̃𝑖𝑗º 2R𝐵𝑟,ℓnew𝑖 =𝑒𝑚𝑖𝑚
new𝑖
ℓ𝑖 ̧𝑒𝑚 ̃𝑖𝑗𝑚
new𝑖 ̃
ℓ𝑖𝑗 2 R𝐵𝑟.
12: WriteO𝑖 diag¹ℓ𝑖newº^1 ¹diag¹ℓ𝑖º𝑒𝑚𝑖𝑚
```
```
new
𝑖 O𝑖 ̧𝑒𝑚 ̃𝑖𝑗𝑚
new
𝑖 P ̃𝑖𝑗V𝑗ºto HBM.
13: Writeℓ𝑖 ℓ𝑖new,𝑚𝑖 𝑚new𝑖 to HBM.
14: end for
15:end for
16:ReturnO.
```
We showFlashAttention’s correctness, runtime, and memory requirement (proof in Appendix C).
Theorem 1.Algorithm 1 returnsO=softmax¹QK>ºVwith𝑂¹𝑁^2 𝑑ºFLOPs and requires𝑂¹𝑁ºadditional
memory beyond inputs and output.

### 3.2 Analysis: IO Complexity of FlashAttention

We analyze the IO complexity ofFlashAttention, showing significant reduction in HBM accesses compared
to standard attention. We also provide a lower bound, proving that no exact attention algorithm can

(^2) This style of aggregation is calledalgebraic aggregation[33].


```
Attention Standard FlashAttention
GFLOPs 66.6 75.
HBM R/W (GB) 40.3 4.
Runtime (ms) 41.7 7.
```
```
Sparsity Speedup
```
```
% Non-Zero Blocks
```
```
20 60
```
```
50
```
```
100
```
```
150
```
```
Fwd + Bwd (ms)
```
```
Effect of Block Size
```
```
Block Size
```
```
64128 256 512
```
```
Fwd Runtime (ms)
6
```
```
2
HBM Accesses (GB)
```
```
Dense
FlashAttention
```
```
Block-Sparse
FlashAttention
```
```
2
```
```
4
```
```
6
```
```
HBM Runtime
Accesses
```
Figure 2: Left: Forward + backward runtime of standard attention andFlashAttentionfor GPT-2 medium
(seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor affecting
runtime.Middle: Forward runtime ofFlashAttention(seq. length 1024, head dim. 64, 16 heads, batch size 64) on
A100 GPU. Fewer HBM accesses result in faster runtime, up to a point.Right: The runtime (for seq. length 4K) of
block-sparseFlashAttentionis faster thanFlashAttentionby a factor proportional to the sparsity.

asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C.
Theorem 2.Let𝑁be the sequence length,𝑑be the head dimension, and𝑀be size of SRAM with𝑑𝑀𝑁𝑑.
Standard attention (Algorithm 0) requiresΘ¹𝑁𝑑 ̧𝑁^2 ºHBM accesses, whileFlashAttention(Algorithm 1)
requiresΘ¹𝑁^2 𝑑^2 𝑀^1 ºHBM accesses.

For typical values of𝑑(64-128) and𝑀(around 100KB),𝑑^2 is many times smaller than𝑀, and thus
FlashAttentionrequires many times fewer HBM accesses than standard implementation. This leads to
both faster execution and lower memory footprint, which we validate in Section 4.3.
The main idea of the proof is that given the SRAM size of𝑀, we can load blocks ofKVof sizeΘ¹𝑀ºeach
(Algorithm 1 line 6). For each block ofKandV, we iterate over all blocks ofQ(Algorithm 1 line 8) to compute
the intermediate values, resulting inΘ¹𝑁𝑑𝑀^1 ºpasses overQ. Each pass loadsΘ¹𝑁𝑑ºelements, which
amounts toΘ¹𝑁^2 𝑑^2 𝑀^1 ºHBM accesses. We similarly prove that the backward pass of standard attention
requiresΘ¹𝑁𝑑 ̧𝑁^2 ºHBM accesses while the backward pass ofFlashAttentionrequiresΘ¹𝑁^2 𝑑^2 𝑀^1 º
HBM accesses (Appendix B).
We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all
values of𝑀(the SRAM size) when computing exact attention.
Proposition 3. Let𝑁be the sequence length,𝑑be the head dimension, and𝑀 be size of SRAM with
𝑑𝑀𝑁𝑑. There does not exist an algorithm to compute exact attention with𝑜¹𝑁^2 𝑑^2 𝑀^1 ºHBM accesses
for all𝑀in the range»𝑑 𝑁𝑑¼.
The proof relies on the fact that for𝑀= Θ¹𝑁𝑑ºany algorithm must performΩ¹𝑁^2 𝑑^2 𝑀^1 º= Ω¹𝑁𝑑º
HBM accesses. This type of lower bound over a subrange of𝑀is common in the streaming algorithms
literature [ 88 ]. We leave proving parameterized complexity [ 27 ] lower bounds in terms of𝑀as exciting future
work.
We validate that the number of HBM accesses is the main determining factor of attention run-time.
In Fig. 2 (left), we see that even thoughFlashAttentionhas higher FLOP count compared to standard
attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much
faster runtime. In Fig. 2 (middle), we vary the block size𝐵𝑐ofFlashAttention, which results in different
amounts of HBM accesses, and measure the runtime of the forward pass. As block size increases, the number
of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough
block size (beyond 256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations).
Moreover, larger block size will not fit into the small SRAM size.

### 3.3 Extension: Block-SparseFlashAttention

We extendFlashAttentionto approximate attention: we propose block-sparseFlashAttention, whose
IO complexity is smaller thanFlashAttentionby a factor proportional to the sparsity.
Given inputsQKV 2 R𝑁𝑑and a mask matrixM ̃2 f 0  1 g𝑁𝑁, we want to compute:

```
S=QK> 2 R𝑁𝑁 P=softmax¹S 𝟙M ̃º 2R𝑁𝑁 O=PV 2 R𝑁𝑑
```
```
where¹S 𝟙M ̃º𝑘𝑙=S𝑘𝑙ifM ̃𝑘𝑙= 1 and1ifM𝑘𝑙= 0. We requireM ̃ to have block form: for some block sizes
𝐵𝑟 𝐵𝑐, for all𝑘𝑙,M ̃𝑘𝑙=M𝑖𝑗with𝑖=b𝑘𝐵𝑟c 𝑗=b𝑙𝐵𝑐cfor someM2 f 0  1 g𝑁𝐵𝑟𝑁𝐵𝑐.
```

```
Given a predefined block sparsity maskM2 f 0  1 g𝑁𝐵𝑟𝑁𝐵𝑐we can easily adapt Algorithm 1 to only
compute the nonzero blocks of the attention matrix. The algorithm is identical to Algorithm 1, except we
skip zero blocks. We reproduce the algorithm description in Algorithm 5 in Appendix B.
We also analyze the IO complexity of block-sparseFlashAttention.
```
Proposition 4. Let𝑁be the sequence length,𝑑be the head dimension, and𝑀 be size of SRAM with
𝑑𝑀𝑁𝑑. Block-sparseFlashAttention(Algorithm 5) requiresΘ¹𝑁𝑑 ̧𝑁^2 𝑑^2 𝑀^1 𝑠ºHBM accesses
where𝑠is the fraction of nonzero blocks in the block-sparsity mask.

```
We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the
IO complexity. For large sequence lengths𝑁,𝑠is often set to𝑁^1 ^2 [ 11 ] or𝑁^1 log𝑁[ 3 , 17 , 92 ], resulting
inΘ¹𝑁
```
```
p
𝑁ºorΘ¹𝑁log𝑁ºIO complexity. For downstream experiments, we use the fixed butterfly sparsity
pattern [17], which has been shown to be able to approximate arbitrary sparsity [16].
In Fig. 2 (right), we validate that as the sparsity increases, the runtime of block-sparseFlashAttention
improves proportionally. On the LRA benchmark, block-sparseFlashAttentionachieves 2.8speedup,
while performing on par with standard attention (Section 4).
```
## 4 Experiments

We evaluate the impact of usingFlashAttentionto train Transformer models. We validate two claims
about training time and model accuracy, and report attention runtime and memory benchmarks.

- Training Speed.FlashAttentionoutperforms the MLPerf 1.1 [ 58 ] speed record for BERT by 15%, and
    speeds up GPT-2 up to 3over HuggingFace [ 87 ] and 1  8 over Megatron [ 77 ] over standard Transformers.
    FlashAttentionspeeds up the long-range arena (LRA) benchmark 2.4.
- Quality.FlashAttentionscales Transformers to longer sequences, yielding higher quality.FlashAt-
    tentiontrains GPT-2 with context length 4K faster than Megatron trains GPT-2 with context length
1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two long-
document classification tasks. Finally,FlashAttentionyields thefirst Transformerthat can achieve
better-than-random performance on the challenging Path-X task (sequence length 16K), and block-sparse
FlashAttentionyields thefirst sequence modelthat we know of that can achieve better-than-random
performance on Path-256 (sequence length 64K).
- Benchmarking Attention.We measure the runtime and memory performance ofFlashAttention
    and block-sparseFlashAttentionbased on sequence length. We confirm that the memory footprint
    ofFlashAttentionscales linearly with seq. length and is up to 3faster than standard attention for
    common seq. lengths (up to 2K). We confirm that runtime of block-sparseFlashAttentionscales linearly
    in seq. length and is faster than all existing approximate attention baselines.
Additional experiment details are in Appendix E.

### 4.1 Faster Models withFlashAttention

```
BERT. FlashAttentionyields the fastest single-node BERT training speed that we know of. We train a
BERT-large [ 22 ] model withFlashAttentionon Wikipedia. Table 1 compares our training time to the
implementation from Nvidia that set the training speed record for MLPerf 1.1 [ 58 ]. Our implementation is
15% faster.
Table 1: Training time of BERT-large, starting from the same initialization provided by the MLPerf benchmark, to
reach the target accuracy of 72.0% on masked language modeling. Averaged over 10 runs on 8A100 GPUs.
```
```
BERT Implementation Training time (minutes)
Nvidia MLPerf 1.1 [58] 20.01.
FlashAttention(ours) 17.41.
```
```
GPT-2. FlashAttentionyields faster training times for GPT-2 [ 67 ] on the large OpenWebtext dataset [ 32 ]
than the widely used HuggingFace [ 87 ] and Megatron-LM [ 77 ] implementations. Table 2 shows up to 3end-
to-end speedup compared to Huggingface and 1.7speedup compared to Megatron-LM.FlashAttention
```

achieves the same perplexity as the other two implementations, as we do not change the model definition.
Appendix E includes plots of the validation perplexity throughout training, confirming thatFlashAttention
is as numerically stable as the baselines and produces the same training / validation curves.
Table 2: GPT-2 small and medium usingFlashAttentionachieve up to 3speed up compared to Huggingface
implementation and up to 1.7compared to Megatron-LM. Training time reported on 8A100s GPUs.

```
Model implementations OpenWebText (ppl) Training time (speedup)
GPT-2 small - Huggingface [87] 18.2 9.5 days (1.0)
GPT-2 small - Megatron-LM [77] 18.2 4.7 days (2.0)
GPT-2 small -FlashAttention 18.2 2.7 days (3.5)
GPT-2 medium - Huggingface [87] 14.2 21.0 days (1.0)
GPT-2 medium - Megatron-LM [77] 14.3 11.5 days (1.8)
GPT-2 medium -FlashAttention 14.3 6.9 days (3.0)
```
Long-range Arena. We compare vanilla Transformer (with either standard implementation orFlashAt-
tention) on the long-range arena (LRA [ 80 ]) benchmark. We measure accuracy, throughput, and training
time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the
implementation and experimental setting in Tay et al.[80]and Xiong et al.[90].^3 Table 3 shows thatFlashAt-
tentionachieves up 2.4speed-up compared to standard attention. Block-sparseFlashAttentionis
faster than all of the approximate attention methods that we have tested.
Table 3: The performance of standard attention,FlashAttention, block-sparseFlashAttention, and approximate
attention baselines on the Long-Range-Arena benchmarks.

```
Models ListOps Text Retrieval Image Pathfinder Avg Speedup
Transformer 36.0 63.6 81.6 42.3 72.7 59.3 -
FlashAttention 37.6 63.9 81.4 43.5 72.7 59.8 2.4
Block-sparseFlashAttention 37.0 63.0 81.3 43.6 73.3 59.6 2.8
Linformer [84] 35.6 55.9 77.7 37.8 67.6 54.9 2.5
Linear Attention [50] 38.8 63.2 80.7 42.6 72.5 59.6 2.3
Performer [12] 36.8 63.6 82.2 42.1 69.9 58.9 1.8
Local Attention [80] 36.1 60.2 76.7 40.6 66.6 56.0 1.7
Reformer [51] 36.5 63.8 78.5 39.6 69.4 57.6 1.3
Smyrf [19] 36.1 64.1 79.0 39.6 70.5 57.9 1.7
```
### 4.2 Better Models with Longer Sequences

Language Modeling with Long Context. The runtime and memory-efficiency ofFlashAttention
allow us to increase the context length of GPT-2 by 4while still running faster than the optimized
implementation from Megatron-LM. Table 4 shows that that GPT-2 withFlashAttentionand context
length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better
perplexity.
Table 4: GPT-2 small withFlashAttention, with 4larger context length compared to Megatron-LM, is still 30%
faster while achieving 0.7 better perplexity. Training time on 8A100 GPUs is reported.

```
Model implementations Context length OpenWebText (ppl) Training time (speedup)
GPT-2 small - Megatron-LM 1k 18.2 4.7 days (1.0)
GPT-2 small -FlashAttention 1k 18.2 2.7 days (1.7)
GPT-2 small -FlashAttention 2k 17.6 3.0 days (1.6)
GPT-2 small -FlashAttention 4k 17.5 3.6 days (1.3)
```
Long Document Classification. Training Transformers with longer sequences withFlashAttention
improves performance on the MIMIC-III [ 47 ] and ECtHR [ 6 , 7 ] datasets. MIMIC-III contains intensive care
unit patient discharge summaries, each annotated with multiple labels. ECtHR contains legal cases from the

(^3) LRA accuracy results are known to be highly dependent on the tuning procedure [ 90 ]. Our reproduced baselines perform
better than as reported in the original comparison [80].


```
Attention Memory Usage
```
```
Sequence Length
```
```
Attention Runtime (Fwd Pass + Bwd Pass)
```
```
Sequence Length
```
```
Runtime (ms)
```
```
128 256 512 1024 2048 4096 Memory Footprint (GB) 256 8K 16K 32K 64K
```
```
101
```
```
102
```
```
10
```
```
20
```
```
FlashAttention
Block-Sparse FlashAttention
```
```
PyTorch Attention
Megatron Attention
```
```
Linformer Attention
OpenAI Sparse Attention
```
```
8192
```
```
100
```
```
Crossover Points
```
```
20x
```
```
2x
```
```
Figure 3:Left:runtime of forward pass + backward pass.Right:attention memory usage.
```
```
European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights
that were allegedly violaged. Both of these datasets contain very long text documents; the average number of
tokens in MIMIC is 2,395 tokens, and the longest document contains 14,562 tokens, while the average and
longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the sequence
length of a pretrained RoBERTa model [56] (we repeat the positional embeddings, as in Beltagy et al. [3]).
Table 5 shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length
8K outperforms length 512 by 8.5 points on ECtHR. The discrepancies may be due to subtle distribution
shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift
in the document length, whereas ECtHR contains general language.
```
```
Table 5: Long Document performance (mi-
cro𝐹 1 ) at different sequence lengths using
FlashAttention.
```
```
512 1024 2048 4096 8192 16384
MIMIC-III [47] 52.8 50.7 51.7 54.6 56.4 57.
ECtHR [6] 72.2 74.3 77.1 78.6 80.7 79.
```
```
Table 6: We report the first Transformer
model that can achieve non-random perfor-
mance on Path-X and Path-256.
```
```
Model Path-X Path-
Transformer 77
Linformer [84] 77
Linear Attention [50] 77
Performer [12] 77
Local Attention [80] 77
Reformer [51] 77
SMYRF [19] 77
FlashAttention 61.4 7
Block-sparseFlashAttention 56.0 63.
```
Path-X and Path-256. The Path-X and Path-256 benchmarks are challenging tasks from the long-range
arena benchmark designed to test long context. The task is to classify whether two points in a black and
white 128128 (or 256256) image have a path connecting them, and the images are fed to the transformer
one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved
random performance [ 80 ]. There has been a search for alternative architectures that can model such long
context [ 37 ]. We present here the first result of Transformer models being able to solve Path-X and Path-
(Table 6). We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating
the positional embeddings.FlashAttentionachieves 61.4 accuracy on Path-X. Additionally, block-sparse
FlashAttentionenables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy^4 on
Path-256.

### 4.3 Benchmarking Attention

We vary sequence length and measure runtime and memory usage ofFlashAttentionand block-sparse
FlashAttentionagainst various attention baselines on one A100 GPU with 40 GB HBM, with dropout and
a padding mask. We compare against reference implementations for exact attention, approximate attention,
and sparse attention. We report a subset of baselines in the main body; Appendix E contains more baselines
and full details.

(^4) Path-256 requires longer sequences but has relatively shorter paths than Path-X, so it is easier to obtain a higher accuracy.


```
Runtime. Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass ofFlashAt-
tentionand block-sparseFlashAttentioncompared to the baselines in exact, approximate, and sparse
attention (exact numbers in Appendix E). Runtime grows quadratically with sequence length, butFlashAt-
tentionruns significantly faster thanexact attentionbaselines, up to 3 faster than the PyTorch
implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with se-
quence length, butFlashAttentionstill runs faster than approximate and sparse attention for short
sequences due to fewer memory accesses. Theapproximate attentionruntimes begin to cross over with
FlashAttentionat sequences between 512 and 1024. On the other hand, block-sparseFlashAttention
is faster than all implementations of exact, sparse, and approximate attention that we know of, across all
sequence lengths.
```
```
Memory Footprint. Figure 3 (right) shows the memory footprint ofFlashAttentionand block-sparse
FlashAttentioncompared to various exact, approximate, and sparse attention baselines.FlashAttention
and block-sparseFlashAttentionhave the same memory footprint, which grows linearly with sequence
length.FlashAttentionis up to 20more memory efficient thanexact attentionbaselines, and is more
memory-efficient than theapproximate attentionbaselines. All other algorithms except for Linformer run
out of memory on an A100 GPU before 64K, andFlashAttentionis still 2more efficient than Linformer.
```
## 5 Limitations and Future Directions

We discuss limitations of our approach and future directions. Related work is given in Appendix A.
Compiling to CUDA.Our current approach to building IO-aware implementations of attention requires
writing a new CUDA kernel for each new attention implementation. This requires writing the attention
algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort.
Implementations may also not be transferrable across GPU architectures. These limitations suggest the
need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and
compiling to IO-aware implementations in CUDA—similar to efforts such as Halide in image processing [ 70 ].
IO-Aware Deep Learning. We believe that the IO-aware approach can extend beyond attention.
Attention is the most memory-intensive computation in Transformers, but every layer in a deep network
touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss
these potential extensions in Appendix D.
Multi-GPU IO-Aware Methods.Our IO-aware implementation of attention is optimal within con-
stants for computing attention on a single GPU. However, the attention computation may be parallelizable
across multiple GPUs [ 72 ]. Using multiple GPUs adds an additional layer to IO analysis—accounting for
data transfer between GPUs. We hope our work inspires future work in this direction.

```
Acknowledgments
```
Our implementation uses Apex’s FMHA code (https://github.com/NVIDIA/apex/tree/master/apex/
contrib/csrc/fmha) as a starting point. We thank Young-Jun Ko for the in-depth explanation of his FMHA
implementation and for his thoughtful answers to our questions about CUDA. We thank Sabri Eyuboglu,
Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen, and Xun Huang for their constructive feedback and
suggestions on early drafts of the paper. We thank Markus Rabe and Charles Staats for helpful discussion of
their attention algorithm.
We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos.
CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); ARL under
No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak
Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine
Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba,
TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce,
Total, the HAI-GCP & HAI-Azure Cloud Credits for Research program, the Stanford Data Science Initiative
(SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate
Fellowship (NDSEG) Program, and members of the Stanford DAWN project: Facebook, Google, and
VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes


notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations
expressed in this material are those of the authors and do not necessarily reflect the views, policies, or
endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government. Atri Rudra’s research is
supported by NSF grant CCF-1763481.

## References

```
[1]Alok Aggarwal and S Vitter, Jeffrey. The input/output complexity of sorting and related problems.
Communications of the ACM, 31(9):1116–1127, 1988.
```
```
[2]Irwan Bello. LambdaNetworks: Modeling long-range interactions without attention. arXiv preprint
arXiv:2102.08602, 2021.
```
```
[3]Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer.arXiv
preprint arXiv:2004.05150, 2020.
```
```
[4]L Susan Blackford, Antoine Petitet, Roldan Pozo, Karin Remington, R Clint Whaley, James Demmel,
Jack Dongarra, Iain Duff, Sven Hammarling, Greg Henry, et al. An updated set of basic linear algebra
subprograms (blas).ACM Transactions on Mathematical Software, 28(2):135–151, 2002.
```
```
[5]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.
Advances in neural information processing systems, 33:1877–1901, 2020.
```
```
[6]Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. Neural legal judgment prediction in English.
InProceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages
4317–4323, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1424.
URLhttps://www.aclweb.org/anthology/P19-1424.
```
```
[7]Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and
Prodromos Malakasiotis. Paragraph-level rationale extraction through regularization: A case study on
european court of human rights cases. InProceedings of the Annual Conference of the North American
Chapter of the Association for Computational Linguistics, Mexico City, Mexico, 2021. Association for
Computational Linguistics.
```
```
[8]Benjamin Charlier, Jean Feydy, Joan Alexis Glaunès, François-David Collin, and Ghislain Durif. Kernel
operations on the gpu, with autodiff, without memory overflows.Journal of Machine Learning Research,
22(74):1–6, 2021. URLhttp://jmlr.org/papers/v22/20-275.html.
```
```
[9]Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying
sparse and low-rank attention. InAdvances in Neural Information Processing Systems (NeurIPS), 2021.
```
[10]Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory
cost.arXiv preprint arXiv:1604.06174, 2016.

[11]Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse
transformers.arXiv preprint arXiv:1904.10509, 2019.

[12]Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane,
Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking
attention with performers. InInternational Conference on Learning Representations (ICLR), 2020.

[13]Xiang Dai, Ilias Chalkidis, Sune Darkner, and Desmond Elliott. Revisiting transformer-based models for
long document classification.arXiv preprint arXiv:2204.06683, 2022.

[14]Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov.
Transformer-XL: Attentive language models beyond a fixed-length context. InProceedings of the 57th
Annual Meeting of the Association for Computational Linguistics, pages 2978–2988, 2019.


[15]Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and Christopher Ré. Learning fast algorithms
for linear transforms using butterfly factorizations. InInternational Conference on Machine Learning
(ICML), 2019.

[16]Tri Dao, Nimit Sohoni, Albert Gu, Matthew Eichhorn, Amit Blonder, Megan Leszczynski, Atri Rudra,
and Christopher Ré. Kaleidoscope: An efficient, learnable representation for all structured linear maps.
InInternational Conference on Learning Representations (ICLR), 2020.

[17]Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Ré.
Pixelated butterfly: Simple and efficient sparse training for neural network models. InInternational
Conference on Learning Representations (ICLR), 2022.

[18]Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh
Rao, Atri Rudra, and Christopher Ré. Monarch: Expressive structured matrices for efficient and accurate
training. InInternational Conference on Machine Learning (ICML), 2022.

[19]Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-efficient attention
using asymmetric clustering.Advances in Neural Information Processing Systems, 33:6476–6489, 2020.

[20]Christopher De Sa, Albert Gu, Rohan Puttagunta, Christopher Ré, and Atri Rudra. A two-pronged
progress in structured dense matrix vector multiplication. InProceedings of the Twenty-Ninth Annual
ACM-SIAM Symposium on Discrete Algorithms, pages 1060–1079. SIAM, 2018.

[21]Peter J Denning. The working set model for program behavior.Communications of the ACM, 11(5):
323–333, 1968.

[22]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep
bidirectional transformers for language understanding. 2019.

[23]Xin Dong, Shangyu Chen, and Sinno Jialin Pan. Learning to prune deep neural networks via layer-wise
optimal brain surgeon.arXiv preprint arXiv:1705.07565, 2017.

[24]Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is
worth 16x16 words: Transformers for image recognition at scale. InInternational Conference on Learning
Representations, 2020.

[25]Y Eidelman and I Gohberg. On a new class of structured matrices.Integral Equations and Operator
Theory, 34(3):293–324, 1999.

[26]Jean Feydy, Joan Glaunès, Benjamin Charlier, and Michael Bronstein. Fast geometric learning with
symbolic matrices.Advances in Neural Information Processing Systems, 33, 2020.

[27] Jörg Flum and Martin Grohe.Parameterized Complexity Theory. Springer, 2006.

[28]Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural
networks. InInternational Conference on Learning Representations, 2018.

[29]Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. Stabilizing the
lottery ticket hypothesis.arXiv preprint arXiv:1903.01611, 2019.

[30]Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity
and the lottery ticket hypothesis. InInternational Conference on Machine Learning, pages 3259–3269.
PMLR, 2020.

[31]Karan Goel, Albert Gu, Chris Donahue, and Christopher Ré. It’s raw! audio generation with state-space
models. InInternational Conference on Machine Learning (ICML), 2022.

[32] Aaron Gokaslan, Vanya Cohen, Pavlick Ellie, and Stefanie Tellex. Openwebtext corpus, 2019.


[33]Jim Gray, Surajit Chaudhuri, Adam Bosworth, Andrew Layman, Don Reichart, Murali Venkatrao,
Frank Pellow, and Hamid Pirahesh. Data cube: A relational aggregation operator generalizing group-by,
cross-tab, and sub-totals.Data mining and knowledge discovery, 1(1):29–53, 1997.

[34]Andreas Griewank and Andrea Walther.Evaluating derivatives: principles and techniques of algorithmic
differentiation. SIAM, 2008.

[35]Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. Hippo: Recurrent memory with
optimal polynomial projections. InAdvances in neural information processing systems (NeurIPS), 2020.

[36]Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. Combining
recurrent, convolutional, and continuous-time models with linear state space layers.Advances in Neural
Information Processing Systems, 34, 2021.

[37]Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state
spaces. InThe International Conference on Learning Representations (ICLR), 2022.

[38]Song Han, Jeff Pool, John Tran, and William J Dally. Learning both weights and connections for efficient
neural networks.arXiv preprint arXiv:1506.02626, 2015.

[39]Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks
with pruning, trained quantization and huffman coding. InInternational Conference on Learning
Representations, 2016.

[40]John Hennessy and David Patterson. Memory hierarchy design.Computer Architecture: A Quantitative
Approach, pages 390–525, 2003.

[41] Sara Hooker. The hardware lottery.arXiv preprint arXiv:2009.06489, 2020.

[42]Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V Le. Transformer quality in linear time. arXiv
preprint arXiv:2202.10447, 2022.

[43]Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, and Torsten Hoefler. Data movement is all
you need: A case study on optimizing transformers.Proceedings of Machine Learning and Systems, 3:
711–732, 2021.

[44]Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via microbenchmarking. GPU
Technology Conference, 2021.

[45]Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU
architecture via microbenchmarking.arXiv preprint arXiv:1804.06826, 2018.

[46]Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. Dissecting the graphcore IPU
architecture via microbenchmarking.arXiv preprint arXiv:1912.03413, 2019.

[47]Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi,
Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible
critical care database.Scientific data, 3(1):1–9, 2016.

[48]Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah
Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-datacenter performance analysis of a tensor
processing unit. InProceedings of the 44th annual international symposium on computer architecture,
pages 1–12, 2017.

[49]Thomas Kailath, Sun-Yuan Kung, and Martin Morf. Displacement ranks of matrices and linear equations.
Journal of Mathematical Analysis and Applications, 68(2):395–407, 1979.

[50]Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs:
Fast autoregressive transformers with linear attention. InInternational Conference on Machine Learning,
pages 5156–5165. PMLR, 2020.


[51]Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. InThe
International Conference on Machine Learning (ICML), 2020.

[52]Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut.
Albert: A lite BEDRT for self-supervised learning of language representations. InThe International
Conference on Learning Representations (ICLR), 2020.

[53]Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan,
Guangwen Yang, and Depei Qian. The deep learning compiler: A comprehensive survey. IEEE
Transactions on Parallel and Distributed Systems, 32(3):708–727, 2020.

[54]Valerii Likhosherstov, Krzysztof Choromanski, Jared Davis, Xingyou Song, and Adrian Weller. Sub-linear
memory: How to make performers slim.arXiv preprint arXiv:2012.11346, 2020.

[55]Ji Lin, Yongming Rao, Jiwen Lu, and Jie Zhou. Runtime neural pruning. In I. Guyon, U. V. Luxburg,
S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural
Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[56]Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach.
arXiv preprint arXiv:1907.11692, 2019.

[57]Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer.
Luna: Linear unified nested attention.Advances in Neural Information Processing Systems, 34, 2021.

[58]Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson,
Hanlin Tang, Gu-Yeon Wei, Peter Bailis, Victor Bittorf, et al. Mlperf training benchmark.Proceedings
of Machine Learning and Systems, 2:336–349, 2020.

[59]Frank McSherry, Michael Isard, and Derek G Murray. Scalability! but at whatfCOSTg? In15th
Workshop on Hot Topics in Operating Systems (HotOS XV), 2015.

[60]Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint
arXiv:1805.02867, 2018.

[61] NVIDIA. Nvidia Tesla V100 GPU architecture, 2017.

[62] NVIDIA. Nvidia A100 tensor core GPU architecture, 2020.

[63] NVIDIA. Nvidia H100 tensor core GPU architecture, 2022.

[64]D Stott Parker. Random butterfly transformations with applications in computational linear algebra.
1995.

[65]Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-
performance deep learning library.Advances in neural information processing systems, 32, 2019.

[66]Markus N Rabe and Charles Staats. Self-attention does not need𝑂¹𝑛^2 ºmemory. arXiv preprint
arXiv:2112.05682, 2021.

[67]Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners.OpenAI blog, 1(8):9, 2019.

[68]Jack Rae and Ali Razavi. Do transformers need deep long-range memory? InProceedings of the 58th
Annual Meeting of the Association for Computational Linguistics, Online, July 2020. Association for
Computational Linguistics. URLhttps://www.aclweb.org/anthology/2020.acl-main.672.

[69]Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. Compressive trans-
formers for long-range sequence modelling. InThe International Conference on Learning Representations
(ICLR), 2020.


[70]Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman
Amarasinghe. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in
image processing pipelines.Acm Sigplan Notices, 48(6):519–530, 2013.

[71]Raghu Ramakrishnan, Johannes Gehrke, and Johannes Gehrke.Database management systems, volume 3.
McGraw-Hill New York, 2003.

[72]Benjamin Recht and Christopher Ré. Parallel stochastic gradient algorithms for large-scale matrix
completion.Mathematical Programming Computation, 5(2):201–226, 2013.

[73]Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, and Bo Dai.
Combiner: Full attention transformer with sparse computation cost.Advances in Neural Information
Processing Systems, 34, 2021.

[74]Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse
attention with routing transformers.Transactions of the Association for Computational Linguistics, 9:
53–68, 2021.

[75] Amit Sabne. XLA: Compiling machine learning for peak performance. 2020.

[76]Victor Sanh, Thomas Wolf, and Alexander M Rush. Movement pruning: Adaptive sparsity by fine-tuning.
arXiv preprint arXiv:2005.07683, 2020.

[77]Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro.
Megatron-LM: Training multi-billion parameter language models using model parallelism.arXiv preprint
arXiv:1909.08053, 2019.

[78]Vikas Sindhwani, Tara Sainath, and Sanjiv Kumar. Structured transforms for small-footprint deep
learning. InAdvances in Neural Information Processing Systems, pages 3088–3096, 2015.

[79]Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span
in transformers. InProceedings of the Annual Meeting of the Association for Computational Linguistics,
2019.

[80]Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu
Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers.
InInternational Conference on Learning Representations, 2020.

[81]Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey.arXiv
preprint arXiv:2009.06732, 2020.

[82]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing
systems, 30, 2017.

[83]Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet:
Scaling transformers to 1,000 layers.arXiv preprint arXiv:2203.00555, 2022.

[84]Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with
linear complexity.arXiv preprint arXiv:2006.04768, 2020.

[85]Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance
model for multicore architectures.Communications of the ACM, 52(4):65–76, 2009.

[86]Michael E Wolf and Monica S Lam. A data locality optimizing algorithm. InProceedings of the ACM
SIGPLAN 1991 conference on Programming language design and implementation, pages 30–44, 1991.


[87]Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric
Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen,
Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame,
Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing.
InProceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System
Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics. URL
https://www.aclweb.org/anthology/2020.emnlp-demos.6.

[88]David P Woodruff. Optimal space lower bounds for all frequency moments. InSODA, volume 4, pages
167–175. Citeseer, 2004.

[89]Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with
lightweight and dynamic convolutions. InThe International Conference on Learning Representations
(ICLR), 2019.

[90]Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas
Singh. Nyströmformer: A nystöm-based algorithm for approximating self-attention. InProceedings of
the AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence, volume 35,
page 14138, 2021.

[91]Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng,
and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In
Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 558–567, 2021.

[92]Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago
Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer
sequences.Advances in Neural Information Processing Systems, 33, 2020.

[93]Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh
Susskind. An attention free transformer.arXiv preprint arXiv:2105.14103, 2021.

[94]Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and
Bryan Catanzaro. Long-short transformer: Efficient transformers for language and vision.Advances in
Neural Information Processing Systems, 34, 2021.


## A Related Work

```
IO-Aware Runtime Optimization.The broad concept of optimizing for reading and writing to fast/slow
memory has a long history in computer science and has been known by many names. We draw the most
direct connection to the literature of analyzing I/O complexity in this work [ 1 ], but concepts of memory
hierarchies are fundamental and has appeared in many forms, from the working set model [ 21 ], to data
locality [ 86 ], to the Roofline model of arithmetic intensity [ 85 ], to analyses of scalability [ 59 ], to standard
textbook treatments of computer architecture [ 40 ]. We hope that this work encourages the community to
adopt these ideas in more parts of the deep learning stack.
Efficient ML Models with Structured Matrices.Matrix multiply is the core computational bottle-
neck of most machine learning models. To reduce the computational complexity, there have been numerous
approaches to learn over a more efficient set of matrices. These matrices are calledstructured matrices, which
have subquadratic (𝑜¹𝑛^2 ºfor dimension𝑛𝑛) number of parameters and runtime. Most common examples
of structured matrices are sparse and low-rank matrices, along with fast transforms commonly encountered
in signal processing (Fourier, Chebyshev, sine/cosine, orthogonal polynomials). There have been several
more general classes of structured matrices proposed in machine learning: Toeplitz-like [ 78 ], low-displacement
rank [ 49 ], quasi-separable [ 25 ]). The butterfly pattern we use for our block-sparse attention is motivated
by the fact that butterfly matrices [ 15 , 64 ] and their products have been shown to be able to express any
structured matrices with almost optimal runtime and number of parameters [ 16 , 20 ]. However, even though
structured matrices are efficient in theory, they have not seen wide adoption since it is hard to translate their
efficiency to wall-clock speedup since dense unconstrained matrix multiply has very optimize implementation,
a phenomenon known as the hardware lottery [ 41 ]. Extensions of butterfly matrices [ 17 , 18 ] aimed to make
butterfly matrices more hardware-friendly.
Sparse Training.Our block-sparseFlashAttentioncan be seen as a step towards making sparse model
training more efficient. Sparse models have seen success in compressing models for inference (pruning) by
sparsifying the weight matrices [ 23 , 38 , 39 , 55 , 76 ]. For model training, the lottery tickets hypothesis [ 28 , 29 , 30 ]
suggests that there are a set of small sub-networks derived from a larger dense network that performs as
well as the original dense network. Out block-sparseFlashAttentioncan also be seen as a fixed lottery
ticket in the context of attention: we fix the sparsity pattern to be the butterfly pattern through training,
and observe that it performs almost as well as the (dense)FlashAttentionon the Long-range Arena tasks.
Efficient Transformer.Transformer-based models have become the most widely-used architecture in
natural language processing [ 22 ] and computer vision [ 24 , 91 ]. However, one of their computational bottlenecks
is that their time and memory scales quadratic in the sequence length. There are numerous approaches to
overcome this bottleneck, including approximation with hashing (i.e., sparse) such as Reformer [ 51 ] and
Smyrf [ 19 ] and with low-rank approximation such as Performer [ 12 , 54 ]. One can even combine sparse and
low-rank approximation for better accuracy (e.g., Longformer [ 3 ], BigBird [ 92 ], Scatterbrain [ 9 ], Long-short
transformer [ 94 ], Combiner [ 73 ]). Other approaches include compressing along the sequence dimension to
attend to multiple tokens at once [ 52 , 57 , 79 , 89 ]. One can also attend over the states from previous sequences
to help lengthen the context (e.g., Transformer-XL [ 14 ] and Compressive Transformer [ 69 ]). We recommend
the survey [81] for more details.
There are several lines of work on developing other modules instead of attention to model longer context.
HiPPO [ 35 ] and its extensions, most notably S4 [ 31 , 36 , 37 ] projects the history on a polynomial basis,
allowing accurate reconstruction of the history through state-space models. They combine the strengths of
CNNs (efficient training), RNNs (efficient inference), and continuous models (robust to change in sampling
rates). LambdaNetworks [ 2 ], AFT [ 93 ] and FLASH [ 42 ] are other attempts at replacing attention in the
context of image classification and language modeling.
```
## B Algorithm Details

We first derive the forward and backward passes of attention and show that they can be computed in a
memory-efficient manner (requiring extra memory linear instead of quadratic in the sequence length). Though
they reduce the amount of extra memory required, naively they still incur quadratic HBM accesses, resulting
in slower execution speed. We describe theFlashAttentionalgorithm to implement both the forward


```
and the backward passes on GPUs that reduces HBM accesses, leading to both faster runtime and smaller
memory footprint.
```
### B.1 Memory-efficient forward pass

```
The main challenge in making attention memory-efficient is the softmax that couples the columns ofK(and
columns ofV). Our approach is to compute the softmax normalization constant separately to decouple the
columns. This technique [ 60 ] has been used in the literature [ 51 , 66 ] to show that attention computation
does not need quadraticextramemory (though the number of HBM accesses is still quadratic, resulting in
slow run-time).
For simplicity, we omit here the max-shifting step during softmax. The full algorithm in Appendix B.
contains all the steps.
Recall that given input sequencesQKV 2 R𝑁𝑑, we want to compute the attention outputO 2 R𝑁𝑑:
```
```
S=QK> 2 R𝑁𝑁 P=softmax¹Sº 2R𝑁𝑁 O=PV 2 R𝑁𝑑
```
```
We have that𝑆𝑖𝑗=𝑞𝑇𝑖𝑘𝑗where𝑞𝑖and𝑘𝑗are the𝑖-th and𝑗-th columns ofQandKrespectively. Define
the normalization constants of softmax:
𝐿𝑖=
```
##### ∑︁

```
𝑗
```
##### 𝑒𝑞

```
𝑖𝑇𝑘𝑗
 (1)
```
```
Let𝑣𝑗be the𝑗-th column ofV, then the𝑖-th columns of the output is
```
##### 𝑜𝑖=𝑃𝑖:V=

##### ∑︁

```
𝑗
```
##### 𝑃𝑖𝑗𝑣𝑗=

##### ∑︁

```
𝑗
```
##### 𝑒𝑞

```
𝑇𝑖𝑘𝑗
```
```
𝐿𝑖
```
##### 𝑣𝑗 (2)

```
We see that once𝐿𝑖is computed, we can compute𝑜𝑖without extra memory by repeatedly summing
𝑒𝑞𝑇𝑖𝑘𝑗
𝐿𝑖 𝑣𝑗. Therefore the forward pass can be computed with𝑂¹𝑛ºextra memory:
```
1. Compute𝐿𝑖for all𝑖according to Eq. (1), which takes𝑂¹𝑛ºextra memory.
2. Compute𝑜𝑖for all𝑖according to Eq. (2), which takes𝑂¹𝑑ºextra memory.

### B.2 Memory-efficient backward pass

We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe
and Staats[66]suggests that the backward pass can be done without quadratic extra memory by applying
gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly
and show how it can be computed in a memory-efficient manner.
Suppose that there is a scalar loss function𝜙, and let the output gradient bedO 2 R𝑛𝑑(wheredOdenotes
𝜕𝜙
𝜕O). We want to compute the input gradientsdQdKdV^2 R

```
𝑛𝑑 (wheredQdKdVdenote 𝜕𝜙
𝜕Q
```
```
𝜕𝜙
𝜕K
```
𝜕𝜙
𝜕V
respectively).
The gradientdVis easy to see. Applying reverse-mode autodiff by hand (aka the chain rule), we obtain
(in matrix notation)dV=P𝑇dO. Thus:

##### 𝑑𝑣𝑗=

##### ∑︁

```
𝑖
```
##### 𝑃𝑖𝑗𝑑𝑜𝑖=

##### ∑︁

```
𝑖
```
##### 𝑒𝑞

```
𝑇
𝑖𝑘𝑗
𝐿𝑖
```
##### 𝑑𝑜𝑖 (3)

```
Since we already computed𝐿𝑖,𝑑𝑣𝑗can be computed without extra memory by repeated summing.
The gradientsdQanddKare a little more complicated. We go through the gradientsdPanddSfirst.
From Eq. (2), we have thatdP=dOV𝑇, and so:
```
```
𝑑𝑃𝑖𝑗=𝑑𝑜𝑇𝑖𝑣𝑗
```
```
Recall that𝑃𝑖:=softmax¹𝑆𝑖:º. Using the fact that the Jacobian of𝑦=softmax¹𝑥ºisdiag¹𝑦º𝑦𝑦𝑇, we
have that
𝑑𝑆𝑖:=¹diag¹𝑃𝑖:º𝑃𝑖:𝑃𝑇𝑖:º𝑑𝑃𝑖:=𝑃𝑖:𝑑𝑃𝑖:¹𝑃𝑇𝑖:𝑑𝑃𝑖:º𝑃𝑖:
```

```
wheredenotes pointwise multiplication.
Define
𝐷𝑖=𝑃𝑇𝑖:𝑑𝑃𝑖:=
```
##### ∑︁

```
𝑗
```
##### 𝑒𝑞

```
𝑇𝑖𝑘𝑗
```
```
𝐿𝑖
```
##### 𝑑𝑜𝑇𝑖𝑣𝑗=𝑑𝑜𝑇𝑖

##### ∑︁

```
𝑗
```
##### 𝑒𝑞

```
𝑖>𝑘𝑗
𝐿𝑖
```
##### 𝑣𝑗=𝑑𝑜𝑇𝑖𝑜𝑖 (4)

```
then
𝑑𝑆𝑖:=𝑃𝑖:𝑑𝑃𝑖:𝐷𝑖𝑃𝑖:
Hence
𝑑𝑆𝑖𝑗=𝑃𝑖𝑗𝑑𝑃𝑖𝑗𝐷𝑖𝑃𝑖𝑗=𝑃𝑖𝑗¹𝑑𝑃𝑖𝑗𝐷𝑖º
Now we can get the gradientsdQanddK. Recall that𝑆𝑖𝑗=𝑞𝑇𝑖𝑘𝑗, so
```
##### 𝑑𝑞𝑖=

##### ∑︁

```
𝑗
```
##### 𝑑𝑆𝑖𝑗𝑘𝑗=

##### ∑︁

```
𝑗
```
##### 𝑃𝑖𝑗¹𝑑𝑃𝑖𝑗𝐷𝑖º𝑘𝑗=

##### ∑︁

```
𝑗
```
##### 𝑒𝑞

```
𝑖𝑇𝑘𝑗
𝐿𝑖
```
##### ¹𝑑𝑜𝑇𝑖𝑣𝑗𝐷𝑖º𝑘𝑗 (5)

```
Similarly,
```
```
𝑑𝑘𝑗=
```
##### ∑︁

```
𝑖
```
##### 𝑑𝑆𝑖𝑗𝑞𝑖=

##### ∑︁

```
𝑖
```
##### 𝑃𝑖𝑗¹𝑑𝑃𝑖𝑗𝐷𝑖º𝑞𝑖=

##### ∑︁

```
𝑖
```
##### 𝑒𝑞

```
𝑖𝑇𝑘𝑗
𝐿𝑖
```
##### ¹𝑑𝑜𝑇𝑖𝑣𝑗𝐷𝑖º𝑞𝑖 (6)

```
Therefore the backward pass can also be computed with𝑂¹𝑛ºextra memory:
```
1. Compute𝑑𝑣𝑗for all𝑗according to Eq. (3), which takes𝑂¹𝑑ºextra memory.
2. Compute𝐷𝑖for all𝑖according to Eq. (4), which takes𝑂¹𝑛ºextra memory.
3. Compute𝑑𝑞𝑖for all𝑖according to Eq. (5), which takes𝑂¹𝑑ºextra memory.
4. Compute𝑑𝑘𝑗for all𝑗according to Eq. (6), which takes𝑂¹𝑑ºextra memory.

### B.3 FlashAttention: Forward Pass

We describe the full details ofFlashAttentionforward pass. Given input sequencesQKV 2 R𝑁𝑑, we
want to compute the attention outputO 2 R𝑁𝑑:

```
S=𝜏QK> 2 R𝑁𝑁 Smasked=mask¹𝑆º 2R𝑁𝑁 P=softmax¹Smaskedº 2R𝑁𝑁
Pdropped=dropout¹P 𝑝dropº O=PdroppedV 2 R𝑁𝑑
```
```
where𝜏 2 Ris some softmax scaling (typicallyp^1 𝑑),maskis some masking function that sets some entries of
the input to1and keep other entries the same (e.g., key padding mask when sequences in the batch don’t
have the same lengths and are padded), anddropout¹𝑥 𝑝ºapplies dropout to𝑥elementwise (i.e., output 1 𝑥𝑝
with probability 1 𝑝and output 0 with probability𝑝for each element𝑥).
The full algorithm is in Algorithm 2. We save the outputO, the softmax statisticsℓand𝑚, and the
pseudo-random number generator stateRfor the backward pass.
```

```
Algorithm 2FlashAttentionForward Pass
Require:MatricesQKV 2 R𝑁𝑑in HBM, on-chip SRAM of size𝑀, softmax scaling constant𝜏 2 R,
masking functionmask, dropout probability𝑝drop.
1:Initialize the pseudo-random number generator stateRand save to HBM.
2:Set block sizes𝐵𝑐=
```
##### 𝑀

```
4 𝑑
```
##### 

```
 𝐵𝑟=min
```
##### 𝑀

```
4 𝑑
```
##### 

#####  𝑑

##### 

##### .

```
3:InitializeO=¹ 0 º𝑁𝑑 2 R𝑁𝑑ℓ=¹ 0 º𝑁 2 R𝑁𝑚=¹1º𝑁 2 R𝑁in HBM.
4:DivideQinto𝑇𝑟=
```
```
l
𝑁
𝐵𝑟
```
```
m
blocksQ 1 Q𝑇𝑟of size𝐵𝑟𝑑each, and divideKVin to𝑇𝑐=
```
```
l
𝑁
𝐵𝑐
```
```
m
blocks
K 1 K𝑇𝑐andV 1 V𝑇𝑐, of size𝐵𝑐𝑑each.
5:DivideOinto𝑇𝑟blocksO𝑖O𝑇𝑟of size𝐵𝑟𝑑each, divideℓinto𝑇𝑟blocksℓ𝑖ℓ𝑇𝑟of size𝐵𝑟each,
divide𝑚into𝑇𝑟blocks𝑚 1 𝑚𝑇𝑟of size𝐵𝑟each.
6:for 1 𝑗𝑇𝑐do
7: LoadK𝑗V𝑗from HBM to on-chip SRAM.
8: for 1 𝑖𝑇𝑟do
9: LoadQ𝑖O𝑖ℓ𝑖𝑚𝑖from HBM to on-chip SRAM.
10: On chip, computeS𝑖𝑗=𝜏Q𝑖K𝑇𝑗 2 R𝐵𝑟𝐵𝑐.
11: On chip, computeSmasked𝑖𝑗 =mask¹S𝑖𝑗º.
12: On chip, compute𝑚 ̃𝑖𝑗=rowmax¹Smasked𝑖𝑗 º 2R𝐵𝑟,P ̃𝑖𝑗=exp¹Smasked𝑖𝑗 𝑚 ̃𝑖𝑗º 2R𝐵𝑟𝐵𝑐(pointwise),
ℓ ̃𝑖𝑗=rowsum¹P ̃𝑖𝑗º 2R𝐵𝑟.
13: On chip, compute𝑚new𝑖 =max¹𝑚𝑖𝑚 ̃𝑖𝑗º 2R𝐵𝑟,ℓnew𝑖 =𝑒𝑚𝑖𝑚
new𝑖
ℓ𝑖 ̧𝑒𝑚 ̃𝑖𝑗𝑚
new𝑖 ̃
ℓ𝑖𝑗 2 R𝐵𝑟.
14: On chip, computeP ̃dropped𝑖𝑗 =dropout¹P ̃𝑖𝑗 𝑝dropº.
15: WriteO𝑖 diag¹ℓ𝑖newº^1 ¹diag¹ℓ𝑖º𝑒𝑚𝑖𝑚
```
```
new𝑖
O𝑖 ̧𝑒𝑚 ̃𝑖𝑗𝑚
```
```
new𝑖 ̃
Pdropped𝑖𝑗 V𝑗ºto HBM.
16: Writeℓ𝑖 ℓ𝑖new,𝑚𝑖 𝑚new𝑖 to HBM.
17: end for
18:end for
19:ReturnOℓ𝑚R.
```
### B.4 FlashAttention: Backward Pass

We describe the full details ofFlashAttentionbackward pass. Given input sequencesQKV 2 R𝑁𝑑, the
outputO 2 R𝑁𝑑, and the output gradientdO, we want to compute the input gradientsdQdKdV 2 R𝑁𝑑.
We first describe the standard attention backward pass in Algorithm 3 for completeness.

```
Algorithm 3Standard Attention Backward Pass
Require:MatricesQKVdO 2 R𝑁𝑑,P 2 R𝑁𝑁in HBM.
1:LoadPdOby blocks from HBM, computedV=P>dO 2 R𝑁𝑑, writedVto HBM.
2:LoaddOVby blocks from HBM, computedP=dOV> 2 R𝑁𝑁, writedPto HBM.
3:ReadPdPfrom HBM, computedS 2 R𝑁𝑁where𝑑𝑆𝑖𝑗=𝑃𝑖𝑗¹𝑑𝑃𝑖𝑗
```
##### Í

```
𝑙𝑃𝑖𝑙𝑑𝑃𝑖𝑙º, writedSto HBM.
4:LoaddSandKby blocks from HBM, computedQ=dSK, writedQto HBM.
5:LoaddSandQby blocks from HBM, computedK=dS>Q, writedKto HBM.
6:ReturndQdKdV.
```
```
We now make two observations aboutFlashAttentionbackward pass:
```
```
1.We do not need to store the dropout mask of size𝑂¹𝑁^2 ºfrom the forward pass. Instead, we can save
the pseudo-random number generator states from the forward pass and re-generate the dropout mask
in the backward pass. This allows us to only use𝑂¹𝑁ºextra memory.
```
```
2.When computing the softmax gradient, we use Eq. (4) to compute𝐷𝑖=𝑃>𝑖:𝑑𝑃𝑖:without reducing over
𝑃𝑖:and𝑑𝑃𝑖:of size𝑁(they might not fit into SRAM). Instead we can rewrite𝐷𝑖=𝑑𝑜>𝑖𝑜𝑖and compute
the dot product between vectors of size𝑑.
```

```
The fullFlashAttentionbackward pass algorithm is in Algorithm 4. Conceptually it is just a block
version of the derivation in Appendix B.2.
```
```
Algorithm 4FlashAttentionBackward Pass
Require:MatricesQKVOdO 2 R𝑁𝑑in HBM, vectorsℓ𝑚 2 R𝑁in HBM, on-chip SRAM of size𝑀,
softmax scaling constant𝜏 2 R, masking functionmask, dropout probability𝑝drop, pseudo-random
number generator stateRfrom the forward pass.
1:Set the pseudo-random number generator state toR.
2:Set block sizes𝐵𝑐=
```
##### 𝑀

```
4 𝑑
```
##### 

```
 𝐵𝑟=min
```
##### 𝑀

```
4 𝑑
```
##### 

#####  𝑑

##### 

##### .

```
3:DivideQinto𝑇𝑟=
```
```
l
𝑁
𝐵𝑟
```
```
m
blocksQ 1 Q𝑇𝑟of size𝐵𝑟𝑑each, and divideKVin to𝑇𝑐=
```
```
l
𝑁
𝐵𝑐
```
```
m
blocks
K 1 K𝑇𝑐andV 1 V𝑇𝑐, of size𝐵𝑐𝑑each.
4:DivideOinto𝑇𝑟blocksO𝑖O𝑇𝑟of size𝐵𝑟𝑑each, dividedOinto𝑇𝑟blocksdO𝑖dO𝑇𝑟of size
𝐵𝑟𝑑each, divideℓinto𝑇𝑟blocksℓ𝑖ℓ𝑇𝑟of size𝐵𝑟each, divide𝑚into𝑇𝑟blocks𝑚 1 𝑚𝑇𝑟of size
𝐵𝑟each.
5:InitializedQ=¹ 0 º𝑁𝑑in HBM and divide it into𝑇𝑟blocksdQ 1 dQ𝑇𝑟of size𝐵𝑟𝑑each. Initialize
dK=¹ 0 º𝑁𝑑dV=¹ 0 º𝑁𝑑in HBM and dividedKdVin to𝑇𝑐blocksdK 1 dK𝑇𝑐anddV 1 dV𝑇𝑐,
of size𝐵𝑐𝑑each.
6:for 1 𝑗𝑇𝑐do
7: LoadK𝑗V𝑗from HBM to on-chip SRAM.
8: InitializedK ̃ 𝑗=¹ 0 º𝐵𝑐𝑑dV ̃𝑗=¹ 0 º𝐵𝑐𝑑on SRAM.
9: for 1 𝑖𝑇𝑟do
10: LoadQ𝑖O𝑖dO𝑖dQ𝑖ℓ𝑖𝑚𝑖from HBM to on-chip SRAM.
11: On chip, computeS𝑖𝑗=𝜏Q𝑖K𝑇𝑗 2 R𝐵𝑟𝐵𝑐.
12: On chip, computeSmasked𝑖𝑗 =mask¹S𝑖𝑗º.
13: On chip, computeP𝑖𝑗=diag¹𝑙𝑖º^1 exp¹Smasked𝑖𝑗 𝑚𝑖º 2R𝐵𝑟𝐵𝑐.
14: On chip, compute dropout maskZ𝑖𝑗 2 R𝐵𝑟𝐵𝑐where each entry has value 1 𝑝^1 dropwith probability
1 𝑝dropand value 0 with probability𝑝drop.
15: On chip, computePdropped𝑖𝑗 =P𝑖𝑗Z𝑖𝑗(pointwise multiply).
16: On chip, computedV ̃𝑗 dV ̃𝑗 ̧¹Pdropped𝑖𝑗 º>dO𝑖 2 R𝐵𝑐𝑑.
17: On chip, computedPdropped𝑖𝑗 =dO𝑖V>𝑗 2 R𝐵𝑟𝐵𝑐.
18: On chip, computedP𝑖𝑗=dPdropped𝑖𝑗 Z𝑖𝑗(pointwise multiply).
19: On chip, compute𝐷𝑖=rowsum¹dO𝑖O𝑖º 2R𝐵𝑟.
20: On chip, computedS𝑖𝑗=P𝑖𝑗¹dP𝑖𝑗𝐷𝑖º 2R𝐵𝑟𝐵𝑐.
21: WritedQ𝑖 dQ𝑖 ̧𝜏dS𝑖𝑗K𝑗 2 R𝐵𝑟𝑑to HBM.
22: On chip, computedK ̃𝑗 dK ̃𝑗 ̧𝜏dS>𝑖𝑗Q𝑖 2 R𝐵𝑐𝑑.
23: end for
24: WritedK𝑗 dK ̃𝑗dV𝑗 dV ̃𝑗to HBM.
25:end for
26:ReturndQdKdV.
```
```
We see that similar to the forward pass, the backward pass performs𝑂¹𝑁^2 ºFLOPs and only requires
𝑂¹𝑁ºextra memory beyond inputs, output, output gradient, and input gradients.
We analyze the IO-complexity of the backward pass, similar to the forward pass (Theorem 2).
```
Theorem 5.Let𝑁be the sequence length,𝑑be the head dimension, and𝑀be size of SRAM with𝑑𝑀𝑁𝑑.
Standard attention (Algorithm 0) backward pass requiresΘ¹𝑁𝑑 ̧𝑁^2 ºHBM accesses, whileFlashAttention
backward pass (Algorithm 4) requiresΘ¹𝑁^2 𝑑^2 𝑀^1 ºHBM accesses.

```
The proof is in Appendix C.
```

### B.5 Comparison with Rabe and Staats [66]

We describe here some similarities and differences between ourFlashAttentionalgorithm and the algorithm
of Rabe and Staats [66].
Conceptually, bothFlashAttentionand Rabe and Staats[66]operate on blocks of the attention matrix
using the well-established technique of tiling (or softmax scaling) [ 51 , 60 ]. To reduce the memory footprint,
both methods avoid storing the large attention matrix in the forward pass and recompute it in the backward
pass.
The first major difference is that Rabe and Staats[66]focuses on the reducing the total memory footprint
(maximum amount of GPU memory required) whileFlashAttentionfocuses on reducing memory accesses
(the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the
primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount
of memory required (e.g., if an operation incurs𝐴memory accesses, then its total memory requirement is at
most𝐴). As a result,FlashAttentionis faster than standard attention (2-4) while Rabe and Staats[66]
is around the same speed or slightly slower than standard attention. In terms of total memory required, both
methods offer substantial memory saving.
The second difference between the two methods is the way information is summarized from each block
to pass to the next block. Rabe and Staats[66]summarizes each block with its temporary output along
with the softmax normalization statistics. At the end of the forward pass, the temporary outputs of all the
blocks are combined using the statistics to produce the final output.FlashAttentioninstead incrementally
updates the output (Algorithm 1 line 12) after processing each block, so only one copy of the output is needed
(instead of𝐾copies for𝐾blocks). This means thatFlashAttentionhas smaller total memory requirement
compared to Rabe and Staats [66].
The final major difference is the way the backward pass is computed. Rabe and Staats[66]uses gradient
checkpointing to recompute the attention matrix and the temporary output of each block.FlashAttention
instead simplifies the backward pass analytically (Appendices B.2 and B.4). It only recomputes the attention
matrix and does not recompute the temporary output of each block. This reduces the memory requirement
for the backward pass and yields speedup.

## C Proofs

```
Proof of Theorem 1.We first count the number of FLOPs and extra memory required.
The dominating FLOPs are from matrix multiplication. In the inner loop, (Algorithm 1 line 9), we
computeQ𝑖K>𝑗 2 R𝐵𝑟𝐵𝑐forQ𝑖 2 R𝐵𝑟𝑑andK𝑗 2 R𝐵𝑐𝑑, which takes𝑂¹𝐵𝑟𝐵𝑐𝑑ºFLOPs. We also compute
```
(Algorithm 1 line 12)P ̃𝑖𝑗V𝑗 2 R𝐵𝑟𝑑forP ̃𝑖𝑗 2 R𝐵𝑟𝐵𝑐andV𝑗 2 R𝐵𝑐𝑑, which takes𝑂¹𝐵𝑟𝐵𝑐𝑑ºFLOPs. We

```
execute the inner loops𝑇𝑐𝑇𝑟=
```
```
l
𝑁
𝐵𝑐
```
```
m l
𝑁
𝐵𝑟
```
```
m
times. Therefore the total number of FLOPs is
```
##### 𝑂

##### 

##### 𝑁^2

##### 𝐵𝑐𝐵𝑟

##### 𝐵𝑟𝐵𝑐𝑑

##### 

##### =𝑂¹𝑁^2 𝑑º

```
In terms of extra memory required, we see that we need𝑂¹𝑁ºmemory to store the statistics¹ℓ𝑚º.
We now prove the algorithm’s correctness by induction on𝑗for 0 𝑗𝑇𝑐. LetK:𝑗 2 R𝑗𝐵𝑐𝑑be the
first𝑗𝐵𝑐rows ofK, and similarlyV:𝑗 2 R𝑗𝐵𝑐𝑑the the first𝑗𝐵𝑐rows ofV. LetS::𝑗=QK>:𝑗 2 R𝑁𝑗𝐵𝑐, and
P::𝑗=softmax¹S::𝑗º 2R𝑁𝑗𝐵𝑐(softmax applied row-wise). Let𝑚𝑗ℓ¹𝑗ºO¹𝑗ºbe the values of𝑚ℓOin HBM
after the𝑗-th iteration of the outer loop (Algorithm 1 line 5). (Note that these values of𝑚ℓOare updated
after each iteration of the outer loop.) We want to show that after the𝑗-th iteration of the outer loop, we
have computed in HBM:
```
```
𝑚¹𝑗º=rowmax¹S::𝑗º 2R𝑁 ℓ¹𝑗º=rowsum¹exp¹S::𝑗𝑚¹𝑗ººº 2R𝑁 O¹𝑗º=P::𝑗V:𝑗 2 R𝑁𝑑
```
```
Based on our initialization (Algorithm 1 line 2), this claim is true for𝑗= 0 (i.e., before the any iteration
of the outer loop is executed). Suppose that the claim holds for some𝑗= 0 𝑇𝑐 1. We want to show that
the claim also holds for𝑗 ̧ 1. Indeed, when we update the statistics in the inner loop (Algorithm 1 line 10)
```

```
on the¹𝑗 ̧ 1 º-th iteration of the outer loop, we update𝑚¹𝑗 ̧^1 º=max¹𝑚¹𝑗º𝑚 ̃ºwhere𝑚 ̃ 2 R𝑁is the row-max
ofS:𝑗:𝑗 ̧ 1 , the slice ofSfrom column𝑗𝐵𝑐to column¹𝑗 ̧ 1 º𝐵𝑐 1. This implies that
```
```
𝑚¹𝑗 ̧^1 º=rowmax¹S::𝑗 ̧ 1 º 2R𝑁
```
```
Similarly, we update
ℓ¹𝑗 ̧^1 º=𝑒𝑚
¹𝑗º𝑚¹𝑗 ̧ 1 º
ℓ¹𝑗º ̧𝑒𝑚 ̃𝑚
```
##### ¹𝑗 ̧ 1 º ̃

##### ℓ

```
whereℓ ̃=rowsum¹exp¹S:𝑗:𝑗 ̧ 1 𝑚 ̃ºº 2R𝑁. By the same algebraic manipulation in Section 3.1, we obtain:
```
```
ℓ¹𝑗 ̧^1 º=rowsum¹exp¹S::𝑗 ̧ 1 𝑚¹𝑗 ̧^1 ººº 2R𝑁
```
```
LetV𝑗:𝑗 ̧ 1 be the slice ofVfrom column𝑗𝐵𝑐to column¹𝑗 ̧ 1 º𝐵𝑐 1 , we also update:
```
```
O¹𝑗 ̧^1 º=diag¹ℓ¹𝑗 ̧^1 ºº^1 ¹diag¹ℓ¹𝑗ºº𝑒𝑚
```
```
¹𝑗º𝑚¹𝑗 ̧ 1 º
O¹𝑗º ̧𝑒𝑚 ̃𝑚
```
```
¹𝑗 ̧ 1 º
exp¹S𝑗:𝑗 ̧ 1 𝑚 ̃ºV𝑗:𝑗 ̧ 1 º
=diag¹ℓ¹𝑗 ̧^1 ºº^1 ¹diag¹ℓ¹𝑗ºº𝑒𝑚
```
```
¹𝑗º𝑚¹𝑗 ̧ 1 º
P::𝑗V:𝑗 ̧𝑒𝑚
```
```
¹𝑗 ̧ 1 º
exp¹S𝑗:𝑗 ̧ 1 ºV𝑗:𝑗 ̧ 1 º
=diag¹ℓ¹𝑗 ̧^1 ºº^1 ¹diag¹ℓ¹𝑗ºº𝑒𝑚
¹𝑗º𝑚¹𝑗 ̧ 1 º
diag¹ℓ¹𝑗ººexp¹S::𝑗𝑚¹𝑗ººV:𝑗 ̧𝑒𝑚
¹𝑗 ̧ 1 º
exp¹S𝑗:𝑗 ̧ 1 ºV𝑗:𝑗 ̧ 1 º
```
```
=diag¹ℓ¹𝑗 ̧^1 ºº^1 ¹𝑒𝑚
```
```
¹𝑗 ̧ 1 º
exp¹S::𝑗ºV:𝑗 ̧𝑒𝑚
```
```
¹𝑗 ̧ 1 º
exp¹S𝑗:𝑗 ̧ 1 ºV𝑗:𝑗 ̧ 1 º
=diag¹ℓ¹𝑗 ̧^1 ºº^1 ¹exp¹S::𝑗𝑚¹𝑗 ̧^1 ººV:𝑗 ̧exp¹S𝑗:𝑗 ̧ 1 𝑚¹𝑗 ̧^1 ººV𝑗:𝑗 ̧ 1 º
```
```
=diag¹ℓ¹𝑗 ̧^1 ºº^1
```
##### 

```
exp
```
##### 

##### S::𝑗 S𝑗:𝑗 ̧ 1

##### 

##### 𝑚¹𝑗 ̧^1 º

#####  V

```
:𝑗
V𝑗:𝑗 ̧ 1
```
##### 

```
=softmax¹S:𝑗 ̧ 1 ºV:𝑗 ̧ 1 
```
We then see that the claim is also true for𝑗 ̧ 1. By induction, the claim is true for all𝑗= 0 𝑇𝑐.
When𝑗=𝑇𝑐, we conclude that the final value ofOin HBM issoftmax¹SºV=softmax¹QK>ºV.


```
Proof of Theorem 2.We first analyze the IO complexity of standard attention implementation. The inputs
QKV 2 R𝑁𝑑reside in HBM, and the at the end of the algorithm the outputO 2 R𝑁𝑑is written to HBM.
In the first step of computing the matrix multiplyS=QK>, the inputsQKare read from HBM and the
outputS 2 R𝑁𝑁is written to HBM (Algorithm 0 line 1). This incursΘ¹𝑁𝑑 ̧𝑁^2 ºHBM accesses.
In the second step of computingP=softmax¹Sº, the inputSis read from HBM and the outputPis
written to HBM (Algorithm 0 line 2). This incursΘ¹𝑁^2 ºHBM accesses.
In the last step of computingO=PV, the inputsPVare read from global memory and the outputOis
written to HBM (Algorithm 0 line 3). This incursΘ¹𝑁𝑑 ̧𝑁^2 ºHBM accesses.
Overall, standard attention implementation requiresΘ¹𝑁𝑑 ̧𝑁^2 ºglobal memory accesses.
We now analyze the IO complexity of streaming attention.
Following Algorithm 1, we see that each element ofKandVis loaded from HBM once (Algorithm 1
line 6). We make𝑇𝑐passes overQandO, each pass loading all ofQand all ofOto HBM (Algorithm 1
line 8). Therefore the number of HBM accesses isΘ¹𝑁𝑑 ̧𝑁𝑑𝑇𝑐º= Θ¹𝑁𝑑𝑇𝑐º.
We derive the conditions on the block sizes𝐵𝑐and𝐵𝑟. We need the blocksK𝑗andV𝑗of size𝐵𝑐𝑑to fit
into on-chip memory, which translates to:
```
##### 𝐵𝑐𝑑=𝑂¹𝑀º ,𝐵𝑐=𝑂

##### 

##### 𝑀

##### 𝑑

##### 

##### 

```
Similarly, we need the blocksQ𝑖O𝑖of size𝐵𝑟𝑑to fit into on-chip memory, which translates to:
```
##### 𝐵𝑟𝑑=𝑂¹𝑀º ,𝐵𝑟=𝑂

##### 

##### 𝑀

##### 𝑑

##### 

##### 

```
Finally, we need the blockS𝑖𝑗of size𝐵𝑟𝐵𝑐to fit into on-chip memory, which translates to:
```
```
𝐵𝑟𝐵𝑐=𝑂¹𝑀º
```

We therefore set:

```
𝐵𝑐= Θ
```
##### 

##### 𝑀

##### 𝑑

##### 

#####  𝐵𝑟= Θ

##### 

```
min
```
##### 

##### 𝑀

##### 𝑑

##### 

##### 𝑀

##### 𝐵𝑐

##### 

##### = Θ

##### 

```
min
```
##### 

##### 𝑀

##### 𝑑

#####  𝑑

##### 

##### 

We then have:

```
𝑇𝑐=
```
##### 𝑁

##### 𝐵𝑐

##### = Θ

##### 

##### 𝑁𝑑

##### 𝑀

##### 

##### 

```
As a result, the number of HBM accesses is:
```
##### Θ¹𝑁𝑑𝑇𝑐º= Θ

##### 

##### 𝑁^2 𝑑^2

##### 𝑀

##### 

##### 

##### 

```
Proof of Proposition 3.For contradiction, suppose that there exists an algorithm that computes exact
attention where the number for HBM access for all𝑀2 »𝑑 𝑁𝑑¼is
```
##### 𝑜

##### 

##### 𝑁^2 𝑑^2

##### 𝑀

##### 

##### 

```
In the regime of𝑀= Θ¹𝑁𝑑º, this results in the number of HBM accesses:
```
##### 𝑜

##### 

##### 𝑁^2 𝑑^2

##### 𝑁𝑑

##### 

##### =𝑜¹𝑁𝑑º

```
However, the input to attention (matricesQKV) and the outputOhave size𝑁𝑑and they start out being
in HBM, so if the algorithm computes exact attention it must incur at leastΩ¹𝑁𝑑ºHBM accesses. This is a
contradiction. 
```
```
Proof of Theorem 5.The IO complexity of the attention backward is very similar to the IO complexity of
the attention forward (Theorem 2). Here we provide a sketch of the proof.
We first analyze the IO complexity of standard attention backward pass. The inputsQKVdO 2 R𝑁𝑑
reside in HBM, and the at the end of the algorithm the outputsdQdKdV 2 R𝑁𝑑are written to HBM.
At each step of the standard attention backward pass, one needs to load inputs of size𝑁𝑑or𝑁^2 from
HBM, and needs to write the outputs of size𝑁^2 or𝑁𝑑to HBM. This incursΘ¹𝑁𝑑 ̧𝑁^2 ºHBM accesses.
We now analyze the IO complexity ofFlashAttentionbackward pass.
Similar to Theorem 2, we see that each element ofKandVis loaded from HBM once. Each element of
dKanddVis only written to HBM once. We make𝑇𝑐passes overQOdO, each pass loading all ofQOdO
to HBM. We also make𝑇𝑐passes overdQ, each pass reading/writing all ofdQfrom/to HBM. Therefore the
number of HBM accesses isΘ¹𝑁𝑑 ̧𝑁𝑑𝑇𝑐º= Θ¹𝑁𝑑𝑇𝑐º.
As in the proof of Theorem 2, the constraints on the block sizes are that:
```
##### 𝐵𝑐= Θ

##### 

##### 𝑀

##### 𝑑

##### 

#####  𝐵𝑟= Θ

##### 

```
min
```
##### 

##### 𝑀

##### 𝑑

#####  𝑑

##### 

##### 

We then have:

```
𝑇𝑐=
```
##### 𝑁

##### 𝐵𝑐

##### = Θ

##### 

##### 𝑁𝑑

##### 𝑀

##### 

##### 

```
As a result, the number of HBM accesses is:
```
##### Θ¹𝑁𝑑𝑇𝑐º= Θ

##### 

##### 𝑁^2 𝑑^2

##### 𝑀

##### 

##### 

##### 


```
Algorithm 5Block-SparseFlashAttentionForward Pass
Require:MatricesQKV 2 R𝑁𝑑in HBM, on-chip SRAM of size𝑀, softmax scaling constant𝜏 2 R,
masking functionmask, dropout probability𝑝drop, block sizes𝐵𝑐=
```
##### 𝑀

```
4 𝑑
```
##### 

```
 𝐵𝑟 =min
```
##### 𝑀

```
4 𝑑
```
##### 

#####  𝑑

##### 

```
, block
sparsity mask𝑀2 f 0  1 g𝑁𝐵𝑟𝑁𝐵𝑐..
1:Initialize the pseudo-random number generator stateRand save to HBM.
2:InitializeO=¹ 0 º𝑁𝑑 2 R𝑁𝑑ℓ=¹ 0 º𝑁 2 R𝑁𝑚=¹1º𝑁 2 R𝑁in HBM.
3:DivideQinto𝑇𝑟=
```
```
l
𝑁
𝐵𝑟
```
```
m
blocksQ 1 Q𝑇𝑟of size𝐵𝑟𝑑each, and divideKVin to𝑇𝑐=
```
```
l
𝑁
𝐵𝑐
```
```
m
blocks
K 1 K𝑇𝑐andV 1 V𝑇𝑐, of size𝐵𝑐𝑑each.
4:DivideOinto𝑇𝑟blocksO𝑖O𝑇𝑟of size𝐵𝑟𝑑each, divideℓinto𝑇𝑟blocksℓ𝑖ℓ𝑇𝑟of size𝐵𝑟each,
divide𝑚into𝑇𝑟blocks𝑚 1 𝑚𝑇𝑟of size𝐵𝑟each.
5:for 1 𝑗𝑇𝑐do
6: LoadK𝑗V𝑗from HBM to on-chip SRAM.
7: for 1 𝑖𝑇𝑟do
8: if𝑀𝑖𝑗≠ 0 then
9: LoadQ𝑖O𝑖ℓ𝑖𝑚𝑖from HBM to on-chip SRAM.
10: On chip, computeS𝑖𝑗=𝜏Q𝑖K𝑇𝑗 2 R𝐵𝑟𝐵𝑐.
11: On chip, computeSmasked𝑖𝑗 =mask¹S𝑖𝑗º.
12: On chip, compute𝑚 ̃𝑖𝑗=rowmax¹Smasked𝑖𝑗 º 2R𝐵𝑟,P ̃𝑖𝑗=exp¹Smasked𝑖𝑗 𝑚 ̃𝑖𝑗º 2R𝐵𝑟𝐵𝑐(pointwise),
ℓ ̃𝑖𝑗=rowsum¹P ̃𝑖𝑗º 2R𝐵𝑟.
13: On chip, compute𝑚new𝑖 =max¹𝑚𝑖𝑚 ̃𝑖𝑗º 2R𝐵𝑟,ℓnew𝑖 =𝑒𝑚𝑖𝑚
new𝑖
ℓ𝑖 ̧𝑒𝑚 ̃𝑖𝑗𝑚
new𝑖 ̃
ℓ𝑖𝑗 2 R𝐵𝑟.
14: On chip, computeP ̃dropped𝑖𝑗 =dropout¹P ̃𝑖𝑗 𝑝dropº.
15: WriteO𝑖 diag¹ℓ𝑖newº^1 ¹diag¹ℓ𝑖º𝑒𝑚𝑖𝑚
new𝑖
O𝑖 ̧𝑒𝑚 ̃𝑖𝑗𝑚
new𝑖 ̃
Pdropped𝑖𝑗 V𝑗ºto HBM.
16: Writeℓ𝑖 ℓ𝑖new,𝑚𝑖 𝑚new𝑖 to HBM.
17: end if
18: end for
19:end for
20:ReturnOℓ𝑚R.
```
## D Extension Details

### D.1 Block-sparseFlashAttention

We describe the full block-sparseFlashAttentionalgorithm in Algorithm 5. The algorithm is identical
to Algorithm 2, except that we skip zero blocks.
We prove the IO-complexity of block-sparseFlashAttention.

```
Proof of Proposition 4.The proof is very similar to the proof of Theorem 2. For the block-sparse case, notice
that we only need to load blocks corresponding to nonzero blocks. As a result, the number of HBM accesses
are scaled by𝑠, the fraction of nonzero blocks in the block-sparsity mask. However, for small values of𝑠, we
would still need to write the resultO 2 R𝑁𝑑. Therefore the number of HBM accesses is
```
##### Θ

##### 

##### 𝑁𝑑 ̧

##### 𝑁^2 𝑑^2

##### 𝑀

##### 𝑠

##### 

##### 

##### 

### D.2 Potential Extensions

We discuss here a few potential extensions of the IO-aware approach to speed up deep learning training.
Multi-GPU Attention.Large language models are trained on hundreds or thousands of GPUs, and
one typically splits the attention computation between 4-8 GPUs on the same node [ 77 ]. This introduces
another level of memory hierarchy: beside GPU SRAM and GPU HBM, we also have the HBM of other


```
GPUs. For very long sequences, the different GPUs on the same node can cooperate to compute attention by
taking into account the asymmetry of different levels of memory hierarchy.
Sparse MLP layers.Typical dense MLP layers are compute-bound and not memory-bound. To improve
their efficiency, MLP layers with sparse weight matrices can be used [ 17 ]. However, many sparse MLP layers
are instead memory-bound, and their speedup is often not proportional to the sparsity. We believe that an
IO-aware implementation can alleviate this issue and realize the benefits of sparsity. We are excited about
future work in this direction, to reduce the computational requirement of large models and improve their
wall-block runtime.
Kernel machine learning. Our approach inFlashAttentionrelies on the fact that the𝑁𝑁
attention matrix is a function of a low-rank matrixQK>(of rank𝑑𝑁). As a result, we can repeatedly
load the inputsQKand recompute the block of the attention matrix that we need, significantly reducing
HBM access. As similar scenario happens in kernel machine learning: each element𝐾𝑖𝑗of the𝑁𝑁kernel
matrixKis a function of two vectors of size𝑑𝑁, as it measures the similarity between two datapoints𝑥𝑖
and𝑥𝑗. The KeOps library [ 8 , 26 ] is a successful example of how reducing memory reads/writes can speed up
kernel operations. We hope that this will motivate kernel methods that focus more on reducing IOs instead
of just FLOPs.
```
## E Full Experimental Results

### E.1 BERT

We train BERT-large following the training procedure and hyperparameters of the reference MLPerf 1.1
implementation. In particular, we use the LAMB optimizer with learning rate 3.75e-3, with batch size 448,
trained for at most 7100 steps. The training is stopped once the validation accuracy (for masked language
modeling) reaches the target 72.0%, and the wall-clock run-time is measured. We train with FP16 precision
using Apex AMP (with O2 optimization level).
We compare our results with the reported training speed from Nvidia that was submitted to MLPerf 1.1
(Table 1).
We use the same train / validation data split provided by MLPerf 1.1 reference implementation. In
particular, we evaluate on the same 10000 validation examples as the baseline from Nvidia.
We train the model on 8A100-80GB GPUs. Each training run takes between 16 and 19 minutes, and we
average the results of 10 runs.

### E.2 GPT-2

We use the standard implementations of GPT-2 [ 67 ] from Huggingfacetransformerslibrary and from
Nvidia’s Megatron-LM repo. We follow the training recipe of the Megatron-LM repo.
We use an effective batch size of 512, and use gradient accumulation to fit into available GPU memory.
We use the AdamW optimizer, with learning rate 6e-4 for GPT-2 small and 1.5e-4 for GPT-2 medium, and
weight decay of 0.1. All models are trained with the same hyperparameters for 400K steps. We run all
implementations with mixed-precision training (PyTorch AMP).
We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select 0.5% of the dataset
as the validation set, with the rest being used as training set. This random selection of validation set is done
once, and all models are evaluated on the same validation set.
We train the model on 8A100-40GB GPUs, and we measure the wall-clock training time. Training
GPT-2 small takes between 2.7-9.5 days, and training GPT-2 medium takes between 6.9-21.0 days (Table 2).
In Fig. 4, we plot of the validation perplexity throughout training of GPT-2 small/medium, using either
HuggingFace implementation or ourFlashAttentionimplementation. We see thatFlashAttentionbe-
haves the same as the baseline implementation and the validation perplexity curves of the two implementations
almost lie on top of each other.

```
Long Document Classification. For MIMIC-III and ECtHR, we follow the hyperparameters of Dai et al.
[13].
```

```
100k 200k 300k
```
#### Training steps

```
10
```
```
15
```
```
20
```
```
25
```
```
30
```
#### Val perplexity

#### GPT-2-small HuggingFace

#### GPT-2-small FlashAttention

#### GPT-2-medium HuggingFace

#### GPT-2-medium FlashAttention

```
Figure 4: Validation perplexity of GPT-2 small/medium using two implementations. We confirm that
FlashAttentionyields the same validation curves as the baseline implementation from HuggingFace.
```
### E.3 LRA details

We follow the hyperparameters from the Long-range arena paper [ 80 ], the Long-range arena repo (https:
//github.com/google-research/long-range-arena), and the Nyströmformer reproduction [ 90 ]. To be
generous to the baseline methods, if we are unable to reproduce the performance of any baseline for any of
the five tasks, we report the better performance from Tay et al.[80]or Xiong et al.[90]for that baseline on
that task.
After hyperparameter tuning, almost all of the attention methods achieve similar accuracy on all of the
five LRA tasks.
We run all methods with mixed-precision training, except for Performer (not stable with mixed precision)
and Local Attention (implementation does not support FP16).
To calculate the overall wallclock-time speedup, we take the geometric mean of the wallclock-time speedup
of each of the five tasks.

```
Path-X For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments
from the long-range arena paper[ 80 ]. For both, we first pretrain a model on Path-64. We take the checkpoint
after 200 epochs, upsample its positional embedding (we duplicate the positional embeddings gridwise in
space), and fine-tune it on the downstream task for 200 epochs with one epoch of linear warmup, and cosine
decay of the learning rate. For Path-X, we take the best performing checkpoint (according to val accuracy),
and additionally fine-tune it for 200 epochs with the same warmup and learning rate (this adds roughly 4
points of accuracy toFlashAttentionfor Path-X, but the model starts overfitting afterwards).
```
### E.4 Comparison with Apex FMHA

We compare our method/implementation with Apex FMHA (https://github.com/NVIDIA/apex/tree/
master/apex/contrib/csrc/fmha).
When we started this project, Apex FMHA was the fastest implementation of attention (that we knew
of), tailored for short sequences of length at most 512. In fact, almost all MLPerf submissions for BERT
training benchmark running on Nvidia GPUs use FMHA for their model code, as of MLPerf 1.1 [ 58 ]. Since


Table 7: Runtime (ms) ofFlashAttentioncompared to FMHA by sequence length, with masking and dropout,
measured on an A100-SXM4-40GB GPU. Batch size 64, 16 heads, head dimension 64 (i.e., BERT-large size).

```
Attention Method 128 256 512
Apex FMHA forward 0.10 0.29 1.14
FlashAttentionforward 0.08 0.22 0.81
Apex FMHA backward 0.17 0.52 1.81
FlashAttentionbackward 0.20 0.53 2.00
Apex FMHA forward + backward 0.27 0.81 2.95
FlashAttentionforward + backward 0.28 0.75 2.81
```
FMHA targets BERT models, it only supports head dimension 64, and only runs on A100 GPUs. FMHA
fuses the attention computationdropout¹softmax¹mask¹QK>ºººVinto one CUDA kernel. In the forward
pass, it stores the attention matrixsoftmax¹mask¹QK𝑇ººto HBM to be used in gradient computation. As a
result, it does not offer substantial memory saving (though for shorter sequences memory footprint is often
not a primary concern).
We use FMHA code as a starting point, and apply two well-established techniques (tiling and recomputa-
tion) to deal with long sequences and to save memory as mentioned in Section 3. As a result, we can support
much longer sequences (e.g., up to length 64K). We also support more head dimensions (16, 32, 64, 128) and
broader GPU types (all Turing and Ampere GPUs at the time of writing).
In Table 7, we compare the performance ofFlashAttentionand Apex FMHA for short sequences (as
FMHA only supports sequence length at most 512). GenerallyFlashAttentionis slightly faster than
FMHA in the forward pass and slightly slower than FMHA in the backward pass. This is because we do not
store the attention matrix in the forward pass and recompute it in the backward pass. Compared to FMHA,
the overall runtime ofFlashAttentionis about 4% slower for sequence length 128, 8% faster for sequence
length 256, and 5% faster for sequence length 512.

### E.5 Speedup On Different Hardware and Configurations

Speedup varies between different types of GPU types and generations depending on HBM bandwidth and
SRAM size. In this section, we profileFlashAttentionspeedup on different GPUs and configurations.

```
Figure 5: Speedup over standard PyTorch attention at different sequence lengths, on A100.
```
A100 Figure 5 shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention
heads, across different sequence lengths. We generally see 2-4speedup, and we see more speedup when
using dropout and masking due to kernel fusion.


Figure 6: Speedup over standard PyTorch attention at different sequence lengths, on A100, with head
dimension 128.

A100, Head Dimension 128 Speedup also changes when we increase the head dimension. Each block
requires more memory, so we need to use smaller block sizes to fit into SRAM. Figure 6 shows speedup with
head dimension 128 on an A100 (batch size 16, 12 heads). We see less speedup overall—but we can still see
significant speedup (up to 3) with a causal mask, where half the blocks are masked out.

```
Figure 7: Speedup over standard PyTorch attention at different sequence lengths, on RTX 3090.
```
RTX 3090 Figure 7 shows speedup on an RTX 3090 GPU. Here, we use batch size 12 with 12 attention
heads. We observe slightly higher speedups on the RTX 3090 (between 2.5-4.5), since the memory bandwidth
on an RTX 3090 is lower than on an A100 (roughly 900 GB/s vs. 1.5 TB/s).

T4 Figure 8 shows speedup on a T4 GPU. T4 SRAM is smaller than A100, so we need to make the block
sizes smaller inFlashAttention. As a result, we observe less speedup on T4, which matches the IO
complexity analysis in Section 3.2. T4 GPUs are commonly used for inference, so we also report speedup on
the forward pass only.


```
Figure 8: Speedup over standard PyTorch attention at different sequence lengths, on T4.Top:Combined
forward pass + backward pass.Bottom:Forward pass only.
```
### E.6 Full Benchmarking Results

We report the full benchmarking results and experimental details on A100.

```
Baselines We compare against reference implementations for exact attention from PyTorch/HuggingFace
and Megatron, approximate attention, and sparse attention. For approximate attention, we compare against
reference implementations of Reformer [ 51 ], Local Attention [ 68 ], Linformer Attention [ 84 ], Smyrf [ 19 ], and
LongShortFormer (LSFormer) [ 94 ]. For sparse attention, we compare against reference implementations of
Block-Sparse Attention form OpenAI [ 11 ], Longformer[ 3 ], and BigBird Attention [ 92 ]. For the approximate
and sparse attention, we use a compression ratio of 1/8, or a compressed sequence length of 256, whichever is
smaller.
```
```
Setup We measure runtime and memory usage of the attention computation with 8 heads of dimension 64,
and batch size 16 on a machine with one A100 GPU with 40 GB of GPU HBM. We vary sequence length
in our experiments. We compute attention on random vectors forQ,K, andV(we do not measure the
projection from the hidden layer). For dropout, we use dropout 0.1; for masking, we use a padding mask
with uniformly-random mask lengths between the total sequence length and the total sequence length minus
```
20. To measure runtime, we take the average of 100 measurements of the attention call. We only measure
memory footprint once, since it does not vary between runs.


```
Table 8: Pointers to results tables.
```
```
Dropout Masking Pass Table
Yes Yes Forward Table 9
Yes Yes Backward Table 10
Yes Yes Combined Table 11
No Yes Forward Table 12
No Yes Backward Table 13
No Yes Combined Table 14
Yes No Forward Table 15
Yes No Backward Table 16
Yes No Combined Table 17
No No Forward Table 18
No No Backward Table 19
No No Combined Table 20
No No Memory Usage (Combined) Table 21
```
```
Table 9: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout and masking. Best inbold, second best underlined.
```
```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.36 0.34 0.78 2.54 9.33 36.33 - - - -
Megatron 0.40 0.40 1.10 3.65 16.19 - - - - -
Reformer 2.03 3.15 5.67 11.02 22.59 46.14 97.38 212.13 - -
Local Attention 0.83 0.86 1.01 2.20 7.13 14.32 28.60 57.79 117.67 -
Linformer 0.67 0.52 0.69 0.71 1.65 3.18 6.15 12.16 24.17 52.39
Smyrf 2.27 2.34 3.91 7.44 14.71 29.22 58.27 116.41 - -
LSformer 1.18 1.27 1.34 3.38 11.40 22.55 44.95 89.76 179.66 -
Block Sparse 1.12 1.11 2.13 2.77 6.95 20.91 - - - -
Longformer 1.22 1.14 1.08 1.95 5.72 12.98 - - - -
BigBird 1.13 1.12 1.12 1.77 6.03 13.68 - - - -
FlashAttention 0.04 0.06 0.21 0.82 2.85 10.41 41.74 167.19 670.76 2682.35
Block-SparseFlashAttention 0.06 0.06 0.06 0.12 0.44 0.86 1.70 3.29 6.55 13.34
```
We report timing results on the forward pass, backward pass, and combined forward + backward pass.
We measure each method with and without dropout, masking, or both—except for Block Sparse, Longformer,
and BigBird. These methods did not successfully run the backward pass with masking due to a bug in
external libraries, so we measured them without masking to be generous. We use FP16 for all measurements,
except for Local Attention, whose implementation only supports FP32.
For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the
following exceptions: The Megatron implementation does not support sequence lengths longer than 2048.
Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not
support sequence lengths longer than 8092.
We measure memory usage on the combined forward + backward pass, without dropout or masking.

```
Results Table 8 summarizes all the experimental configurations and contains pointers to the results tables.
```

Table 10: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout and masking. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.37 0.49 1.66 5.81 22.32 87.67 - - - -
Megatron 0.35 0.32 0.77 2.42 8.43 - - - - -
Reformer 2.37 4.59 8.91 17.68 35.13 70.05 140.01 - - -
Local Attention 0.55 0.62 1.49 4.03 13.78 27.61 55.20 110.27 221.40 -
Linformer 0.89 0.80 0.81 0.93 2.48 4.75 9.29 18.27 36.53 -
Smyrf 1.41 2.83 5.43 10.72 21.25 42.31 84.48 168.95 - -
LSformer 1.75 1.76 3.01 7.50 20.07 39.08 76.39 150.82 - -
Block Sparse 1.29 1.28 2.18 3.04 7.27 21.16 - - - -
Longformer 1.27 1.31 1.29 2.04 5.24 10.74 25.95 - - -
BigBird 1.33 1.28 1.32 1.81 5.55 11.44 27.45 - - -
FlashAttention 0.30 0.26 0.68 2.02 6.84 26.89 105.70 418.96 1666.89 6660.44
Block-SparseFlashAttention 0.30 0.27 0.29 0.59 1.50 2.94 5.82 11.85 23.98 47.61
```
Table 11: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length,with dropout and masking. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.84 0.86 2.35 8.29 31.75 124.19 - - - -
Megatron 0.87 0.89 1.33 4.21 16.50 - - - - -
Reformer 4.30 7.76 14.60 28.74 57.79 116.34 237.57 - - -
Local Attention 1.40 1.60 2.06 6.06 20.94 42.01 84.08 168.48 339.45 -
Linformer 1.57 1.49 1.55 1.60 4.19 8.04 15.71 30.92 61.47 -
Smyrf 3.41 5.08 9.35 18.18 36.03 71.68 143.04 285.87 - -
LSformer 3.08 3.10 4.26 10.90 31.59 61.72 121.51 241.18 - -
Block Sparse 2.54 2.52 3.71 5.44 13.29 39.19 - - - -
Longformer 2.47 2.49 2.51 3.10 10.39 22.49 60.44 - - -
BigBird 2.51 2.49 2.52 3.40 10.97 23.89 63.28 - - -
FlashAttention 0.43 0.41 0.95 2.55 9.56 37.49 147.75 586.61 2339.11 9341.30
Block-SparseFlashAttention 0.44 0.44 0.45 0.89 1.95 4.12 7.64 16.60 32.73 64.11
```
Table 12: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with masking. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.30 0.30 0.63 1.93 7.08 27.45 112.90 - - -
Megatron 0.45 0.41 0.43 1.52 5.80 - - - - -
Reformer 1.87 3.00 5.37 10.43 21.40 43.83 92.80 203.24 - -
Local Attention 0.70 0.81 1.02 2.09 6.64 13.34 26.77 54.02 110.11 -
Linformer 0.63 0.50 0.67 0.65 1.36 2.60 5.04 9.92 19.69 43.47
Smyrf 2.38 2.32 3.76 7.16 14.14 28.09 55.98 111.73 - -
LSformer 1.22 1.29 1.44 3.28 10.99 21.72 43.29 86.32 172.76 -
Block Sparse 0.96 1.04 1.66 2.16 5.41 16.15 - - - -
Longformer 0.99 0.98 0.99 1.56 4.79 11.07 32.98 - - -
BigBird 0.96 1.02 1.02 1.48 5.05 11.59 34.16 - - -
FlashAttention 0.03 0.04 0.17 0.68 2.28 8.40 33.55 134.14 537.50 2150.88
Block-SparseFlashAttention 0.05 0.04 0.05 0.11 0.35 0.68 1.33 2.54 5.34 10.73
```
Table 13: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with masking. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.44 0.46 1.53 5.33 20.34 79.87 - - - -
Megatron 0.29 0.31 0.65 1.95 6.49 - - - - -
Reformer 2.31 4.47 8.68 17.20 34.14 68.09 136.02 - - -
Local Attention 0.51 0.62 1.30 3.81 13.33 26.72 53.41 106.82 214.15 -
Linformer 0.76 0.81 0.94 0.87 2.24 4.25 8.35 16.38 32.67 72.11
Smyrf 1.34 2.77 5.30 10.46 20.73 41.27 82.41 164.86 - -
LSformer 1.66 1.61 3.09 7.42 19.68 38.35 74.92 147.86 - -
Block Sparse 1.24 1.25 2.04 2.91 6.78 19.67 - - - -
Longformer 1.27 1.23 1.24 1.85 4.99 10.21 24.89 - - -
BigBird 1.43 1.50 1.44 1.69 5.25 10.86 26.26 - - -
FlashAttention 0.21 0.22 0.62 1.84 5.77 22.25 86.21 338.91 1343.91 5361.09
Block-SparseFlashAttention 0.22 0.22 0.26 0.57 1.55 3.13 5.98 12.21 23.49 47.85
```

Table 14: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length,with masking. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.80 0.81 2.08 7.23 27.51 107.58 - - - -
Megatron 0.81 0.83 1.09 3.36 12.39 - - - - -
Reformer 4.16 7.46 14.06 27.68 55.66 112.15 229.37 - - -
Local Attention 1.39 1.68 2.08 5.83 20.04 40.16 80.44 161.35 325.11 -
Linformer 1.51 1.42 1.56 1.67 3.67 6.99 13.63 26.77 53.36 117.56
Smyrf 3.38 4.93 9.07 17.66 34.94 69.55 138.72 277.41 - -
LSformer 3.08 3.10 4.26 10.90 31.59 61.72 121.51 241.18 - -
Block Sparse 2.39 2.40 3.31 5.02 12.25 35.94 - - - -
Longformer 2.36 2.34 2.38 2.94 9.83 21.35 58.12 - - -
BigBird 2.35 2.35 2.37 3.25 10.36 22.57 60.63 - - -
FlashAttention 0.32 0.30 0.83 2.37 7.95 30.77 119.98 473.65 1883.43 7513.01
Block-SparseFlashAttention 0.34 0.34 0.36 0.69 1.85 3.89 7.16 14.85 30.46 60.03
```
Table 15: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.26 0.24 0.57 1.80 6.56 25.34 - - - -
Megatron 0.27 0.27 0.56 1.88 6.56 - - - - -
Reformer 1.83 2.96 5.31 10.33 21.19 43.42 91.96 201.34 - -
Local Attention 0.51 0.60 0.78 2.01 6.23 12.52 25.07 50.50 102.18 -
Linformer 0.47 0.37 0.49 0.52 1.37 2.65 5.12 10.13 20.25 44.16
Smyrf 2.12 2.01 3.15 5.97 11.83 23.36 46.48 92.72 - -
LSformer 1.28 1.33 1.51 3.39 11.40 22.54 44.96 89.85 179.73 -
Block Sparse 1.03 1.00 1.72 2.39 5.96 17.88 - - - -
Longformer 1.02 1.03 1.03 1.73 5.10 11.63 34.22 - - -
BigBird 0.99 1.03 1.01 1.58 5.36 12.27 35.56 - - -
FlashAttention 0.10 0.10 0.22 0.83 2.81 10.38 41.63 167.01 668.74 2678.11
Block-SparseFlashAttention 0.54 0.51 0.68 0.61 0.67 1.10 1.89 3.71 7.18 14.41
```
Table 16: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length,
with dropout. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.44 0.35 0.90 2.94 10.77 41.67 - - - -
Megatron 0.28 0.33 0.92 2.94 10.80 - - - - -
Reformer 2.24 4.34 8.39 16.62 33.02 65.77 131.52 - - -
Local Attention 0.51 0.58 1.41 3.71 12.96 25.98 51.94 103.72 207.78 -
Linformer 0.84 0.74 0.79 0.85 2.28 4.37 8.66 17.02 33.78 -
Smyrf 1.27 2.56 4.90 9.66 19.16 38.13 76.17 152.39 - -
LSformer 1.67 1.77 3.03 7.52 20.10 39.13 76.35 150.83 - -
Block Sparse 1.27 1.36 2.15 3.04 7.27 21.18 - - - -
Longformer 1.28 1.34 1.38 1.98 5.24 10.74 25.95 - - -
BigBird 1.48 1.47 1.50 1.81 5.57 11.38 27.43 - - -
FlashAttention 0.15 0.18 0.58 1.86 6.50 26.21 104.27 416.10 1661.92 6643.01
Block-SparseFlashAttention 0.17 0.17 0.17 0.40 1.10 2.04 4.43 9.33 18.28 37.31
```
Table 17: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length,with dropout. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.66 0.67 1.43 4.82 17.47 67.29 - - - -
Megatron 0.88 0.90 1.49 4.73 17.41 - - - - -
Reformer 4.06 7.28 13.68 26.98 54.27 109.39 223.80 - - -
Local Attention 1.09 1.40 1.99 5.61 19.23 38.62 77.30 154.63 311.12 -
Linformer 1.31 1.21 1.30 1.39 3.73 7.15 14.05 27.69 55.00 -
Smyrf 3.00 4.37 8.05 15.66 31.04 61.64 123.04 245.65 - -
LSformer 3.07 3.17 4.31 10.89 31.54 61.78 121.56 240.94 - -
Block Sparse 2.54 2.52 3.71 5.44 13.29 39.19 - - - -
Longformer 2.47 2.49 2.51 3.10 10.39 22.49 60.44 - - -
BigBird 2.51 2.49 2.52 3.40 10.97 23.89 63.28 - - -
FlashAttention 0.35 0.36 0.80 2.52 9.16 36.70 146.13 583.45 2332.01 9323.63
Block-SparseFlashAttention 0.91 0.83 0.94 0.92 1.83 3.50 7.02 13.56 26.71 53.92
```

Table 18: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length.
Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.21 0.22 0.43 1.27 4.32 16.47 67.77 - - -
Megatron 0.24 0.26 0.42 1.33 4.28 - - - - -
Reformer 1.77 2.82 5.01 9.74 20.03 41.11 87.39 192.40 - -
Local Attention 0.48 0.57 0.80 1.90 5.76 11.56 23.13 46.65 94.74 -
Linformer 0.46 0.36 0.45 0.50 1.09 2.09 4.01 7.90 15.70 35.40
Smyrf 1.94 1.96 3.01 5.69 11.26 22.23 44.21 88.22 - -
LSformer 1.21 1.34 1.34 3.31 11.01 21.71 43.27 86.32 172.85 -
Block Sparse 0.96 1.04 1.66 2.16 5.41 16.15 - - - -
Longformer 0.99 0.98 0.99 1.56 4.79 11.07 32.98 - - -
BigBird 0.96 1.02 1.02 1.48 5.05 11.59 34.16 - - -
FlashAttention 0.08 0.09 0.18 0.68 2.40 8.42 33.54 134.03 535.95 2147.05
Block-SparseFlashAttention 0.56 0.52 0.63 0.65 0.61 0.96 1.69 3.02 5.69 11.77
```
Table 19: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length.
Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.26 0.29 0.78 2.44 8.82 33.87 - - - -
Megatron 0.29 0.30 0.80 2.59 8.86 - - - - -
Reformer 2.18 4.21 8.14 16.12 32.02 63.84 127.60 - - -
Local Attention 0.51 0.64 1.28 3.60 12.52 25.08 50.22 100.23 200.66 -
Linformer 0.69 0.76 0.69 0.80 2.04 3.88 7.67 15.04 30.11 63.15
Smyrf 1.24 2.49 4.77 9.42 18.65 37.12 74.15 148.35 - -
LSformer 1.68 1.61 3.02 7.40 19.72 38.27 74.89 147.99 - -
Block Sparse 1.24 1.25 2.04 2.91 6.78 19.67 - - - -
Longformer 1.27 1.23 1.24 1.85 4.99 10.21 24.89 - - -
BigBird 1.43 1.50 1.44 1.69 5.25 10.86 26.26 - - -
FlashAttention 0.11 0.16 0.52 1.62 5.45 21.57 84.75 336.00 1338.56 5343.19
Block-SparseFlashAttention 0.11 0.12 0.16 0.38 1.20 2.34 4.69 9.10 18.74 37.04
```
Table 20: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by
sequence length. Best inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 0.67 0.70 1.18 3.67 13.22 50.44 - - - -
Megatron 0.74 0.65 1.23 3.80 13.21 - - - - -
Reformer 3.93 7.01 13.15 25.89 52.09 105.00 215.13 - - -
Local Attention 1.09 1.27 1.99 5.38 18.32 36.77 73.67 147.29 296.35 -
Linformer 1.31 1.25 1.30 1.29 3.20 6.10 11.93 23.39 46.72 100.52
Smyrf 2.98 4.23 7.78 15.12 29.96 59.45 118.60 237.02 - -
LSformer 3.03 3.05 4.26 10.70 30.77 60.15 118.33 234.94 - -
Block Sparse 2.39 2.40 3.31 5.02 12.25 35.94 - - - -
Longformer 2.36 2.34 2.38 2.94 9.83 21.35 58.12 - - -
BigBird 2.35 2.35 2.37 3.25 10.36 22.57 60.63 - - -
FlashAttention 0.31 0.31 0.73 2.29 7.64 30.09 118.50 470.51 1876.08 7492.85
Block-SparseFlashAttention 0.74 0.77 0.82 0.88 1.71 3.21 6.56 12.60 24.93 50.39
```
Table 21: Memory usage (MB) of various exact/approximate/sparse attention mechanisms by sequence length. Best
inbold, second best underlined.

```
Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536
PyTorch Attention 36 104 336 1184 4416 17024 - - - -
Megatron 36 104 336 1184 4416 - - - - -
Reformer 377 754 1508 3016 6033 12067 24134 - - -
Local Attention 53 110 232 592 1696 3392 6784 13568 27136 -
Linformer 25 52 114 287 832 1652 3292 6572 13132 26252
Smyrf 217 434 868 1737 3474 6947 13894 27788 - -
LSformer 72 152 333 796 2540 5068 10125 20240 - -
Block Sparse 33 82 228 408 910 2401 - - - -
Longformer 30 61 124 277 681 1370 2748 - - -
BigBird 33 66 131 294 708 1431 2872 - - -
FlashAttention 22 44 104 209 418 836 1672 3344 6688 13376
Block-SparseFlashAttention 22 44 104 209 418 836 1672 3344 6690 13384
```

