When implementing multiheaded attention there are 2 simple ways to do it. Either you can concat the result of multiple heads or you could try doing it all in one pass. 
Here is a simple test to figure out the performance of each multiheaded self attention. This test is performed with 3 heads, 1024 input channels, 512 bottleneck channels and a spatial resolution of 16 by 16. Each model is run with 16 samples per batch and 10 times.

### Concatenating tensors
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
                  SLOW PASS        38.39%      26.295ms        99.74%      68.320ms       6.832ms     397.384us         0.46%      86.641ms       8.664ms         200 b      -2.46 Kb       1.41 Gb    -240.00 Mb            10  
               aten::conv2d         0.70%     476.547us        21.61%      14.806ms     164.506us     209.826us         0.24%      72.142ms     801.575us           0 b           0 b     720.00 Mb           0 b            90  
          aten::convolution         0.72%     490.877us        20.92%      14.329ms     159.211us     206.288us         0.24%      71.932ms     799.244us           0 b           0 b     720.00 Mb           0 b            90  
         aten::_convolution         1.32%     902.351us        20.20%      13.838ms     153.757us     301.002us         0.35%      71.726ms     796.952us           0 b           0 b     720.00 Mb           0 b            90  
    aten::cudnn_convolution        16.30%      11.162ms        18.55%      12.707ms     141.188us      71.114ms        82.00%      71.325ms     792.504us           0 b           0 b     720.00 Mb    -360.00 Kb            90  
               aten::einsum        22.72%      15.563ms        34.23%      23.447ms     390.780us     610.232us         0.70%       9.148ms     152.474us           0 b           0 b     360.00 Mb           0 b            60  
                  aten::bmm         3.04%       2.083ms         3.55%       2.429ms      40.488us       7.404ms         8.54%       7.404ms     123.392us           0 b           0 b     360.00 Mb           0 b            60  
                aten::stack         0.25%     171.692us         1.50%       1.028ms     102.751us      46.045us         0.05%       2.209ms     220.875us           0 b           0 b     240.00 Mb           0 b            10  
                  aten::cat         0.08%      51.787us         1.17%     801.687us      80.169us      20.995us         0.02%       2.163ms     216.270us           0 b           0 b     240.00 Mb           0 b            10  
                 aten::_cat         0.63%     430.194us         1.09%     749.900us      74.990us       2.108ms         2.43%       2.142ms     214.171us           0 b           0 b     240.00 Mb           0 b            10  
Self CPU time total: 68.498ms
CUDA time total: 86.724ms

### Single big pass
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
              FAST PASS         4.01%       1.307ms        98.00%      31.923ms       3.192ms     220.777us         0.38%      57.963ms       5.796ms          40 b      -2.62 Kb       2.34 Gb    -240.00 Mb            10  
           aten::einsum        11.86%       3.862ms        87.09%      28.369ms     567.383us     799.177us         1.38%      55.301ms       1.106ms           0 b           0 b       2.23 Gb           0 b            50  
              aten::bmm         5.47%       1.780ms         7.05%       2.298ms      45.959us      42.508ms        73.20%      42.508ms     850.166us           0 b           0 b       1.05 Gb           0 b            50  
          aten::reshape         1.41%     459.138us        11.67%       3.803ms      38.025us     199.638us         0.34%      11.076ms     110.755us           0 b           0 b       1.17 Gb           0 b           100  
            aten::clone         3.49%       1.137ms         9.00%       2.933ms      48.882us     349.431us         0.60%      10.876ms     181.265us           0 b           0 b       1.17 Gb           0 b            60  
            aten::copy_         2.38%     776.662us         2.38%     776.662us      12.944us      10.526ms        18.13%      10.526ms     175.441us           0 b           0 b           0 b           0 b            60  
              aten::div         3.65%       1.189ms         4.12%       1.341ms     134.104us       1.575ms         2.71%       1.575ms     157.491us           0 b           0 b     240.00 Mb           0 b            10  
          aten::permute        51.71%      16.844ms        52.38%      17.062ms      68.250us     918.118us         1.58%     918.118us       3.672us           0 b           0 b           0 b           0 b           250  
          aten::softmax         0.29%      93.318us         2.42%     786.787us      78.679us      20.485us         0.04%     866.307us      86.631us           0 b           0 b     120.00 Mb           0 b            10  
         aten::_softmax         1.62%     527.630us         2.13%     693.469us      69.347us     834.557us         1.44%     845.822us      84.582us           0 b           0 b     120.00 Mb           0 b            10  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 32.574ms
CUDA time total: 58.074ms


### Conclusion
Clearly the single big pass has an edge in performance. It manages to reduce CUDA time to 67% but it 
carries a heavy penalty on GPU memory. 

