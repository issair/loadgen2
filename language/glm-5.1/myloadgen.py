"""
本测试程序测试


假设本测试启动10个虚拟用户：


1.假设本系统有 2000 虚拟用户。
2.系统通道并发虚拟用户有10个。


先对2000个用户做执行的采用的 泊松分布

[vu1] : [user1+tool] - 10s - [user2+tool] - 10s - [user3+tool]
[vu2] : [user1+tool] - 13s - [user2+tool] - 17s - [user3+tool]
...
[vun] : [user1+tool] - 19s - [user2+tool] - 1s - [user3+tool] ...

接着，创建 10个 执行通道
           [        10s     ]   [      20s       ]
[slot01] : [vu001:user1+tool]   [vu001:user2+tool]
[slot02] : [vu002:user1+tool]   [vu002:user2+tool]
[slot03] : [vu003:user1+tool]   [vu003:user2+tool]
[slot04] : [vu006:user1+tool]   [vu004:user2+tool]
[slot05] : [vu007:user1+tool       ] [vu006:user2+tool]
[slot06] : [vu101:user1+tool]
[slot07] : [vu202:user1+tool]
[slot08] : [vu201:user1+tool]
[slot09] : [vu205:user1+tool]
[slot10] : [vu401:user1+tool]
"""
