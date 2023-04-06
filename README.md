```
Tài liệu tham khảo
```
1. https://leimao.github.io/blog/PyTorch-Distributed-Training/
2. https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


Lưu ý cần khi gặp lỗi về nccl thì cần phải thêm dòng sau vào .bashrc

```
NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp"
```