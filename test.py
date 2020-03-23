import torch

if __name__ == '__main__':
    a = torch.Tensor([[[1,2],[2,4],[3,5]],
                      [[2,2],[3,3],[4,4]],
                      [[4,5],[5,4],[6,5]]])
    print(a.size())
    #indexs = torch.LongTensor([[0],[1],[2]])
    print(a[[0,1,2],[1,2,2]])