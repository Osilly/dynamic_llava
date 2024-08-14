# import torch

# # 假设你已经有了一个[B, N, 1]形状的hard_keep_decision矩阵
# B, N = 3, 5  # 示例：Batch size B=3, N=5
# hard_keep_decision = torch.randint(0, 2, (B, N, 1)).to(torch.float32)  # 随机生成0和1

# print("hard_keep_decision:\n", hard_keep_decision)

# # 获取keep_index和unkeep_index
# keep_index = [
#     torch.nonzero(hard_keep_decision[b, :, 0] == 1, as_tuple=False) for b in range(B)
# ]
# unkeep_index = [
#     torch.nonzero(hard_keep_decision[b, :, 0] == 0, as_tuple=False) for b in range(B)
# ]

# print(keep_index)
# a = torch.rand([B, N, 2])
# keep_a = [
#     a[b].gather(dim=0, index=keep_index[b].expand(-1, a[b].shape[-1])) for b in range(B)
# ]
# print(a)
# print(keep_a)


# # # 显示结果
# # print("\nKeep Indices per Batch:")
# # for b, idx in enumerate(keep_index):
# #     print(f"Batch {b}: {idx}")

# # print("\nUnkeep Indices per Batch:")
# # for b, idx in enumerate(unkeep_index):
# #     print(f"Batch {b}: {idx}")


# import torch

# # 假设有以下数据
# B = 2  # 批次数量
# N = 5  # 每个批次中的节点数量
# C = 3  # 特征数量

# # 模拟数据
# merge_unselect_node_index_batch_list = [
#     torch.tensor([[3], [1], [1], [3], [2]]),
#     torch.tensor([[0], [0], [1], [2], [4]]),
# ]

# unkeep_image_hidden_states_batch_list = [
#     torch.rand(N, C),
#     torch.rand(N, C),
# ]

# # 处理每个批次
# result_list = []
# for batch_idx in range(B):
#     unkeep_states = unkeep_image_hidden_states_batch_list[batch_idx]
#     indices = merge_unselect_node_index_batch_list[
#         batch_idx
#     ].squeeze()  # 移除多余的维度

#     # 创建一个结果张量，初始化为0
#     result_tensor = torch.zeros_like(unkeep_states)

#     # 对于每个索引，聚合并计算平均
#     unique_indices, counts = indices.unique(return_counts=True)
#     for idx, count in zip(unique_indices, counts):
#         # 获取所有匹配当前索引的位置
#         mask = indices == idx
#         # 聚合这些位置的向量
#         result_tensor[idx] = unkeep_states[mask].sum(0) / count

#     result_list.append(result_tensor)

# # 输出结果张量
# for i, result in enumerate(result_list):
#     print(f"Batch {i} Result:\n{result}")

a = [1, 2]
b = a
b[0] = 3
print(a)
