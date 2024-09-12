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

# a = [1, 2]
# b = a
# b[0] = 3
# print(a)

# import torch

# # 假设的维度和数据
# B, H, N, C = 2, 3, 5, 4  # 示例维度
# key_states = torch.randn(B, H, N, C)  # 随机生成的key_states数据
# # decision = torch.tensor([True, False, True, False, True])  # 示例决策矩阵
# decision = torch.tensor([False, False, False, False, False])  # 示例决策矩阵

# # # 取反决策矩阵
# # inverse_decision = ~decision

# # 扩展decision的维度来匹配key_states的前三个维度
# expanded_decision = decision.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(B, H, -1, C)

# # 使用布尔索引进行选择
# selected_key_states = key_states[expanded_decision].view(B, H, -1, C)

# # 打印结果以确认
# print(selected_key_states.shape)  # 应当是 [B, H, N1, C] 其中N1是decision中False的数量
# print(selected_key_states)

import bisect


print(bisect.bisect_right([0, 2000, 4000], 0))
