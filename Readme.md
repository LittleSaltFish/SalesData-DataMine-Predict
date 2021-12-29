# 数据清洗

1. 数据总览：`pandas_profiling`
2. 为所有表添加 `result`项以便分析，`result`项补零

# 结论

1. 表 `user_info`的城市数据缺失，需要填补
2. 表 `user_info`的 `app_num`为常量1，可以直接删除此列
3. 各表之间契合关系较好，除 `city`项外几乎无缺失数据

# 效果较好的模型

- 去空+`sampling_strategy`=0.05  ~0.72
  - 加载模型 - 去空全预测结果为0.71
  - 加载模型 - 不去空全预测结果为0.73
- 不去空+不调整倾斜数据~0.74
  - 加载模型 - 全预测结果均为0.74

# 模型融合

随机森林准确率高，值得考虑

逻辑回归召回率高但准确率低，谨慎考虑

决策树两样均低，不应考虑

一分类效果奇差，且运行速度极慢

# 问题记录

## Fail to allocate bitmap、MemoryError: In RendererAgg: Out of memory

- 问题简述：

  - 在使用 `pandas_profiling`分析 `visit_info`数据时，出现 `Fail to allocate bitmap`错误
  - 在使用 `pairplot`制图时，出现 `MemoryError: In RendererAgg: Out of memory`错误
- 问题原因：内存耗尽
- 解决方案：

  - 方案1：关闭chrmoe、vscode等程序后使用cmd运行python文件，成功导出分析文件。
  - 方案2：不放回的抽取若干数据（1%）作为示例进行展示
  - 方案3（仅问题2）：每次循环添加 `plt.cla()`，以清除当前ax对象。
- 参考链接：[ref](https://www.pythonpool.com/python-memory-error/)
