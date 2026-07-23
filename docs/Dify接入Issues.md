# Dify真实接入 Issues

## Issue 1：现有接入审计

- [x] Dify聊天API适配器
- [x] Dify数据集检索API适配器
- [x] Dify Vision文件上传适配器
- [x] 本地GraphRAG自动降级
- [x] 确认当前缺少Dify实例及应用/知识库密钥

## Issue 2：配置与连接诊断

- [x] `.env`自动加载
- [x] 不泄露密钥的配置状态接口
- [x] Dify `/info`在线探测
- [x] 健康接口显示Dify就绪状态
- [x] 填写真实 `DIFY_API_BASE_URL`
- [x] 填写 `DIFY_APP_API_KEY`
- [x] 填写 `DIFY_DATASET_API_KEY`

## Issue 3：知识库同步

- [x] 自动创建或复用知识库
- [x] 上传Markdown原始文档
- [x] 上传结构化概念、例题、易错点和练习
- [x] 同名文档幂等跳过
- [x] `--dry-run`与`--force`
- [x] 对真实Dify执行同步并记录dataset_id
- [x] 25个Dify文档承载79个逻辑切片，全部索引完成
- [x] 5个典型问题全部Top-1命中正确知识

## Issue 4：应用问答闭环

- [x] 学科、年级、辅导模式传入Dify
- [x] 解析Dify检索引用
- [x] 置信度与证据核验
- [x] Dify故障降级
- [x] 真实Dify Chatflow已发布并通过端到端问答验收

## Issue 5：答辩证据

- [x] Dify应用编排截图
- [ ] 知识库文档与分段截图
- [x] API在线状态与端到端运行结果
- [x] Dify与本地GraphRAG均保留并可自动降级
