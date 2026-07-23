# 如意智学 RuyiTutor

面向中小学教学的 Dify 智能知识库辅导助手。项目对应生产实习 **大模型-RAG 选题 3**，首期聚焦初中数学与物理，强调“有依据地教、分步骤地引导、可量化地评估”。

## 项目亮点

- **双引擎可用**：配置 Dify 后调用真实知识库应用；未配置时自动使用本地混合 RAG，答辩现场不怕断网或额度问题。
- **真实模型与向量检索**：已接通 DeepSeek API，并使用本地 BGE-small-zh-v1.5 + FAISS 完成中文语义检索；失败时仍可降级到词法检索。
- **GraphRAG 思路**：79 个可检索知识切片，关键词召回后按知识图谱补充前置概念、易错点和关联知识。
- **教学式回答**：不直接堆答案，输出“思路—步骤—自检—拓展”，并附可核验资料来源。
- **个性化学情**：按知识点记录正确率和掌握度，自动推荐下一学习内容。
- **可评价**：内置检索命中率、引用完整率、拒答安全性等自动测试，技术结果不只靠截图。
- **完整产品形态**：学生工作台、知识图谱、学习驾驶舱三种视图，适合 10–15 分钟现场演示。
- **教师质量闭环**：班级薄弱点、低置信度记录、回答反馈和真实学习路径。
- **图片问题入口**：支持本地 OCR，可在未安装 OCR 时手动校正题干后继续 RAG。
- **防幻觉核验**：检索证据覆盖、算式与单位检查，低证据回答自动降置信度。
- **自适应练习闭环**：智能选题、提示次数、错因分类、变式练习、错题本与遗忘衰减掌握度。
- **混合检索可解释**：词法、字符语义、标题与图谱扩展分别计分，引用可追溯到分项得分。
- **知识治理可落地**：教师可新增结构化条目、自动生成版本、即时重建本地索引并处理质量问题。
- **工程可靠性**：SQLite 迁移、WAL 并发、备份脚本、流式接口、输入限长与 17 项自动测试。

## 快速启动

```powershell
cd 3\RuyiTutor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

浏览器访问 <http://127.0.0.1:8000>。无需 API Key 即可体验本地 RAG。也可直接运行 `run.ps1`，或使用 `docker compose up --build -d`。

## 接入 Dify

复制 `.env.example` 为 `.env`，填写：

```env
DIFY_API_BASE_URL=http://localhost/v1
DIFY_APP_API_KEY=app-xxx
DIFY_DATASET_API_KEY=dataset-xxx
DIFY_DATASET_NAME=如意智学-初中知识库
```

在 Dify 中创建聊天助手并绑定知识库，推荐使用混合检索、Top K=5、开启重排序。系统提示词可直接使用 `dify/system_prompt.md`。上传知识库：

```powershell
python scripts/upload_to_dify.py
```

脚本会自动创建知识库并上传基础文档及结构化课程条目。上传结束后，把输出的 `dataset_id` 写入 `.env` 的 `DIFY_DATASET_ID`。工作流节点配置见 `dify/workflow_blueprint.json`。

## 项目结构

```text
RuyiTutor/
├─ app.py                    FastAPI 接口与静态站点
├─ ruyitutor/                RAG、图谱、学情与 Dify 适配器
├─ knowledge_base/           可追溯教学资料与知识图谱
├─ web/                      产品化学生端
├─ scripts/                  Dify 批量建库脚本
├─ tests/                    自动化质量验证
├─ data/                     SQLite 学习档案（运行后生成）
└─ docs/                     答辩、架构、部署与交付材料
```

## 测试

```powershell
python -m unittest discover -s tests -v
python evaluation/run_evaluation.py
python evaluation/run_ceval.py
python scripts/backup_database.py
```

演示账号：`demo-student`，密码：`ruyi-demo-2026`。生产部署前必须更换默认密码。

公开评测采用 C-Eval 的初中数学、初中物理验证集（CC BY-NC-SA 4.0），共 38 题；当前 `deepseek-v4-flash` 实测数学 78.95%、物理 100%、整体 89.47%。教育部 2022 年数学与物理课程标准原始 PDF 已保存为权威数据源，但因官方文件为扫描版，未完成 OCR 前不会冒充已入库。

## 演示问题

1. `为什么负负得正？请不要直接给结论，带我一步一步理解。`
2. `一元一次方程 3x-5=10 怎么解？`
3. `速度和平均速度有什么区别？`
4. `我还需要先学会哪些知识，才能理解勾股定理？`

资料为团队按课程标准自行编写的示例知识条目，仅用于教学演示，不替代学校教师的正式教学。
