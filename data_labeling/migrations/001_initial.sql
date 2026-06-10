-- 001_initial.sql —— 记录 data_labeling/db.py 当前 schema（Section 6 G）
-- 见 data_labeling/db.py 模块顶部 docstring 的 Schema 段。
-- 本文件是"已应用"的初始 schema，不做事（建表由 db.py 的 init_db 负责），
-- 仅作为迁移框架的"第 1 个迁移"占位 + 启动历史记录的起点。

-- 验证当前 schema 是否存在的 SQL（不修改 schema）：
SELECT name FROM sqlite_master
WHERE type IN ('table', 'index')
ORDER BY name;
