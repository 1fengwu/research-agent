本目录是 AutoResearch 角色 B 的固定 seed 演示数据。

- demo_panel.csv：完整演示面板，14 个标的，320 个交易日；固定 seed=42。
- demo_panel.csv.sha256：文件字节的 sha256；运行期内容指纹使用 canonical_panel_hash() 计算。
- demo_split.json：固定 seed=42 的时间切分、指纹与分段日期。
- demo_profile.json：固定演示面板的数据画像。
- sample.csv：5 个标的、80 个交易日的小样本，供 D 前端联调。
