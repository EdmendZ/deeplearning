# Git 仓库修复记录

> 修复日期：2026-06-02
> 仓库：https://github.com/EdmendZ/deeplearning.git

## 一、问题描述

每次 `git push` 都会失败，因为仓库中包含 230MB 的大文件（GitHub 单文件限制 100MB），且 `.gitignore` 完全不起作用 —— 写了规则却没生效。

## 二、根本原因

### 1. `.gitignore` 编码错误（核心原因）

文件被 Windows 记事本以 **UTF-16 LE** 编码保存，但 git 只识别 **UTF-8**。

验证方法：

```bash
file .gitignore
# 输出：Unicode text, UTF-16, little-endian text, with CRLF line terminators
```

文件二进制内容显示每个字符之间都有 `\x00` 空字节，git 把规则全部当成乱码丢弃，所以所有忽略规则都失效了。

### 2. 大文件已被 git 追踪

`.gitignore` **只对未追踪的新文件有效**，对已被 `git add` 过的文件不生效。

历史中的大文件：

| 文件 | 大小 |
|------|------|
| `dl_tutorial/data/fashion-mnist_train.csv` | 133 MB |
| `dl_tutorial/data/train.csv` | 76 MB |
| `dl_tutorial/data/fashion-mnist_test.csv` | 22 MB |

### 3. 历史提交中残留大文件

即使在新提交中 `git rm --cached` 移除，**旧提交的 tree 中仍然引用大文件 blob**，push 时仍会上传所有历史对象。

## 三、修复步骤

### 第一步：修复 `.gitignore` 编码

使用 Python 强制以 UTF-8（无 BOM）写入：

```python
content = '''# Data files
*.csv
*.txt
# ... 其他规则
'''
with open('.gitignore', 'w', encoding='utf-8', newline='') as f:
    f.write(content)
```

验证编码：

```bash
python -c "print('Has null bytes:', b'\x00' in open('.gitignore', 'rb').read())"
# 输出：Has null bytes: False ✅
```

### 第二步：从 git 追踪中移除已追踪文件

`.gitignore` 对已追踪文件无效，必须手动移除：

```bash
git rm --cached -r .idea/
git rm --cached -r dl_tutorial/.idea/
git rm --cached "*.ipynb"
git rm --cached -r dl_tutorial/data/
```

> `--cached` 只从 git 索引移除，**本地文件保留**。

### 第三步：清理 git 历史

`git rm --cached` 只让新提交不追踪，但旧提交仍引用大文件。必须**重写历史**：

```bash
# 创建孤儿分支（无历史记录）
git checkout --orphan clean

# 提交当前干净的工作树
git commit -m "Initial commit"

# Force push 覆盖远程历史
git push --force origin clean:main

# 清理本地旧对象
git reflog expire --expire=now --all
git gc --aggressive --prune=now
```

### 第四步：重建有意义的提交历史

按章节拆分成 7 个清晰的 commit：

```
* Add .gitignore configuration
* Ch08-09: CNN and RNN architectures
* Ch06-07: PyTorch fundamentals and deep learning
* Ch04-05: Backpropagation and optimizer comparison
* Ch03: Training neural networks with gradient descent
* Ch02: Build basic neural network from scratch
* Project setup: common neural network utilities
```

## 四、修复结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 仓库 pack 大小 | 51.53 MiB | **29.41 KiB** |
| 追踪文件数 | 72 个 | **41 个** (仅源代码) |
| 大文件 | 3 个（230 MB） | **0 个** |
| `.gitignore` 编码 | UTF-16 LE ❌ | UTF-8 ✅ |
| Commit 历史 | 4 个混乱提交 | **7 个语义化提交** |
| Push 状态 | 失败 | **成功** ✅ |

## 五、最终 `.gitignore` 内容

```gitignore
# Data files (large CSV, etc.)
*.csv
*.txt
*.log

# Python
*.pyc
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/

# Jupyter Notebook
*.ipynb
.ipynb_checkpoints/

# macOS
*.DS_Store

# Environment
.env
.env.local

# Coverage
.coverage
.coverage.*

# Cache
.cache/

# IDE
.idea/
*.swp
*.swo

# Model files (large binary files)
*.pth
*.pt
*.pkl
*.h5
*.hdf5
*.ckpt
checkpoints/

# Data directories
data/
datasets/
```

## 六、经验教训

### 1. Windows 上的 `.gitignore` 必须是 UTF-8
不要用记事本（默认 UTF-16 或带 BOM）创建/编辑 `.gitignore`。推荐：
- VS Code / PyCharm（默认 UTF-8）
- 命令行：`echo "*.csv" > .gitignore`（bash/git-bash）
- Python：明确指定 `encoding='utf-8'`

### 2. 检查文件编码的方法

```bash
file .gitignore              # 查看编码类型
python -c "print(open('.gitignore','rb').read()[:30])"  # 看前 30 字节
git check-ignore -v <file>   # 验证规则是否生效
```

### 3. `.gitignore` 对已追踪文件无效
**先 `.gitignore`，再 `git add`** —— 顺序很重要。如果已经误追踪：

```bash
git rm --cached <file>       # 从追踪中移除（本地保留）
git rm --cached -r <dir>/    # 移除整个目录
```

### 4. 清理历史中的大文件
`git rm --cached` 只能让新提交不带大文件，**历史中的大文件仍存在**。彻底清除方案：
- **简单粗暴**：`git checkout --orphan` + force push（适合个人项目）
- **保留历史**：`git filter-repo --strip-blobs-bigger-than 100M`（推荐）
- **官方工具**：BFG Repo-Cleaner

### 5. Force push 的风险
本次使用了 `git push --force`，**会覆盖远程所有历史**。仅适用于：
- 个人项目
- 没有其他协作者
- 你完全确定不需要旧历史

团队项目中应使用 `--force-with-lease` 并提前通知协作者。

## 七、相关命令速查

```bash
# 检查编码
file .gitignore

# 列出已追踪的所有文件
git ls-files --cached

# 查看仓库大小
git count-objects -vH

# 找出历史中的大文件
git ls-tree -rl HEAD | awk '$4 > 1048576 {print $4, $5}' | sort -rn

# 验证 gitignore 规则
git check-ignore -v <file>

# 列出未追踪但会被忽略的文件
git status --ignored
```
