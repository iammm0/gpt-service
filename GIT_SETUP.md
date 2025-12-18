# Git 仓库配置说明

## 已完成的操作

### gnn-service
- ✅ 已添加远程仓库：`https://github.com/iammm0/gnn-service.git`
- ⚠️ 需要拉取远程最新代码（网络连接后执行）

### gpt-service
- ✅ 已初始化 Git 仓库
- ✅ 已添加所有文件并提交
- ✅ 已添加远程仓库：`https://github.com/iammm0/gpt-service.git`
- ✅ 已设置主分支为 `main`
- ⚠️ 需要推送到远程仓库（网络连接后执行）

## 后续操作步骤

### 1. 拉取 gnn-service 最新代码

```bash
cd gnn-service
git fetch origin
git pull origin main
# 如果有冲突，需要解决冲突后再提交
```

### 2. 推送 gpt-service 到远程仓库

```bash
cd gpt-service
git push -u origin main
```

如果遇到认证问题，可能需要：
- 配置 GitHub 个人访问令牌（Personal Access Token）
- 或使用 SSH 方式连接

### 3. 配置 GitHub 认证（如果需要）

#### 使用 Personal Access Token

1. 访问 GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. 生成新 token，勾选 `repo` 权限
3. 推送时使用 token 作为密码

#### 使用 SSH（推荐）

```bash
# 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 将公钥添加到 GitHub
# 然后修改远程仓库地址为 SSH 格式
cd gpt-service
git remote set-url origin git@github.com:iammm0/gpt-service.git

cd ../gnn-service
git remote set-url origin git@github.com:iammm0/gnn-service.git
```

## 验证配置

### 检查远程仓库配置

```bash
# gnn-service
cd gnn-service
git remote -v
# 应该显示：origin  https://github.com/iammm0/gnn-service.git

# gpt-service
cd gpt-service
git remote -v
# 应该显示：origin  https://github.com/iammm0/gpt-service.git
```

### 检查提交状态

```bash
# gpt-service
cd gpt-service
git log --oneline
# 应该显示初始提交

# gnn-service
cd gnn-service
git status
# 查看当前状态
```

## 注意事项

1. **gnn-service** 的远程仓库有最新代码，拉取时注意：
   - 如果有本地修改，先提交或暂存
   - 如果有冲突，需要手动解决

2. **gpt-service** 是空项目，直接推送即可

3. 如果网络不稳定，可以：
   - 使用代理
   - 使用 SSH 方式
   - 分批推送大文件

## 当前状态

- ✅ gnn-service 远程仓库已配置
- ✅ gpt-service 本地仓库已初始化并提交
- ✅ gpt-service 远程仓库已配置
- ⏳ 等待网络连接后执行推送/拉取操作

