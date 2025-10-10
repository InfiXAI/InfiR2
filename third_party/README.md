
内容：
1. 使用git submodule安装库，可保证库的

要求
1. 如果还有引其他库，可以也通过下面命令, 将库放在third_party下，然后软连接到具体文件夹下

```bash
git submodule add https://github.com/NVIDIA/TransformerEngine.git third_party/transformer_engine

ln -s ../third_party/transformer_engine ./transformer_engine
```