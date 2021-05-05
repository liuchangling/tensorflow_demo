# 量化的步骤
1. 从一个灵感开始
2. 把灵感细化成明确的可执行的交易策略
3. 把策略转成程序
4. 检验策略效果

# 日志

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)