import logging
import os

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # 使用 tqdm 的 write 方法来输出
            self.flush()
        except Exception:
            self.handleError(record)

def init_logger(args):
    logging.basicConfig(level=logging.DEBUG, handlers=[])

    try:
        filename = args.log_file
    except Exception:
        filename = "result.log"

    # 添加一个 handler 输出到文件
    file_handler = logging.FileHandler(os.path.join(args.result_dir, filename))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # 添加一个兼容 tqdm 的 handler 输出到控制台
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setLevel(logging.INFO)
    tqdm_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(tqdm_handler)
