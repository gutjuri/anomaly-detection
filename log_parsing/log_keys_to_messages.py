import logging
import sys
import time
from drain3.file_persistence import FilePersistence

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(message)s')

config = TemplateMinerConfig()
config.load("drain3.ini")
config.profiling_enabled = False
persistence = FilePersistence("drain3_state.bin")
template_miner = TemplateMiner(persistence, config=config)

line_count = 0

start_time = time.time()
batch_start_time = start_time
batch_size = 10000

for line in sys.stdin:
    try:
        line = line.rstrip()
        _, line = line.split('|', maxsplit=1)
        tokens = line.split()
        for t in tokens:
            print(template_miner.drain.id_to_cluster[int(t)])
        print()
        print("-------------")
        print()
        
    except ValueError:
        print(line)
        exit(1)
