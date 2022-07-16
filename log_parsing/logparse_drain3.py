import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from drain3.file_persistence import FilePersistence
from os.path import dirname

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

grouped = defaultdict(lambda: [])
output_numeric = True

per_user = defaultdict(lambda: 0)
per_user_jid = {}

for line in sys.stdin:
    try:
        line = line.rstrip()
        user, line = line.split(maxsplit=1)
        jid, line = line.split(maxsplit=1)
        month, line = line.split(maxsplit=1)
        day, line = line.split(maxsplit=1)
        ltime, line = line.split(maxsplit=1)
        node, line = line.split(maxsplit=1)
        if ': ' in line:
            executable, line = line.split(': ', 1)
        else:
            executable, line = line[:-1], ''
        per_user[user]+=1
        per_user_jid[jid] = user
    except ValueError:
        print(line)
        exit(1)

    ldate = datetime.strptime(f"2022 {month} {day} {ltime}", '%Y %b %d %H:%M:%S')

    result = template_miner.add_log_message(line)
    line_count += 1
    if line_count % batch_size == 0:
        time_took = time.time() - batch_start_time
        rate = batch_size / time_took
        logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                    f"{len(template_miner.drain.clusters)} clusters so far.")
        batch_start_time = time.time()
    if result["change_type"] != "none":
        result_json = json.dumps(result)
        logger.debug(f"Input ({line_count}): " + line)
        logger.debug("Result: " + result_json)
    template = result["template_mined"]
    cluster_id = result["cluster_id"]
    #print(f"Matched template #{result.cluster_id}: {template}")
    # {template_miner.get_parameter_list(template, line)}
    #print(f"{cluster_id}", end=' ')

    if output_numeric:
        grouped[jid].append(cluster_id)
    else:
        pass
    #print(f"{round(time.mktime(ldate.timetuple()))},{cluster_id},{node}")

time_took = time.time() - start_time
rate = line_count / time_took
logger.info(f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.drain.clusters)} clusters")

sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
for cluster in sorted_clusters:
    logger.info(cluster)

#print("Prefix Tree:")
#template_miner.drain.print_tree()

#template_miner.profiler.report(0)

logger.info(sorted(per_user.items(), key=lambda p : p[1]))

i = 0
if output_numeric:
    for k, v in grouped.items():
        print(k, end='|')
        for jid in v:
            print(jid, end=' ')
            i += 1
        print()
else:
    pass

logger.info(i)
logger.info(len(grouped))