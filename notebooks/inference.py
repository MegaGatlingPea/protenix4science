import sys
sys.path.append('/home/jin_zi_chang/protenix4science')
from protenix.data.utils import convert_bb_configs

output_json = convert_bb_configs('/home/jin_zi_chang/protenix4science/examples/bb_generation.json',dump=True)

print(f"output_json: {output_json}")



