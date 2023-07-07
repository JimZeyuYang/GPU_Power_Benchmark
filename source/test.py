import subprocess

def is_number(s):
    try:    float(s)
    except ValueError: return False
    return True

pwr_draw_options = {
    'power.draw': False,
    'power.draw.average': False,
    'power.draw.instant': False
}

print(pwr_draw_options)

query_options = '--query-gpu='

output = subprocess.run(['nvidia-smi', '--help-query-gpu'], stdout=subprocess.PIPE)
output = output.stdout.decode()

for key, value in pwr_draw_options.items():
    if bool(output.find(key)):
        query_options += key + ','


output = subprocess.run(['nvidia-smi', query_options, '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
output = output.stdout.decode()[:-1].split(', ')

for i, (key, value) in enumerate(pwr_draw_options.items()):
    pwr_draw_options[key] = is_number(output[i])


print(pwr_draw_options)


        
