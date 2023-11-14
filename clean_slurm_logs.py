# %%
import os
# %%

for file_name in os.listdir('.'):
    if file_name.startswith('slurm-'):
        print(f'Removing {file_name}')
        os.remove(file_name)

# %%
