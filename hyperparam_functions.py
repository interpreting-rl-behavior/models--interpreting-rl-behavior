import yaml, munch
import os, shutil

def load_interp_configs(configs_name):
    print('[Loading interpretation hyperparameters]')
    with open('hyperparams/interpreting_configs.yml', 'r') as f:
        hp = yaml.safe_load(f)[configs_name]
    for key, value in hp.items():
        print(key, ':', value)
    hp = munch.munchify(hp)
    return hp

def save_interp_configs_file(uniq_id):
    config_file_str = 'interpreting_configs.yml'
    config_file_path = os.path.join('hyperparams', config_file_str)
    end = config_file_str.find('.yml')
    uniq_config_name = config_file_str[:end] + '_' + str(uniq_id) + config_file_str[end:]
    save_dir = 'hyperparams/hp_saves'
    os.makedirs(save_dir, exist_ok=True)
    uniq_config_name = os.path.join(save_dir, uniq_config_name)
    shutil.copy(config_file_path, uniq_config_name)
    print("Config file saved for your records.")

