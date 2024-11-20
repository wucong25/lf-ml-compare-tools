import argparse
import re
import json


# Modellink参数与Llamafactory参数的映射关系
args_mappings = {
    'rotary_base': 'rope_theta',
    'norm_epsilon': 'rms_norm_eps',
    'adam_beta1': 'adam_beta1',
    'adam_beta2': 'adam_beta2',
    'train_iters': 'max_steps',
    'num_layers': 'num_hidden_layers',
    'init_method_std': 'initializer_range',
    'attention_dropout':'attention_dropout',
    'weight_decay': 'weight_decay',
    'micro_batch_size': 'per_device_train_batch_size',
    'lr': 'learning_rate',
    'lr_warmup_fraction': 'warmup_ratio',
    'seed': 'seed',
    'bf16': 'bf16',
    'fp16': 'fp16',
    'world_size': 'world_size'
}

args_int_list1 = ['train_iters', 'num_layers', 'micro_batch_size', 'seed', 'world_size', 'global_batch_size']
args_int_list2 = ['max_steps', 'num_hidden_layers', 'per_device_train_batch_size', 'seed', 'world_size', 'gradient_accumulation_steps']

args_int_list_all = args_int_list1 + args_int_list2

args_float_list1 = ['rotary_base', 'norm_epsilon', 'adam_beta1', 'adam_beta2', 'init_method_std', 'attention_dropout', 'weight_decay', 'lr', 'lr_warmup_fraction']
args_float_list2 = ['rope_theta', 'rms_norm_eps', 'adam_beta1', 'adam_beta2', 'initializer_range', 'attention_dropout', 'weight_decay', 'learning_rate', 'warmup_ratio']

args_float_list_all = args_float_list1 + args_float_list2


args_bool_list1 = ['bf16', 'fp16', 'no_shuffle', 'use-deter-comp']
args_bool_list2 = ['bf16', 'fp16']

args_bool_list_all = args_bool_list1 + args_bool_list2


def get_argument_by_type(key, value):
    global args_int_list_all
    global args_float_list_all
    global args_bool_list_all
    if str(key) in args_int_list_all:
        if value == 'None':
            return 0
        return int(value)
    
    if str(key) in args_float_list_all:
        if value == 'None':
            return 0.0
        return float(value)

    if str(key) in args_bool_list_all:
        if key == 'None':
            return False

        if isinstance(value, bool):
            return value

        value = value.capitalize()
        if value == 'True':
            return True
        else:
            return False

    return value


special_mappings = {
    'global_batch_size': 'gradient_accumulation_steps',
}


def judge_gbs_and_gas(world_size, global_batch_size, gradient_accumulation_steps, per_device_train_batch_size):
    return world_size * per_device_train_batch_size * gradient_accumulation_steps == global_batch_size


modellink_constant_args = {
    'no_shuffle': 'True',
    'split': '100,0,0',
    'use-deter-comp': 'True',
    'lr_decay_style': 'constant'
}


llamafactory_constant_args = {
    'lr_scheduler_type': 'constant',
    'max_samples': 'None',
    'stage': 'sft',
    'finetuning_type': 'full',
    'val_size': 'None',
    'per_device_eval_batch_size': 'None',
    'eval_strategy': 'None',
    'eval_steps': 'None',
}


# config与Lf之间的映射关系
hf_lf_args_mapping = {
    'rms_norm_eps': 'adam_epsilon',
    'layer_norm_epsilon': 'adam_epsilon',
    'rotary_emb_base': 'rope_theta'
}

# lf与ml映射的默认参数值
lf_mapping_default_args = {
    'rope_theta': 'None',
    'adam_beta1': '0.9',
    'adam_beta2': '0.999',
    'max_steps': '0',
    'num_hidden_layers': '0',
    'initializer_range': '0',
    'attention_dropout': '0.0',
    'weight_decay': '0.0',
    'adam_epsilon': '1e-6',
    'rms_norm_eps': '1e-6',
    'per_device_train_batch_size': '1',
    'learning_rate': '0',
    'warmup_ratio' :0,
    'seed': 'None',
    'bf16': 'False',
    'fp16': 'False',
    'world_size': '1'
}


lf_default_args = {
    'template': 'None',
    'cutoff_len': '1024',
    'adam_beta1': '0.9',
    'adam_beta2': '0.999',
}

tips_nums = 1
results = ['下面是一些对齐建议和可能需要注意的地方：\n']


def init_lf(lf_args):
    lf_args.update(lf_mapping_default_args)
    lf_args.update(lf_default_args)


def initialize_args():
    parser = argparse.ArgumentParser(description='Tools Arguments',
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--hf-config', type=str, default=None,
                       help='The path of config.json')
    parser.add_argument('--modellink-config', type=str, default=None,
                       help='The path of Modellink config.json')
    parser.add_argument('--llamafactory-config', type=str, default=None,
                       help='The path of llamafactory config.json')


    args = parser.parse_args()

    return args


def main():
    args = initialize_args()
    modellink_file_path = args.modellink_config  # 这里替换为实际的文件路径
    lf_file_path = args.llamafactory_config
    hf_config_path = args.hf_config
    modellink_args = {}
    lf_args = {}
    # 先初始化一下
    init_lf(lf_args)
    try:
        begin = False
        with open(modellink_file_path, 'r') as file:
            for line in file:
                # 在这里可以对每一行进行处理，例如打印出来
                line = line.strip()
                if line == '':
                    continue

                if 'end of ModelLink Arguments' in line:
                    print(line)
                    break

                if 'ModelLink Arguments' in line:
                    begin = True
                    print(line)
                    continue

                if begin:
                    pattern = r'\.\.+'
                    # 使用re.split来分割字符串
                    parts = re.split(pattern, line)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    modellink_args[key] = value
                    print(line)

    except FileNotFoundError:
        print(f'文件 {modellink_file_path} 不存在。')
        raise FileNotFoundError(f'{modellink_file_path} file not provided')


    try:
        # 打开JSON文件并读取其内容 需要先加载config.json里的配置 优先级更低
        with open(hf_config_path, 'r', encoding='utf-8') as file:
            # 使用json.load()将文件内容解析为Python字典
            data_dict = json.load(file)
            lf_args.update(data_dict)

            # config里的配置覆盖默认配置
            for key, value in hf_lf_args_mapping.items():
                if key in lf_args:
                    lf_args[value] = lf_args[key]
    except FileNotFoundError:
        print(f'文件 {hf_config_path} 不存在。')
        raise FileNotFoundError(f'{hf_config_path} file not provided')


    try:
        begin = False
        with open(lf_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#') or line == '':
                    continue
                parts = line.split(':')
                key = parts[0].strip()
                value = parts[1].strip()
                lf_args[key] = value
    except FileNotFoundError:
        print(f'文件 {lf_file_path} 不存在。')
        raise FileNotFoundError(f'{lf_file_path} file not provided')


    if len(modellink_args) == 0:
        raise FileNotFoundError(f'{modellink_file_path} 可能不是modellink日志')


    if len(lf_args) == 0:
        raise FileNotFoundError(f'{lf_file_path} 可能不是llamafactory日志')

    print("\n\n\n------------------------Llamafactory Arguments------------------------\n:", lf_args)
    print("\n------------------------end of Llamafactory Arguments------------------------\n")

    check_lf_constant(lf_args)
    check_ml_constant(modellink_args)

    check_lf_and_ml_mappings_args(lf_args, modellink_args)

    check_special_args(lf_args, modellink_args)

    for compare_tip in results:
        print(compare_tip)


def check_special_args(lf_args, modellink_args):
    if 'world_size' not in lf_args:
        raise ValueError('Llamafactory的参数配置中缺少world_size参数，请添加')

    if get_argument_by_type('world_size', lf_args['world_size']) != get_argument_by_type('world_size', modellink_args['world_size']):
        raise ValueError('Llamafactory与Modellink的world_size参数不一致，请设置为一致')
    world_size = int(lf_args['world_size'])
    gradient_accumulation_steps = int(lf_args['gradient_accumulation_steps'])
    per_device_train_batch_size = int(lf_args['per_device_train_batch_size'])

    global_batch_size = int(modellink_args['global_batch_size'])

    if not judge_gbs_and_gas(world_size, global_batch_size, gradient_accumulation_steps, per_device_train_batch_size):
        global tips_nums
        tip = str(tips_nums) + '、'
        tips_nums = tips_nums + 1
        tip = tip + '请检查Modellink参数【global_batch_size】和LLamafactory参数【gradient_accumulation_steps】、【per_device_train_batch_size】是否满足条件：\n\
            global_batch_size == world_size * per_device_train_batch_size * gradient_accumulation_steps \n'
        results.append(tip)


def check_lf_and_ml_mappings_args(lf_args, modellink_args):
    for key1, key2 in args_mappings.items():
        if get_argument_by_type(key1, modellink_args[key1]) != get_argument_by_type(key2, lf_args[key2]):
            global tips_nums
            tip = str(tips_nums) + '、'
            tips_nums = tips_nums + 1
            tip = tip + '请检查Modellink参数【{}:{}】和LLamafactory参数【{}:{}】是否一致\n'.format(key1, modellink_args[key1], key2, lf_args[key2])
            results.append(tip)


def check_lf_constant(lf_args):
    for key, value in llamafactory_constant_args.items():
        if key in lf_args and get_argument_by_type(key, lf_args[key]) != get_argument_by_type(key, llamafactory_constant_args[key]):
            global tips_nums
            tip = str(tips_nums) + '、'
            tips_nums = tips_nums + 1
            if llamafactory_constant_args[key] == 'None':
                tip = tip + '请检查LLamafactory参数：{llamafactory_constant_args[key]}是否已删除\n'
            else:
                tip = tip + '请检查Llamafactory参数：{llamafactory_constant_args[key]}的值是否为{llamafactory_constant_args[value]}\n'
            results.append(tip)


def check_ml_constant(modellink_args):
    for key, value in modellink_constant_args.items():
        global tips_nums
        if key not in modellink_args:
            tip = str(tips_nums) + '、'
            tips_nums = tips_nums + 1
            if key == 'no_shuffle':
                tip = tip + "当前Modellink参数中缺少{}参数，需要手动确保数据集的对齐\n".format(key)
            else:
                tip = tip + "当前Modellink参数中缺少{}参数， 需要加上,如果没有此参数请升级Modellink版本\n".format(key)
            results.append(tip)
            continue

        if get_argument_by_type(key, modellink_args[key]) != get_argument_by_type(key, modellink_constant_args[key]):
            tip = str(tips_nums) + '、'
            tips_nums = tips_nums + 1
            if modellink_constant_args[key] == 'None':
                tip = tip + '请检查Modellink参数：{modellink_constant_args[key]}是否已删除\n'
            else:
                tip = tip + '请检查Modellink参数：{modellink_constant_args[key]}的值是否为{modellink_constant_args[value]}\n'
            results.append(tip)


if __name__ == '__main__':
    main()