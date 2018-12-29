import re


def transfer_items(src, dest):
    for k, v in src.items():
        dest[k] = src[k]


def overwrite_cfg_file(cfg_file, ow_str='None', ow_file='None', new_cfg_file='None'):
    """Overwrite some items of a EasyDict defined config file.
    Args:
        cfg_file: The original config file
        ow_str: Mutually exclusive to ow_file. Specify the new items (separated by ';') to overwrite.
            E.g. "cfg.model = 'ResNet-50'; cfg.im_mean = (0.5, 0.5, 0.5)".
        ow_file: A text file, each line being a new item.
        new_cfg_file: Where to write the updated config. If 'None', overwrite the original file.
    """
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
    if ow_str != 'None':
        cfgs = ow_str.split(';')
        cfgs = [cfg.strip() for cfg in cfgs if cfg.strip()]
    else:
        with open(ow_file, 'r') as f:
            cfgs = f.readlines()
        # Skip empty or comment lines
        cfgs = [cfg.strip() for cfg in cfgs if cfg.strip() and not cfg.strip().startswith('#')]
    for cfg in cfgs:
        key, value = cfg.split('=')
        key = key.strip()
        value = value.strip()
        pattern = r'{}\s*=\s*(.*?)(\s*)(#.*)?(\n|$)'.format(key.replace('.', '\.'))
        def func(x):
            # print(r'=====> {} groups, x.groups(): {}'.format(len(x.groups()), x.groups()))
            # x.group(index), index starts from 1
            # x.group(index) may be `None`
            # x.group(4) is either '\n' or ''
            return '{} = {}'.format(key, value) + (x.group(2) or '') + (x.group(3) or '') + x.group(4)
        new_lines = []
        for line in lines:
            # Skip empty or comment lines
            if not line.strip() or line.strip().startswith('#'):
                new_lines.append(line)
                continue
            line = re.sub(pattern, func, line)
            new_lines.append(line)
        lines = new_lines
    if new_cfg_file == 'None':
        new_cfg_file = cfg_file
    with open(new_cfg_file, 'w') as f:
        # f.writelines(lines)  # Same effect
        f.write(''.join(lines))